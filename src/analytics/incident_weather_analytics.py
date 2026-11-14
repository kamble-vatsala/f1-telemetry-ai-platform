import duckdb
import numpy as np
import pandas as pd

DB_PATH = "data/f1_lakehouse.duckdb"


def classify_finish_status(status_str: str, grid_pos: int, finish_pos: int) -> str:
    s = str(status_str).strip().upper()
    if s in ('DNS', 'DID NOT START', 'WITHDREW') or (grid_pos == 0 and 'LAP' not in s and s != 'FINISHED'):
        return 'DNS (Did Not Start)'
    if 'DISQUALIFIED' in s or s == 'DSQ':
        return 'DSQ (Disqualified)'
    if s == 'FINISHED':
        return 'Finished (Lead Lap)'
    if (s.startswith('+') and 'LAP' in s) or s == 'LAPPED':
        return 'Finished (Lapped)'
    if any(k in s for k in ['COLLISION', 'ACCIDENT', 'SPUN', 'DAMAGE', 'CRASH', 'DEBRIS']):
        return 'DNF - Crash / Collision'
    if any(k in s for k in ['ENGINE', 'GEARBOX', 'PU', 'POWER', 'HYDRAULIC', 'BRAKE', 'ELECTRICAL', 'WATER', 'OIL', 'EXHAUST', 'SUSPENSION', 'TRANSMISSION', 'FUEL', 'MECHANICAL', 'OVERHEAT']):
        return 'DNF - Mechanical / Car'
    if s in ('RETIRED', 'DNF'):
        return 'DNF - Retired'
    return f'DNF / Other ({status_str})'


def analyze_weather_vsc_and_dnfs(year: int, round_num: int, session_type: str = 'R'):
    con = duckdb.connect(DB_PATH, read_only=True)

    results_df = con.execute(f"""
        SELECT Driver, FullName, OfficialTeamName AS Team, GridPosition, FinishPosition, 
               PositionsGained, Points, Status, AvgTrackTemp, AvgAirTemp, IsWetSession,
               SC_Count, VSC_Count, RedFlag_Count
        FROM clean_results
        WHERE Year = {year} AND RoundNumber = {round_num} AND SessionType = '{session_type}'
        ORDER BY FinishPosition NULLS LAST
    """).df()

    laps_df = con.execute(f"""
        SELECT Driver, OfficialTeamName AS Team, LapNumber, Stint, Compound, TyreLife,
               Position, LapTime_Sec, TrackStatus, IsPitLap
        FROM clean_laps
        WHERE Year = {year} AND RoundNumber = {round_num} AND SessionType = '{session_type}'
        ORDER BY LapNumber, Position
    """).df()

    try:
        weather_df = con.execute(f"""
            SELECT Time_Sec, ROUND(Time_Sec / 60.0, 1) AS SessionMinute,
                   TrackTemp, AirTemp, Humidity, WindSpeed, Rainfall
            FROM read_parquet('data/raw_weather/*.parquet', union_by_name=True)
            WHERE Year = {year} AND RoundNumber = {round_num} AND SessionType = '{session_type}'
            ORDER BY Time_Sec
        """).df()
    except Exception:
        weather_df = pd.DataFrame()

    con.close()

    if results_df.empty and laps_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    laps_completed = laps_df.groupby('Driver')['LapNumber'].max().to_dict() if not laps_df.empty else {}
    max_race_laps = int(laps_df['LapNumber'].max()) if not laps_df.empty else 0

    dnf_rows = []
    for _, r in results_df.iterrows():
        drv = r['Driver']
        comp_laps = int(laps_completed.get(drv, 0))
        grid_p = int(r['GridPosition']) if pd.notna(r['GridPosition']) else 0
        fin_p = int(r['FinishPosition']) if pd.notna(r['FinishPosition']) else 20
        raw_status = str(r['Status'])

        if comp_laps == 0 and raw_status.upper() not in ('FINISHED',):
            cat = 'DNS (Did Not Start)'
        else:
            cat = classify_finish_status(raw_status, grid_p, fin_p)

        dnf_rows.append({
            'FinishPosition': fin_p,
            'Driver': drv,
            'Team': r['Team'],
            'GridPosition': grid_p,
            'LapsCompleted': comp_laps,
            'RaceCompletion_Pct': round((comp_laps / max_race_laps) * 100, 1) if max_race_laps > 0 else 0.0,
            'OfficialStatus': raw_status,
            'ReliabilityCategory': cat,
            'Points': r['Points']
        })

    dnf_summary_df = pd.DataFrame(dnf_rows)
    attrition_counts_df = (
        dnf_summary_df.groupby('ReliabilityCategory', as_index=False)
        .agg(DriverCount=('Driver', 'count'), Drivers=('Driver', lambda s: ", ".join(s)))
        .sort_values('DriverCount', ascending=False)
    )

    neutralization_rows = []
    if not laps_df.empty:
        lap_status = laps_df.groupby('LapNumber').agg(
            Has_SC=('TrackStatus', lambda s: s.astype(str).str.contains('4').any()),
            Has_Red=('TrackStatus', lambda s: s.astype(str).str.contains('5').any()),
            Has_VSC=('TrackStatus', lambda s: s.astype(str).str.contains('6|7').any()),
            MedianLap_Sec=('LapTime_Sec', 'median')
        ).reset_index()

        lap_status['NeutralizationType'] = 'GREEN'
        lap_status.loc[lap_status['Has_VSC'], 'NeutralizationType'] = 'VSC (Virtual Safety Car)'
        lap_status.loc[lap_status['Has_SC'], 'NeutralizationType'] = 'SC (Safety Car)'
        lap_status.loc[lap_status['Has_Red'], 'NeutralizationType'] = 'RED FLAG (Session Stopped)'

        neut_laps = lap_status[lap_status['NeutralizationType'] != 'GREEN'].copy()
        if not neut_laps.empty:
            neut_laps['EpisodeID'] = (neut_laps['LapNumber'].diff().fillna(1) > 2).cumsum()
            for ep_id, ep_grp in neut_laps.groupby('EpisodeID'):
                start_l = int(ep_grp['LapNumber'].min())
                end_l = int(ep_grp['LapNumber'].max())
                n_type = ep_grp['NeutralizationType'].mode().iloc[0]

                window_laps = laps_df[(laps_df['LapNumber'] >= max(1, start_l - 1)) & (laps_df['LapNumber'] <= end_l)]
                pitted_drvs = window_laps[window_laps['IsPitLap'] == True]['Driver'].unique().tolist()

                pre_lap = max(1, start_l - 1)
                post_lap = min(max_race_laps, end_l + 1)
                pre_pos = laps_df[laps_df['LapNumber'] == pre_lap].set_index('Driver')['Position'].to_dict()
                post_pos = laps_df[laps_df['LapNumber'] == post_lap].set_index('Driver')['Position'].to_dict()

                gainers, losers = [], []
                for d, p_after in post_pos.items():
                    if d in pre_pos and pd.notna(p_after) and pd.notna(pre_pos[d]):
                        diff = int(pre_pos[d]) - int(p_after)
                        if diff >= 2:
                            gainers.append(f"{d} (+{diff})")
                        elif diff <= -2:
                            losers.append(f"{d} ({diff})")

                neutralization_rows.append({
                    'Episode': int(ep_id),
                    'IncidentType': n_type,
                    'StartLap': start_l,
                    'EndLap': end_l,
                    'Duration_Laps': end_l - start_l + 1,
                    'DriversWhoPitted_CheapStop': ", ".join(pitted_drvs) if pitted_drvs else "None",
                    'BiggestPositionGainers': ", ".join(gainers[:5]) if gainers else "Minimal change",
                    'BiggestPositionLosers': ", ".join(losers[:5]) if losers else "Minimal change"
                })

    neutralization_windows_df = pd.DataFrame(neutralization_rows)

    pit_decision_rows = []
    if not laps_df.empty:
        for drv, d_laps in laps_df.sort_values('LapNumber').groupby('Driver'):
            stint_changes = d_laps[d_laps['Stint'].diff().fillna(0) > 0]
            for _, change_row in stint_changes.iterrows():
                pit_lap = int(change_row['LapNumber'])
                new_stint = int(change_row['Stint'])
                new_comp = str(change_row['Compound'])

                prev_stint_laps = d_laps[d_laps['Stint'] == new_stint - 1]
                next_stint_laps = d_laps[d_laps['Stint'] == new_stint]
                if prev_stint_laps.empty or next_stint_laps.empty:
                    continue

                old_comp = str(prev_stint_laps['Compound'].iloc[-1])

                pre_p_rows = d_laps[d_laps['LapNumber'] == max(1, pit_lap - 2)]['Position'].dropna()
                post_p_rows = d_laps[d_laps['LapNumber'] == min(max_race_laps, pit_lap + 5)]['Position'].dropna()
                pos_before = int(pre_p_rows.iloc[0]) if not pre_p_rows.empty else np.nan
                pos_after = int(post_p_rows.iloc[0]) if not post_p_rows.empty else np.nan
                net_pos_change = (pos_before - pos_after) if (pd.notna(pos_before) and pd.notna(pos_after)) else 0

                pre_pace = prev_stint_laps.tail(4)
                pre_pace = pre_pace[~pre_pace['IsPitLap'].fillna(False)]['LapTime_Sec'].median()
                post_pace = next_stint_laps.head(5)
                post_pace = post_pace[~post_pace['IsPitLap'].fillna(False)]['LapTime_Sec'].median()
                pace_improvement_sec = round(float(pre_pace - post_pace), 3) if (pd.notna(pre_pace) and pd.notna(post_pace)) else 0.0

                pit_window_status = str(change_row['TrackStatus'])
                pitted_under_neutralization = any(c in pit_window_status for c in ['4', '5', '6', '7'])

                wet_compounds = {'INTERMEDIATE', 'WET'}
                if (old_comp in wet_compounds) != (new_comp in wet_compounds):
                    transition_type = f"🌧️☀️ Weather Crossover ({old_comp} -> {new_comp})"
                elif old_comp in wet_compounds and new_comp in wet_compounds:
                    transition_type = f"🌧️ Wet-to-Wet Refresh ({old_comp} -> {new_comp})"
                else:
                    transition_type = f"☀️ Dry Strategy ({old_comp} -> {new_comp})"

                if net_pos_change >= 2 or (pace_improvement_sec >= 1.5 and net_pos_change >= 0):
                    verdict = "✅ Masterstroke (+Gain / Faster)"
                elif net_pos_change >= 0 and pace_improvement_sec >= 0.2:
                    verdict = "👍 Worth It (Held Pos & Faster)"
                elif net_pos_change == -1:
                    verdict = "⚖️ Neutral Cycle (-1 Pos)"
                else:
                    verdict = f"❌ Costly Undercut ({net_pos_change:+d} Pos Lost)"

                pit_decision_rows.append({
                    'Driver': drv,
                    'Team': change_row['Team'],
                    'PitLap': pit_lap,
                    'Transition': transition_type,
                    'PittedUnder_SC_VSC': "Yes (Cheap Stop!)" if pitted_under_neutralization else "No (Green Flag)",
                    'Pos_BeforePit': pos_before,
                    'Pos_5LapsAfter': pos_after,
                    'NetPositions_5Laps': net_pos_change,
                    'PaceDelta_AfterPit_Sec': -pace_improvement_sec,
                    'DecisionVerdict': verdict
                })

    pit_decisions_df = pd.DataFrame(pit_decision_rows).sort_values(['PitLap', 'Pos_BeforePit']) if pit_decision_rows else pd.DataFrame()

    total_dnf = len(dnf_summary_df[dnf_summary_df['ReliabilityCategory'].str.startswith('DNF')])
    total_dns = len(dnf_summary_df[dnf_summary_df['ReliabilityCategory'].str.startswith('DNS')])
    masterstrokes = len(pit_decisions_df[pit_decisions_df['DecisionVerdict'].str.startswith('✅')]) if not pit_decisions_df.empty else 0

    notes = {
        "f1_context": (
            "Races are frequently decided not by raw car speed, but by Race Control neutralizations (Safety Car, "
            "Virtual Safety Car, Red Flags), reliability (DNFs/DNS), and Weather Crossover timing. Pitting under a "
            "VSC or Safety Car saves ~10–12 seconds ('Cheap Pit Stop') because the field is restricted to a reduced "
            "delta speed. In wet-to-dry transitions, pitting 1 lap too early onto dry slicks can cause a driver to "
            "aquaplane off track, while pitting 1 lap too late costs 3–5 seconds per lap."
        ),
        "ds_methodology": (
            "Event-Window Counterfactual Analysis: To evaluate whether a strategic decision ('Intervention') was "
            "worth it, we compare the pre-intervention state [Lap_t - 2] against the post-intervention equilibrium "
            "[Lap_t + 5] after pit-cycle transients settle, measuring both ΔPosition and ΔMedianLapTime."
        ),
        "key_insight": (
            f"Session Reliability: {len(dnf_summary_df) - total_dnf - total_dns} classified finishers, "
            f"{total_dnf} DNFs, and {total_dns} DNS. "
            f"Detected {len(neutralization_windows_df)} SC/VSC/Red Flag episodes and evaluated "
            f"{len(pit_decisions_df)} tyre change decisions ({masterstrokes} rated as Masterstroke gambles)."
        )
    }

    return dnf_summary_df, attrition_counts_df, neutralization_windows_df, pit_decisions_df, weather_df, notes