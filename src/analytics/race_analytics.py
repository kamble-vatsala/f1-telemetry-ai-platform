import duckdb
import numpy as np
import pandas as pd

DB_PATH = "data/f1_lakehouse.duckdb"


def analyze_race_flow_and_trace(year: int, round_num: int, session_type: str = 'R'):
    con = duckdb.connect(DB_PATH, read_only=True)

    laps_df = con.execute(f"""
        SELECT Driver, OfficialTeamName, LapNumber, Position, LapTime_Sec, Compound, TyreLife, IsPitLap, TrackStatus
        FROM clean_laps
        WHERE Year = {year} AND RoundNumber = {round_num} AND SessionType = '{session_type}'
        ORDER BY Driver, LapNumber
    """).df()

    results_df = con.execute(f"""
        SELECT Driver, OfficialTeamName, GridPosition, FinishPosition, Points, Status
        FROM clean_results
        WHERE Year = {year} AND RoundNumber = {round_num} AND SessionType = '{session_type}'
        ORDER BY FinishPosition NULLS LAST
    """).df()
    con.close()

    if laps_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    lap_medians = laps_df.groupby('LapNumber')['LapTime_Sec'].transform('median')
    laps_df['CleanLapTime_Sec'] = laps_df['LapTime_Sec'].fillna(lap_medians)
    laps_df['CumulativeTime_Sec'] = laps_df.groupby('Driver')['CleanLapTime_Sec'].cumsum()

    max_lap = laps_df['LapNumber'].max()
    finishers = laps_df[laps_df['LapNumber'] == max_lap].sort_values('CumulativeTime_Sec')
    if not finishers.empty:
        winner_drv = finishers.iloc[0]['Driver']
        winner_total_time = finishers.iloc[0]['CumulativeTime_Sec']
        virtual_avg_lap_sec = winner_total_time / max_lap
    else:
        winner_drv = laps_df['Driver'].iloc[0]
        virtual_avg_lap_sec = laps_df['CleanLapTime_Sec'].median()

    laps_df['VirtualRefTime_Sec'] = laps_df['LapNumber'] * virtual_avg_lap_sec
    laps_df['GapToVirtualCar_Sec'] = np.round(laps_df['CumulativeTime_Sec'] - laps_df['VirtualRefTime_Sec'], 3)

    grid_rows = []
    for _, r in results_df.iterrows():
        grid_pos = int(r['GridPosition']) if pd.notna(r['GridPosition']) and r['GridPosition'] > 0 else 20
        grid_rows.append({
            'Driver': r['Driver'],
            'OfficialTeamName': r['OfficialTeamName'],
            'LapNumber': 0,
            'Position': grid_pos
        })
    grid_df = pd.DataFrame(grid_rows)
    pos_laps = laps_df[['Driver', 'OfficialTeamName', 'LapNumber', 'Position']].dropna(subset=['Position'])
    position_flow_df = pd.concat([grid_df, pos_laps], ignore_index=True).sort_values(['Driver', 'LapNumber'])

    lap1_pos = laps_df[laps_df['LapNumber'] == 1].set_index('Driver')['Position'].to_dict()
    
    overtake_rows = []
    for _, r in results_df.iterrows():
        drv = r['Driver']
        grid_p = int(r['GridPosition']) if pd.notna(r['GridPosition']) and r['GridPosition'] > 0 else 20
        fin_p = int(r['FinishPosition']) if pd.notna(r['FinishPosition']) else 20
        l1_p = int(lap1_pos.get(drv, grid_p))

        lap1_gain = grid_p - l1_p
        post_lap1_gain = l1_p - fin_p
        total_gain = grid_p - fin_p

        drv_pos_series = laps_df[laps_df['Driver'] == drv]['Position'].dropna()
        pos_changes = int((drv_pos_series.diff().fillna(0) != 0).sum())

        overtake_rows.append({
            'Driver': drv,
            'Team': r['OfficialTeamName'],
            'GridPosition': grid_p,
            'Lap1Position': l1_p,
            'FinishPosition': fin_p,
            'Lap1_PositionsGained': lap1_gain,
            'PostLap1_PositionsGained': post_lap1_gain,
            'Total_PositionsGained': total_gain,
            'PositionChangeEvents': pos_changes,
            'Status': r['Status']
        })

    overtake_df = pd.DataFrame(overtake_rows).sort_values('FinishPosition')

    biggest_mover = overtake_df.sort_values('Total_PositionsGained', ascending=False).iloc[0] if not overtake_df.empty else None
    best_starter = overtake_df.sort_values('Lap1_PositionsGained', ascending=False).iloc[0] if not overtake_df.empty else None

    notes = {
        "f1_context": (
            "A Race Trace ('Worm Chart') is the #1 chart on an F1 pit wall. Instead of plotting raw lap times, "
            "it compares every driver's cumulative elapsed time against a hypothetical 'Virtual Reference Car' driving "
            "at the race winner's constant average speed. When a line slopes UP, the driver is faster than average; "
            "when a line drops sharply by ~22 seconds, the driver made a pit stop; when all lines compress together, "
            "a Safety Car neutralized the field."
        ),
        "ds_methodology": (
            "Linear Time-Series Detrending: Raw cumulative race time grows from 0s to ~5,400s, making 2-second gaps "
            "invisible on a standard chart. By subtracting the linear trend T_ref(L) = L * Mean_Winner_LapTime, "
            "we isolate the stationary residual signal—revealing stint pace slopes and undercut crossovers."
        ),
        "key_insight": (
            f"Race Winner baseline: {winner_drv} (Avg lap: {virtual_avg_lap_sec:.3f}s). "
            f"Biggest overall mover: {biggest_mover['Driver']} ({biggest_mover['Total_PositionsGained']:+d} positions, "
            f"P{biggest_mover['GridPosition']} -> P{biggest_mover['FinishPosition']}). "
            f"Best Lap 1 starter: {best_starter['Driver']} ({best_starter['Lap1_PositionsGained']:+d} spots on Lap 1)."
            if biggest_mover is not None else "Race flow computed."
        )
    }

    return laps_df, position_flow_df, overtake_df, notes