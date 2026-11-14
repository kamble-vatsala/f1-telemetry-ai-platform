import duckdb
import numpy as np
import pandas as pd

DB_PATH = "data/f1_lakehouse.duckdb"


def analyze_tyres_and_pace_distributions(year: int, round_num: int, session_type: str = 'R'):
    con = duckdb.connect(DB_PATH, read_only=True)

    laps_df = con.execute(f"""
        SELECT Driver, OfficialTeamName, LapNumber, Stint, Compound, TyreLife,
               LapTime_Sec, FuelCorrectedLap_Sec, DeltaToBest_Sec, TrackStatus, IsGreenFlag, IsRepresentativePace, IsPitLap
        FROM clean_laps
        WHERE Year = {year} AND RoundNumber = {round_num} AND SessionType = '{session_type}'
        ORDER BY Driver, LapNumber
    """).df()

    stints_df = con.execute(f"""
        SELECT Driver, OfficialTeamName AS Team, Stint, Compound,
               StartLap, EndLap, StintSpanLaps
        FROM clean_stints
        WHERE Year = {year} AND RoundNumber = {round_num} AND SessionType = '{session_type}'
        ORDER BY Driver, Stint
    """).df()
    con.close()

    if laps_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    non_pit = laps_df[(laps_df['LapTime_Sec'] > 40) & (~laps_df['IsPitLap'].fillna(False))]
    comp_best = non_pit.groupby('Compound')['LapTime_Sec'].transform('min')
    laps_df['CompoundBest_Sec'] = comp_best.reindex(laps_df.index)
    
    laps_df['IsCompoundCleanPace'] = (
        laps_df['LapTime_Sec'].notna()
        & (laps_df['TrackStatus'] == '1')
        & (~laps_df['IsPitLap'].fillna(False))
        & (laps_df['LapNumber'] > 1 if session_type in ('R', 'S') else True)
        & (laps_df['LapTime_Sec'] <= laps_df['CompoundBest_Sec'] * 1.07)
    )

    clean_laps = laps_df[laps_df['IsCompoundCleanPace'] == True].copy()
    if clean_laps.empty:
        clean_laps = laps_df[laps_df['LapTime_Sec'].notna()].copy()

    stint_stats = []
    for (drv, stint_num), grp in laps_df.groupby(['Driver', 'Stint']):
        c_grp = grp[grp['IsCompoundCleanPace'] == True]
        comp = grp['Compound'].mode().iloc[0] if not grp['Compound'].dropna().empty else 'UNKNOWN'
        
        slope = np.nan
        if len(c_grp) >= 3 and c_grp['TyreLife'].nunique() >= 3:
            slope = float(np.polyfit(c_grp['TyreLife'], c_grp['FuelCorrectedLap_Sec'], 1)[0])

        stint_stats.append({
            'Driver': drv,
            'Team': grp['OfficialTeamName'].iloc[0],
            'Stint': int(stint_num),
            'Compound': comp,
            'StartLap': int(grp['LapNumber'].min()),
            'EndLap': int(grp['LapNumber'].max()),
            'StintSpanLaps': int(grp['LapNumber'].max() - grp['LapNumber'].min() + 1),
            'CleanStintLaps': len(c_grp),
            'AvgRawPace_Sec': round(float(c_grp['LapTime_Sec'].mean()), 3) if len(c_grp) > 0 else np.nan,
            'MedianRawPace_Sec': round(float(c_grp['LapTime_Sec'].median()), 3) if len(c_grp) > 0 else np.nan,
            'AvgFuelCorrectedPace_Sec': round(float(c_grp['FuelCorrectedLap_Sec'].mean()), 3) if len(c_grp) > 0 else np.nan,
            'TyreDegSlope_SecPerLap': round(slope, 4) if pd.notna(slope) else np.nan
        })

    stints_df = pd.DataFrame(stint_stats).sort_values(['Driver', 'Stint']).reset_index(drop=True)

    pace_rows = []
    for drv, grp in clean_laps.groupby('Driver'):
        times = grp['LapTime_Sec'].dropna()
        fc_times = grp['FuelCorrectedLap_Sec'].dropna()
        if len(times) == 0:
            continue
        q25, q75 = np.percentile(times, [25, 75])
        pace_rows.append({
            'Driver': drv,
            'Team': grp['OfficialTeamName'].iloc[0],
            'CleanLaps': len(times),
            'CompoundsUsed': ", ".join(grp['Compound'].dropna().unique()),
            'BestLap_Sec': round(float(times.min()), 3),
            'MedianPace_Sec': round(float(times.median()), 3),
            'MeanPace_Sec': round(float(times.mean()), 3),
            'IQR_Consistency_Sec': round(float(q75 - q25), 3),
            'StdDev_Sec': round(float(times.std(ddof=0)), 3),
            'FuelCorrectedMedian_Sec': round(float(fc_times.median()), 3)
        })

    pace_dist_df = pd.DataFrame(pace_rows).sort_values('MedianPace_Sec').reset_index(drop=True)
    if not pace_dist_df.empty:
        leader_med = pace_dist_df['MedianPace_Sec'].iloc[0]
        pace_dist_df['GapToPaceLeader_Sec'] = (pace_dist_df['MedianPace_Sec'] - leader_med).round(3)

    valid_stints = stints_df[stints_df['CleanStintLaps'] >= 3].copy()
    if not valid_stints.empty:
        compound_summary_df = valid_stints.groupby('Compound', as_index=False).agg(
            TotalStints=('Stint', 'count'),
            AvgStintLength_Laps=('StintSpanLaps', lambda s: round(s.mean(), 1)),
            MaxStintLength_Laps=('StintSpanLaps', 'max'),
            MedianPace_Sec=('MedianRawPace_Sec', lambda s: round(s.median(), 3)),
            MedianDegSlope_SecPerLap=('TyreDegSlope_SecPerLap', lambda s: round(s.dropna().median(), 4) if not s.dropna().empty else 0.0)
        ).sort_values('MedianPace_Sec')
    else:
        compound_summary_df = pd.DataFrame()

    fastest_med_drv = pace_dist_df.iloc[0] if not pace_dist_df.empty else None
    most_consistent = pace_dist_df[pace_dist_df['CleanLaps'] >= 10].sort_values('IQR_Consistency_Sec').iloc[0] if len(pace_dist_df[pace_dist_df['CleanLaps'] >= 10]) > 0 else None

    notes = {
        "f1_context": (
            "In changeable weather races, drivers run both Wet/Intermediate tyres and Dry Slick tyres. "
            "Because wet laps are 10–15 seconds slower than dry laps, a standard session-wide 107% filter would "
            "erase all Intermediate stints! We apply a Compound-Conditional 107% filter so wet and dry stints are "
            "evaluated against their own compound's benchmark."
        ),
        "ds_methodology": (
            "Stratified Outlier Filtering + OLS Regression: Grouping by Compound C before computing the 1.07 * min(LapTime | C) "
            "threshold preserves multi-modal lap time distributions (wet vs. dry regimes) while still filtering out "
            "Safety Car and Pit In/Out laps."
        ),
        "key_insight": (
            f"Compounds raced: {', '.join(compound_summary_df['Compound'].tolist()) if not compound_summary_df.empty else 'N/A'}. "
            f"Fastest median pace: {fastest_med_drv['Driver']} ({fastest_med_drv['MedianPace_Sec']:.3f}s). "
            f"Most consistent full-distance driver: {most_consistent['Driver']} (IQR: ±{most_consistent['IQR_Consistency_Sec']:.3f}s)."
            if fastest_med_drv is not None and most_consistent is not None else "Tyre analysis complete."
        )
    }

    return stints_df, pace_dist_df, compound_summary_df, clean_laps, notes