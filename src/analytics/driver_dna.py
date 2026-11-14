import duckdb
import numpy as np
import pandas as pd

DB_PATH = "data/f1_lakehouse.duckdb"


def _percentile_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    s = series.copy().astype(float)
    if s.dropna().nunique() <= 1:
        return pd.Series(75.0, index=s.index)
    pct = s.rank(pct=True, ascending=higher_is_better)
    return (45.0 + pct * 54.0).round(1).fillna(50.0)


def analyze_driver_dna_and_sectors(year: int, round_num: int = None, session_type: str = 'R'):
    con = duckdb.connect(DB_PATH)

    where_clause = f"Year = {year}" + (f" AND RoundNumber = {round_num}" if round_num else "")
    sec_where = f"Year = {year} AND RoundNumber = {round_num} AND SessionType = '{session_type}'" if round_num else f"Year = {year} AND SessionType = 'R'"

    sectors_df = con.execute(f"""
        SELECT 
            Driver,
            ANY_VALUE(OfficialTeamName) AS Team,
            ROUND(MIN(Sector1_Sec), 3) AS Best_S1,
            ROUND(MIN(Sector2_Sec), 3) AS Best_S2,
            ROUND(MIN(Sector3_Sec), 3) AS Best_S3,
            ROUND(MIN(LapTime_Sec), 3) AS Actual_Best_Lap,
            ROUND(MIN(Sector1_Sec) + MIN(Sector2_Sec) + MIN(Sector3_Sec), 3) AS Theoretical_Best_Lap,
            ROUND(GREATEST(0, MIN(LapTime_Sec) - (MIN(Sector1_Sec) + MIN(Sector2_Sec) + MIN(Sector3_Sec))), 3) AS Execution_Gap_Sec,
            ROUND(MAX(SpeedST), 1) AS Peak_SpeedTrap_kmh
        FROM clean_laps
        WHERE {sec_where} AND LapTime_Sec IS NOT NULL
        GROUP BY Driver
        HAVING Best_S1 IS NOT NULL AND Best_S2 IS NOT NULL AND Best_S3 IS NOT NULL
        ORDER BY Actual_Best_Lap ASC
    """).df()

    dna_raw = con.execute(f"""
        WITH quali_m AS (
            SELECT Driver, MEDIAN(DeltaToBest_Pct) AS QualiDelta_Pct
            FROM clean_laps
            WHERE {where_clause} AND SessionType IN ('Q', 'SQ') AND LapTime_Sec > 40
            GROUP BY Driver
        ),
        race_m AS (
            SELECT 
                Driver,
                ANY_VALUE(OfficialTeamName) AS Team,
                MEDIAN(DeltaToBest_Pct) FILTER (WHERE IsRepresentativePace) AS RaceDelta_Pct,
                QUANTILE_CONT(DeltaToBest_Sec, 0.75) FILTER (WHERE IsRepresentativePace) 
                  - QUANTILE_CONT(DeltaToBest_Sec, 0.25) FILTER (WHERE IsRepresentativePace) AS Pace_IQR,
                MAX(SpeedST) AS TopSpeed_kmh
            FROM clean_laps
            WHERE {where_clause} AND SessionType = '{session_type}'
            GROUP BY Driver
            HAVING COUNT(*) >= 10
        ),
        deg_m AS (
            SELECT Driver, MEDIAN(TyreDegSlope_SecPerLap) AS MedianDegSlope
            FROM clean_stints
            WHERE {where_clause} AND CleanStintLaps >= 4 AND TyreDegSlope_SecPerLap IS NOT NULL
            GROUP BY Driver
        ),
        craft_m AS (
            SELECT Driver, AVG(COALESCE(PositionsGained, 0)) AS AvgPositionsGained, SUM(Points) AS TotalPoints
            FROM clean_results
            WHERE {where_clause} AND SessionType IN ('R', 'S')
            GROUP BY Driver
        )
        SELECT 
            r.Driver,
            r.Team,
            COALESCE(q.QualiDelta_Pct, r.RaceDelta_Pct) AS QualiDelta_Pct,
            r.RaceDelta_Pct,
            COALESCE(d.MedianDegSlope, 0.10) AS MedianDegSlope,
            COALESCE(c.AvgPositionsGained, 0.0) AS AvgPositionsGained,
            COALESCE(c.TotalPoints, 0.0) AS TotalPoints,
            COALESCE(r.TopSpeed_kmh, 315.0) AS TopSpeed_kmh,
            COALESCE(r.Pace_IQR, 0.80) AS Pace_IQR
        FROM race_m r
        LEFT JOIN quali_m q ON r.Driver = q.Driver
        LEFT JOIN deg_m d ON r.Driver = d.Driver
        LEFT JOIN craft_m c ON r.Driver = c.Driver
        WHERE r.RaceDelta_Pct IS NOT NULL
    """).df()

    if not dna_raw.empty:
        dna_raw['Quali_Pace_Score'] = _percentile_score(dna_raw['QualiDelta_Pct'], higher_is_better=False)
        dna_raw['Race_Pace_Score'] = _percentile_score(dna_raw['RaceDelta_Pct'], higher_is_better=False)
        dna_raw['Tyre_Mgmt_Score'] = _percentile_score(dna_raw['MedianDegSlope'].abs(), higher_is_better=False)
        dna_raw['Overtake_Craft_Score'] = _percentile_score(dna_raw['AvgPositionsGained'], higher_is_better=True)
        dna_raw['Top_Speed_Score'] = _percentile_score(dna_raw['TopSpeed_kmh'], higher_is_better=True)
        dna_raw['Consistency_Score'] = _percentile_score(dna_raw['Pace_IQR'], higher_is_better=False)
        dna_raw['Overall_DNA_Rating'] = dna_raw[[
            'Quali_Pace_Score', 'Race_Pace_Score', 'Tyre_Mgmt_Score',
            'Overtake_Craft_Score', 'Top_Speed_Score', 'Consistency_Score'
        ]].mean(axis=1).round(1)
        dna_raw = dna_raw.sort_values('Overall_DNA_Rating', ascending=False).reset_index(drop=True)

    teammate_h2h_df = con.execute(f"""
        WITH driver_season AS (
            SELECT 
                OfficialTeamName AS Team,
                Driver,
                COUNT(DISTINCT RoundNumber) FILTER (WHERE SessionType = 'R') AS Races,
                SUM(Points) AS TotalPoints,
                ROUND(AVG(FinishPosition) FILTER (WHERE SessionType = 'R'), 2) AS AvgRaceFinish,
                ROUND(AVG(GridPosition) FILTER (WHERE SessionType = 'R' AND GridPosition > 0), 2) AS AvgGridPos,
                MIN(FinishPosition) FILTER (WHERE SessionType = 'R') AS BestFinish
            FROM clean_results
            WHERE Year = {year}
            GROUP BY OfficialTeamName, Driver
        )
        SELECT * FROM driver_season
        WHERE Races >= 2
        ORDER BY Team, TotalPoints DESC, AvgRaceFinish ASC
    """).df()
    con.close()

    top_dna = dna_raw.iloc[0] if not dna_raw.empty else None
    best_exec = sectors_df.sort_values('Execution_Gap_Sec').iloc[0] if not sectors_df.empty else None
    biggest_potential = sectors_df.sort_values('Execution_Gap_Sec', ascending=False).iloc[0] if not sectors_df.empty else None

    notes = {
        "f1_context": (
            "How do you compare two drivers with completely different strengths—like a Qualifying specialist vs. "
            "a Sunday Tyre Whisperer? We decompose driver performance into 6 orthogonal dimensions (Qualifying Speed, "
            "Long-Run Race Pace, Tyre Management, Overtaking Craft, Straight-Line Speed, and Lap-to-Lap Consistency). "
            "Additionally, comparing 'Theoretical Best Lap' (Best S1 + Best S2 + Best S3 across the session) against "
            "'Actual Best Lap' reveals whether a driver hooked up a perfect lap or left time on the table."
        ),
        "ds_methodology": (
            "Multi-Dimensional Percentile Rank Scaling: Because Qualifying Delta (%), Tyre Degradation (s/lap), "
            "Positions Gained (integers), and Speed Trap (km/h) exist on incompatible numerical scales, plotting raw "
            "values on a Radar Chart is mathematically invalid. We apply monotonic Percentile Rank Scaling to map "
            "every feature onto a uniform [45, 99] distribution where 99 represents the session/season benchmark."
        ),
        "key_insight": (
            f"Highest Overall 6-Axis DNA Rating: {top_dna['Driver']} ({top_dna['Overall_DNA_Rating']}/100). "
            f"Cleanest sector execution: {best_exec['Driver']} (only {best_exec['Execution_Gap_Sec']:.3f}s off theoretical optimal). "
            f"Largest untapped sector potential: {biggest_potential['Driver']} ({biggest_potential['Execution_Gap_Sec']:.3f}s faster if best S1+S2+S3 combined)."
            if top_dna is not None and best_exec is not None else "Driver DNA computed."
        )
    }

    return dna_raw, sectors_df, teammate_h2h_df, notes