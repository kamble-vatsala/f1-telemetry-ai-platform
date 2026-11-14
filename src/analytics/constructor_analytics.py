import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import duckdb
import numpy as np
import pandas as pd

from src.analytics.incident_weather_analytics import analyze_weather_vsc_and_dnfs

DB_PATH = "data/f1_lakehouse.duckdb"


def analyze_constructors_and_strategy(year: int, round_num: int, session_type: str = 'R'):
    con = duckdb.connect(DB_PATH, read_only=True)

    event_teams_df = con.execute(f"""
        WITH team_laps AS (
            SELECT 
                OfficialTeamName AS Constructor,
                COUNT(*) AS TotalLaps,
                COUNT(*) FILTER (WHERE IsRepresentativePace) AS CleanLaps,
                ROUND(MIN(LapTime_Sec) FILTER (WHERE LapTime_Sec > 40), 3) AS BestLap_Sec,
                ROUND(MEDIAN(LapTime_Sec) FILTER (WHERE IsRepresentativePace), 3) AS MedianCleanPace_Sec,
                ROUND(
                    QUANTILE_CONT(LapTime_Sec, 0.75) FILTER (WHERE IsRepresentativePace) -
                    QUANTILE_CONT(LapTime_Sec, 0.25) FILTER (WHERE IsRepresentativePace), 3
                ) AS TeamPace_IQR_Sec,
                ROUND(MAX(SpeedST), 1) AS PeakSpeedTrap_kmh,
                ROUND(AVG(SpeedST) FILTER (WHERE IsRepresentativePace), 1) AS AvgSpeedTrap_kmh
            FROM clean_laps
            WHERE Year = {year} AND RoundNumber = {round_num} AND SessionType = '{session_type}'
            GROUP BY OfficialTeamName
        ),
        team_deg AS (
            SELECT 
                OfficialTeamName AS Constructor,
                ROUND(MEDIAN(TyreDegSlope_SecPerLap) FILTER (WHERE CleanStintLaps >= 4), 4) AS MedianTyreDeg_SecPerLap,
                COUNT(DISTINCT Stint) AS TotalStintsRun
            FROM clean_stints
            WHERE Year = {year} AND RoundNumber = {round_num} AND SessionType = '{session_type}'
            GROUP BY OfficialTeamName
        ),
        team_res AS (
            SELECT 
                OfficialTeamName AS Constructor,
                SUM(Points) AS EventPoints,
                MIN(FinishPosition) AS BestFinish,
                SUM(COALESCE(PositionsGained, 0)) AS NetPositionsGained,
                STRING_AGG(Driver || ' (P' || COALESCE(CAST(FinishPosition AS VARCHAR), 'DNF') || ')', ', ') AS DriversResult
            FROM clean_results
            WHERE Year = {year} AND RoundNumber = {round_num} AND SessionType = '{session_type}'
            GROUP BY OfficialTeamName
        )
        SELECT 
            l.Constructor,
            COALESCE(r.DriversResult, 'N/A') AS Drivers_Classification,
            COALESCE(r.EventPoints, 0) AS Points_Scored,
            COALESCE(r.BestFinish, 20) AS Best_Finish,
            COALESCE(r.NetPositionsGained, 0) AS Net_Pos_Gained,
            l.MedianCleanPace_Sec AS Median_Pace_Sec,
            l.BestLap_Sec AS Best_Lap_Sec,
            COALESCE(d.MedianTyreDeg_SecPerLap, 0.05) AS Tyre_Deg_Slope,
            l.PeakSpeedTrap_kmh AS Peak_Speed_kmh,
            l.TeamPace_IQR_Sec AS Consistency_IQR_Sec,
            l.CleanLaps AS Clean_Laps
        FROM team_laps l
        LEFT JOIN team_deg d ON l.Constructor = d.Constructor
        LEFT JOIN team_res r ON l.Constructor = r.Constructor
        ORDER BY COALESCE(l.MedianCleanPace_Sec, 999) ASC;
    """).df()

    if not event_teams_df.empty and event_teams_df['Median_Pace_Sec'].notna().any():
        fastest_team_med = event_teams_df['Median_Pace_Sec'].min()
        event_teams_df['Gap_To_Fastest_Team_Sec'] = (event_teams_df['Median_Pace_Sec'] - fastest_team_med).round(3)
        med_spd = event_teams_df['Peak_Speed_kmh'].median()
        med_pace = event_teams_df['Median_Pace_Sec'].median()
        
        def _aero_tag(row):
            if pd.isna(row['Peak_Speed_kmh']) or pd.isna(row['Median_Pace_Sec']):
                return "⚖️ Balanced Setup"
            if row['Median_Pace_Sec'] <= med_pace and row['Peak_Speed_kmh'] >= med_spd:
                return "🚀 High-Efficiency (Fast Straight + Corner)"
            elif row['Median_Pace_Sec'] <= med_pace and row['Peak_Speed_kmh'] < med_spd:
                return "🦅 High-Downforce (Corner Grip Priority)"
            elif row['Median_Pace_Sec'] > med_pace and row['Peak_Speed_kmh'] >= med_spd:
                return "💨 Low-Drag Trim (Straight-Line Priority)"
            return "🛠️ Drag-Limited / Development"

        event_teams_df['Aero_Setup_Profile'] = event_teams_df.apply(_aero_tag, axis=1)

    _, _, _, pit_dec_df, _, _ = analyze_weather_vsc_and_dnfs(year, round_num, session_type)
    if not pit_dec_df.empty:
        strategy_audit_df = pit_dec_df.groupby('Team', as_index=False).agg(
            Total_Pit_Calls=('PitLap', 'count'),
            Masterstroke_Calls=('DecisionVerdict', lambda s: int(s.str.startswith('✅').sum())),
            WorthIt_Calls=('DecisionVerdict', lambda s: int(s.str.startswith(('✅', '👍')).sum())),
            Costly_Undercut_Calls=('DecisionVerdict', lambda s: int(s.str.startswith('❌').sum())),
            Cheap_SC_VSC_Stops=('PittedUnder_SC_VSC', lambda s: int(s.str.startswith('Yes').sum())),
            Net_PitCycle_Positions=('NetPositions_5Laps', 'sum'),
            Avg_PostPit_PaceDelta_Sec=('PaceDelta_AfterPit_Sec', lambda s: round(float(s.mean()), 3))
        )
        strategy_audit_df['Strategy_Success_Rate_Pct'] = (
            (strategy_audit_df['WorthIt_Calls'] / strategy_audit_df['Total_Pit_Calls']) * 100
        ).round(1)
        strategy_audit_df = strategy_audit_df.sort_values(
            ['Strategy_Success_Rate_Pct', 'Net_PitCycle_Positions'], ascending=[False, False]
        ).reset_index(drop=True)
    else:
        strategy_audit_df = pd.DataFrame()

    season_constructors_df = con.execute(f"""
        WITH s_res AS (
            SELECT 
                OfficialTeamName AS Constructor,
                COUNT(DISTINCT RoundNumber) FILTER (WHERE SessionType = 'R') AS Races_Entered,
                SUM(Points) AS Season_Points,
                COUNT(*) FILTER (WHERE SessionType = 'R' AND FinishPosition = 1) AS Race_Wins,
                COUNT(*) FILTER (WHERE SessionType = 'R' AND FinishPosition <= 3) AS Podiums,
                COUNT(*) FILTER (WHERE SessionType = 'R' AND FinishPosition <= 10) AS Points_Finishes,
                ROUND(AVG(FinishPosition) FILTER (WHERE SessionType = 'R'), 2) AS Avg_Finish_Pos,
                ROUND(AVG(GridPosition) FILTER (WHERE SessionType = 'R' AND GridPosition > 0), 2) AS Avg_Grid_Pos,
                SUM(COALESCE(PositionsGained, 0)) FILTER (WHERE SessionType = 'R') AS Season_Net_Overtakes,
                ROUND(
                    100.0 * COUNT(*) FILTER (WHERE SessionType = 'R' AND (Status = 'Finished' OR Status LIKE '+%' OR Status = 'Lapped')) 
                    / NULLIF(COUNT(*) FILTER (WHERE SessionType = 'R'), 0), 1
                ) AS Mechanical_Finish_Rate_Pct
            FROM clean_results
            WHERE Year = {year}
            GROUP BY OfficialTeamName
        ),
        s_deg AS (
            SELECT 
                OfficialTeamName AS Constructor,
                ROUND(MEDIAN(TyreDegSlope_SecPerLap) FILTER (WHERE CleanStintLaps >= 5), 4) AS Season_Median_TyreDeg
            FROM clean_stints
            WHERE Year = {year} AND SessionType = 'R'
            GROUP BY OfficialTeamName
        )
        SELECT 
            ROW_NUMBER() OVER (ORDER BY r.Season_Points DESC, r.Avg_Finish_Pos ASC) AS WCC_Rank,
            r.Constructor,
            r.Season_Points,
            r.Race_Wins,
            r.Podiums,
            r.Points_Finishes,
            r.Avg_Grid_Pos,
            r.Avg_Finish_Pos,
            r.Season_Net_Overtakes,
            r.Mechanical_Finish_Rate_Pct,
            COALESCE(d.Season_Median_TyreDeg, 0.06) AS Season_TyreDeg_s_lap
        FROM s_res r
        LEFT JOIN s_deg d ON r.Constructor = d.Constructor
        WHERE r.Races_Entered >= 1
        ORDER BY WCC_Rank ASC;
    """).df()
    con.close()

    top_pace_team = event_teams_df.iloc[0]['Constructor'] if not event_teams_df.empty else "N/A"
    top_strat_team = strategy_audit_df.iloc[0]['Team'] if not strategy_audit_df.empty else "N/A"
    wcc_leader = season_constructors_df.iloc[0]['Constructor'] if not season_constructors_df.empty else "N/A"

    notes = {
        "f1_context": (
            "Formula 1 is both a Drivers' Championship and a Constructors' Engineering & Pit-Wall Championship. "
            "Every team chooses an aerodynamic philosophy: 'Low-Drag' cars top the speed traps on straights but slide "
            "more in corners (increasing tyre degradation), while 'High-Downforce' cars protect their tyres and corner "
            "faster at the cost of straight-line speed. Meanwhile, the Pit Wall decides when to undercut rivals or box under VSC/SC."
        ),
        "ds_methodology": (
            "Multi-Level Constructor Aggregation & Counterfactual Pit-Wall Scoring: We aggregate driver telemetry and "
            "stint metrics to the Constructor level using non-parametric medians, classify Aero Setup Profiles via 2D "
            "Median Quadrant Partitioning (SpeedST vs. CleanPace), and score Pit-Wall Strategy Execution via the ratio of "
            "positive-gain pit cycles (ΔPosition_t+5 >= 0) to total stops."
        ),
        "key_insight": (
            f"Fastest Constructor Pace in Session: {top_pace_team} | Best Pit-Wall Strategy Execution: {top_strat_team} | "
            f"{year} World Constructors' Championship Leader: {wcc_leader}."
        )
    }

    return event_teams_df, strategy_audit_df, season_constructors_df, notes