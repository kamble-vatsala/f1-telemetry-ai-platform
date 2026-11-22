from pathlib import Path
import duckdb

def export_executive_tables():
    print("=" * 65)
    print("WORKSTREAM 1: EXPORTING TYPE-SAFE PARQUET TABLES FOR BIGQUERY")
    print("=" * 65)

    project_root = Path(__file__).resolve().parent.parent.parent
    db_path = project_root / "data" / "f1_lakehouse.duckdb"
    export_dir = project_root / "data" / "bq_exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(db_path), read_only=True)

    # 1. Export clean_results with HUGEINT columns safely cast to BIGINT
    results_out = export_dir / "dim_results.parquet"
    con.execute(f"""
        COPY (
            SELECT
                Year, RoundNumber, EventName, Country, Location, EventFormat, SessionType,
                Driver, DriverNumber, FullName, OfficialTeamName, TeamFranchise, TeamColor,
                FinishPosition, GridPosition, PositionsGained, Points, Status, Time_Sec,
                Q1_Sec, Q2_Sec, Q3_Sec, AvgTrackTemp, AvgAirTemp, IsWetSession,
                CAST(SC_Count AS BIGINT) AS SC_Count,
                CAST(VSC_Count AS BIGINT) AS VSC_Count,
                CAST(RedFlag_Count AS BIGINT) AS RedFlag_Count,
                Penalty_Msgs, DeletedLaps_Count
            FROM clean_results
        ) TO '{results_out.as_posix()}' (FORMAT PARQUET);
    """)
    print(f"[+] Exported: {results_out.relative_to(project_root)} ({results_out.stat().st_size / 1024:.1f} KB)")

    # 2. Export clean_stints for Tyre Degradation & Pace Executive Analysis
    stints_out = export_dir / "fact_stints.parquet"
    con.execute(f"""
        COPY (
            SELECT * FROM clean_stints
        ) TO '{stints_out.as_posix()}' (FORMAT PARQUET);
    """)
    print(f"[+] Exported: {stints_out.relative_to(project_root)} ({stints_out.stat().st_size / 1024:.1f} KB)")

    # 3. Export session_weather for Track Condition Filtering
    weather_out = export_dir / "dim_weather.parquet"
    con.execute(f"""
        COPY (
            SELECT * FROM session_weather
        ) TO '{weather_out.as_posix()}' (FORMAT PARQUET);
    """)
    print(f"[+] Exported: {weather_out.relative_to(project_root)} ({weather_out.stat().st_size / 1024:.1f} KB)")

    con.close()
    print("\n[SUCCESS] All 3 Parquet files are ready in data/bq_exports/ for BigQuery upload!")
    print("=" * 65)

if __name__ == "__main__":
    export_executive_tables()