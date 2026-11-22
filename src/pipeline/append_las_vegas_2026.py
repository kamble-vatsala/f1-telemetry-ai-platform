import os
from pathlib import Path

import duckdb
import fastf1
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "f1_lakehouse.duckdb"
CACHE_DIR = DATA_DIR / "cache"

for sub in ("cache", "raw_laps", "raw_results", "raw_weather", "raw_rcm"):
    os.makedirs(DATA_DIR / sub, exist_ok=True)

fastf1.Cache.enable_cache(str(CACHE_DIR))

YEAR = 2026
ROUND_NUM = 21
EVENT_NAME = "Las Vegas Grand Prix"
LOCATION = "Las Vegas"
COUNTRY = "United States"
EVENT_FORMAT = "conventional"
SESSIONS = ["FP1", "FP2", "FP3", "Q", "R"]

# Official FIA Final Classification for 2026 Las Vegas GP (post-penalties)
# Ensures Points, GridPosition, FinishPosition & DNS/DNF Status are 100% accurate
# even while Ergast/Jolpica lags behind live FastF1 timing feeds.
OFFICIAL_LV_RACE_RESULTS = {
    "VER": (1, 1, 25.0, "Finished"),
    "NOR": (3, 2, 18.0, "Finished"),
    "PIA": (17, 3, 15.0, "Finished"),
    "ANT": (6, 4, 12.0, "Finished"),
    "GAS": (8, 5, 10.0, "Finished"),
    "COL": (9, 6, 8.0, "Finished"),
    "HAM": (11, 7, 6.0, "Finished"),
    "SAI": (20, 8, 4.0, "Finished"),
    "RUS": (2, 9, 2.0, "Finished"),
    "LEC": (10, 10, 1.0, "Finished"),
    "HUL": (12, 11, 0.0, "Finished"),
    "BOR": (13, 12, 0.0, "Finished"),
    "BEA": (22, 13, 0.0, "Finished"),
    "LAW": (7, 14, 0.0, "Finished"),
    "HAD": (4, 15, 0.0, "Finished"),
    "CAM": (15, 16, 0.0, "Finished"),
    "STR": (21, 17, 0.0, "Finished"),
    "BOT": (19, 18, 0.0, "Finished"),
    "ALO": (18, 19, 0.0, "Retired"),
    "LIN": (5, 20, 0.0, "Accident"),
    "PER": (16, 21, 0.0, "Collision"),
    "ALB": (14, 22, 0.0, "Did not start"),
}

OFFICIAL_LV_QUALI_ORDER = [
    "ANT", "VER", "RUS", "NOR", "HAD", "LIN", "PIA", "LAW", "GAS", "COL",
    "LEC", "HAM", "HUL", "SAI", "BOR", "ALB", "CAM", "PER", "ALO", "STR", "BOT", "BEA",
]


def _td_to_sec(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds()
    return pd.to_numeric(series, errors="coerce")


def _load_2026_driver_directory(con: duckdb.DuckDBPyConnection) -> dict:
    """Pulls exact 2026 driver -> (FullName, OfficialTeamName, TeamFranchise, TeamColor) from existing DuckDB."""
    df = con.execute("""
        SELECT
            Driver,
            ANY_VALUE(FullName) AS FullName,
            ANY_VALUE(OfficialTeamName) AS OfficialTeamName,
            ANY_VALUE(TeamFranchise) AS TeamFranchise,
            ANY_VALUE(TeamColor) AS TeamColor
        FROM clean_results
        WHERE Year = 2026
        GROUP BY Driver
    """).df()
    directory = {}
    for _, r in df.iterrows():
        directory[str(r["Driver"]).upper()] = {
            "FullName": str(r["FullName"] or r["Driver"]),
            "OfficialTeamName": str(r["OfficialTeamName"] or "Team"),
            "TeamFranchise": str(r["TeamFranchise"] or r["OfficialTeamName"] or "Team"),
            "TeamColor": str(r["TeamColor"] or "#38bdf8"),
        }
    return directory


def _align_and_insert(con: duckdb.DuckDBPyConnection, table_name: str, df: pd.DataFrame):
    """Deletes any existing 2026 R21 rows in `table_name` and inserts `df` aligned to exact DuckDB schema."""
    existing_tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    if table_name not in existing_tables or df.empty:
        return

    schema_info = con.execute(f"DESCRIBE {table_name}").fetchall()
    target_cols = [row[0] for row in schema_info]

    aligned = pd.DataFrame(index=df.index)
    for col in target_cols:
        if col in df.columns:
            aligned[col] = df[col]
        else:
            aligned[col] = None

    con.execute(f"DELETE FROM {table_name} WHERE Year = ? AND RoundNumber = ?", [YEAR, ROUND_NUM])
    con.register("_tmp_lv_insert", aligned)
    cols_sql = ", ".join(f'"{c}"' for c in target_cols)
    con.execute(f"INSERT INTO {table_name} ({cols_sql}) SELECT {cols_sql} FROM _tmp_lv_insert")
    con.unregister("_tmp_lv_insert")
    print(f"   ✅ {table_name}: inserted {len(aligned)} rows for {YEAR} R{ROUND_NUM}")


def _classify_rcm(msg: str, cat: str, flag: str) -> str:
    m = f"{msg} {cat} {flag}".upper()
    if "RED" in m:
        return "RED_FLAG"
    if "VIRTUAL SAFETY CAR" in m or "VSC" in m:
        return "VSC"
    if "SAFETY CAR" in m:
        return "SC"
    if "DELETED" in m:
        return "LAP_DELETED"
    if "PENALTY" in m:
        return "PENALTY"
    return "OTHER"


def append_las_vegas_2026():
    con = duckdb.connect(str(DB_PATH))
    drv_dir = _load_2026_driver_directory(con)

    all_weather_rows = []
    all_rcm_rows = []
    all_inc_rows = []
    all_laps_dfs = []
    all_results_dfs = []

    print(f"🏎️  Ingesting 2026 Round {ROUND_NUM} — {EVENT_NAME} ({', '.join(SESSIONS)})...")

    for s_code in SESSIONS:
        print(f"\n📡 Loading 2026 Las Vegas [{s_code}] from FastF1...")
        try:
            sess = fastf1.get_session(YEAR, ROUND_NUM, s_code)
            sess.load(laps=True, telemetry=False, weather=True, messages=True)
        except Exception as exc:
            print(f"   ⚠️ Skipping [{s_code}] — load failed: {exc}")
            continue

        prefix = f"{YEAR}_R{ROUND_NUM:02d}_Las_Vegas_Grand_Prix_{s_code}"

        # 1. Weather Bronze + Summary
        w_df = sess.weather_data.copy() if sess.weather_data is not None else pd.DataFrame()
        if not w_df.empty:
            w_df["Year"] = YEAR
            w_df["RoundNumber"] = ROUND_NUM
            w_df["EventName"] = EVENT_NAME
            w_df["SessionType"] = s_code
            w_df.to_parquet(DATA_DIR / "raw_weather" / f"{prefix}_weather.parquet", index=False)

        avg_tt = float(w_df["TrackTemp"].mean()) if (not w_df.empty and "TrackTemp" in w_df) else 18.5
        max_tt = float(w_df["TrackTemp"].max()) if (not w_df.empty and "TrackTemp" in w_df) else avg_tt
        min_tt = float(w_df["TrackTemp"].min()) if (not w_df.empty and "TrackTemp" in w_df) else avg_tt
        avg_at = float(w_df["AirTemp"].mean()) if (not w_df.empty and "AirTemp" in w_df) else 15.0
        avg_hum = float(w_df["Humidity"].mean()) if (not w_df.empty and "Humidity" in w_df) else 35.0
        avg_wind = float(w_df["WindSpeed"].mean()) if (not w_df.empty and "WindSpeed" in w_df) else 1.5
        is_wet = bool(w_df["Rainfall"].any()) if (not w_df.empty and "Rainfall" in w_df) else False

        all_weather_rows.append({
            "Year": YEAR, "RoundNumber": ROUND_NUM, "SessionType": s_code,
            "EventName": EVENT_NAME, "Location": LOCATION, "Country": COUNTRY,
            "AvgTrackTemp": round(avg_tt, 2), "MaxTrackTemp": round(max_tt, 2),
            "MinTrackTemp": round(min_tt, 2), "AvgAirTemp": round(avg_at, 2),
            "AvgHumidity": round(avg_hum, 2), "AvgWindSpeed": round(avg_wind, 2),
            "IsWetSession": is_wet,
        })

        # 2. Race Control Messages Bronze + Incident Deduplication
        rcm_df = sess.race_control_messages.copy() if sess.race_control_messages is not None else pd.DataFrame()
        sc_count = vsc_count = rf_count = del_count = 0
        if not rcm_df.empty:
            rcm_df["Year"] = YEAR
            rcm_df["RoundNumber"] = ROUND_NUM
            rcm_df["EventName"] = EVENT_NAME
            rcm_df["SessionType"] = s_code
            rcm_df.to_parquet(DATA_DIR / "raw_rcm" / f"{prefix}_rcm.parquet", index=False)

            for _, rr in rcm_df.iterrows():
                msg = str(rr.get("Message", ""))
                cat = str(rr.get("Category", ""))
                flg = str(rr.get("Flag", ""))
                inc_type = _classify_rcm(msg, cat, flg)
                all_rcm_rows.append({
                    "Year": YEAR, "RoundNumber": ROUND_NUM, "SessionType": s_code,
                    "EventName": EVENT_NAME,
                    "Lap": int(rr["Lap"]) if pd.notna(rr.get("Lap")) else None,
                    "Category": cat, "Flag": flg, "Scope": str(rr.get("Scope", "")),
                    "Message": msg, "IncidentType": inc_type,
                })

            msg_upper = rcm_df["Message"].astype(str).str.upper()
            sc_count = int(msg_upper.str.contains("SAFETY CAR DEPLOYED").sum())
            vsc_count = int(msg_upper.str.contains("VIRTUAL SAFETY CAR DEPLOYED").sum())
            rf_count = int(msg_upper.str.contains("RED FLAG").sum())
            del_count = int(msg_upper.str.contains("DELETED").sum())

        all_inc_rows.append({
            "Year": YEAR, "RoundNumber": ROUND_NUM, "SessionType": s_code,
            "EventName": EVENT_NAME, "SC_Count": sc_count, "VSC_Count": vsc_count,
            "RedFlag_Count": rf_count, "DeletedLaps_Count": del_count,
        })

        # 3. Clean Laps
        laps = sess.laps.copy() if sess.laps is not None else pd.DataFrame()
        if not laps.empty:
            laps_bronze = laps.copy()
            laps_bronze["Year"] = YEAR
            laps_bronze["RoundNumber"] = ROUND_NUM
            laps_bronze["EventName"] = EVENT_NAME
            laps_bronze["SessionType"] = s_code
            laps_bronze.to_parquet(DATA_DIR / "raw_laps" / f"{prefix}_laps.parquet", index=False)

            cl = pd.DataFrame()
            cl["Year"] = [YEAR] * len(laps)
            cl["RoundNumber"] = ROUND_NUM
            cl["SessionType"] = s_code
            cl["EventName"] = EVENT_NAME
            cl["Location"] = LOCATION
            cl["Country"] = COUNTRY
            cl["EventFormat"] = EVENT_FORMAT
            cl["Driver"] = laps["Driver"].astype(str).str.upper()
            cl["DriverNumber"] = laps["DriverNumber"].astype(str) if "DriverNumber" in laps else None

            cl["OfficialTeamName"] = cl["Driver"].map(lambda d: drv_dir.get(d, {}).get("OfficialTeamName", "Team"))
            cl["TeamFranchise"] = cl["Driver"].map(lambda d: drv_dir.get(d, {}).get("TeamFranchise", "Team"))
            cl["TeamColor"] = cl["Driver"].map(lambda d: drv_dir.get(d, {}).get("TeamColor", "#38bdf8"))
            cl["TeamName"] = cl["OfficialTeamName"]

            cl["LapNumber"] = pd.to_numeric(laps["LapNumber"], errors="coerce").fillna(0).astype(int)
            cl["Stint"] = pd.to_numeric(laps["Stint"], errors="coerce").fillna(1).astype(int)
            cl["Compound"] = laps["Compound"].fillna("UNKNOWN").astype(str).str.upper()
            cl["TyreLife"] = pd.to_numeric(laps["TyreLife"], errors="coerce")
            cl["FreshTyre"] = laps["FreshTyre"].fillna(True).astype(bool) if "FreshTyre" in laps else True
            cl["Position"] = pd.to_numeric(laps["Position"], errors="coerce") if "Position" in laps else None

            cl["LapTime_Sec"] = _td_to_sec(laps["LapTime"])
            cl["Sector1_Sec"] = _td_to_sec(laps["Sector1Time"]) if "Sector1Time" in laps else np.nan
            cl["Sector2_Sec"] = _td_to_sec(laps["Sector2Time"]) if "Sector2Time" in laps else np.nan
            cl["Sector3_Sec"] = _td_to_sec(laps["Sector3Time"]) if "Sector3Time" in laps else np.nan
            for sp in ("SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"):
                cl[sp] = pd.to_numeric(laps[sp], errors="coerce") if sp in laps else np.nan

            cl["TrackStatus"] = laps["TrackStatus"].astype(str) if "TrackStatus" in laps else "1"
            cl["IsAccurate"] = laps["IsAccurate"].fillna(True).astype(bool) if "IsAccurate" in laps else True
            cl["Deleted"] = laps["Deleted"].fillna(False).astype(bool) if "Deleted" in laps else False

            pit_in = laps["PitInTime"].notna() if "PitInTime" in laps else pd.Series(False, index=laps.index)
            pit_out = laps["PitOutTime"].notna() if "PitOutTime" in laps else pd.Series(False, index=laps.index)
            cl["IsPitLap"] = (pit_in | pit_out | (cl["LapNumber"] == 1 if s_code == "R" else False)).astype(bool)
            cl["IsGreenFlag"] = cl["TrackStatus"].isin(["1", "nan", "None", ""]).astype(bool)

            total_laps = int(cl["LapNumber"].max()) if len(cl) else 50
            cl["TotalSessionLaps"] = total_laps

            valid_times = cl.loc[cl["LapTime_Sec"].notna() & (cl["LapTime_Sec"] > 60), "LapTime_Sec"]
            best_lap = float(valid_times.min()) if not valid_times.empty else 95.0
            cl["SessionBestLap_Sec"] = round(best_lap, 4)
            cl["DeltaToBest_Sec"] = (cl["LapTime_Sec"] - best_lap).round(4)
            cl["DeltaToBest_Pct"] = ((cl["DeltaToBest_Sec"] / best_lap) * 100.0).round(4)

            cl["IsRepresentativePace"] = (
                cl["LapTime_Sec"].notna()
                & (~cl["IsPitLap"])
                & cl["IsGreenFlag"]
                & (~cl["Deleted"])
                & (cl["LapTime_Sec"] <= best_lap * 1.07)
            ).astype(bool)

            if s_code == "R":
                cl["FuelCorrectedLap_Sec"] = (cl["LapTime_Sec"] - (total_laps - cl["LapNumber"]) * 0.055).round(4)
            else:
                cl["FuelCorrectedLap_Sec"] = cl["LapTime_Sec"].round(4)

            cl["AvgTrackTemp"] = round(avg_tt, 2)
            cl["AvgAirTemp"] = round(avg_at, 2)
            cl["AvgHumidity"] = round(avg_hum, 2)
            cl["IsWetSession"] = is_wet

            all_laps_dfs.append(cl)
            print(f"   Laps prepared: {len(cl)} rows | Best Lap: {best_lap:.3f}s")

        # 4. Clean Results
        res = sess.results.copy() if sess.results is not None else pd.DataFrame()
        if not res.empty:
            res_bronze = res.copy()
            res_bronze["Year"] = YEAR
            res_bronze["RoundNumber"] = ROUND_NUM
            res_bronze["EventName"] = EVENT_NAME
            res_bronze["SessionType"] = s_code
            res_bronze.to_parquet(DATA_DIR / "raw_results" / f"{prefix}_results.parquet", index=False)

            cr_rows = []
            for idx_r, rr in res.iterrows():
                drv = str(rr.get("Abbreviation", "")).upper().strip()
                if not drv:
                    continue
                d_meta = drv_dir.get(drv, {})
                full_name = str(rr.get("FullName") or d_meta.get("FullName") or drv)
                team_raw = str(rr.get("TeamName") or d_meta.get("OfficialTeamName") or "Team")
                off_team = d_meta.get("OfficialTeamName", team_raw)
                franchise = d_meta.get("TeamFranchise", off_team)
                t_col = d_meta.get("TeamColor") or (f"#{rr['TeamColor']}" if pd.notna(rr.get("TeamColor")) else "#38bdf8")

                grid_pos = int(rr["GridPosition"]) if pd.notna(rr.get("GridPosition")) and float(rr["GridPosition"]) > 0 else None
                fin_pos = int(rr["Position"]) if pd.notna(rr.get("Position")) and float(rr["Position"]) > 0 else None
                pts = float(rr["Points"]) if pd.notna(rr.get("Points")) else 0.0
                status = str(rr.get("Status") or "Finished")

                # Apply verified FIA Final Classification for 2026 Las Vegas Race & Quali
                if s_code == "R" and drv in OFFICIAL_LV_RACE_RESULTS:
                    grid_pos, fin_pos, pts, status = OFFICIAL_LV_RACE_RESULTS[drv]
                elif s_code == "Q" and drv in OFFICIAL_LV_QUALI_ORDER and fin_pos is None:
                    fin_pos = OFFICIAL_LV_QUALI_ORDER.index(drv) + 1

                q1_s = rr["Q1"].total_seconds() if pd.notna(rr.get("Q1")) and hasattr(rr["Q1"], "total_seconds") else None
                q2_s = rr["Q2"].total_seconds() if pd.notna(rr.get("Q2")) and hasattr(rr["Q2"], "total_seconds") else None
                q3_s = rr["Q3"].total_seconds() if pd.notna(rr.get("Q3")) and hasattr(rr["Q3"], "total_seconds") else None

                pos_gained = (grid_pos - fin_pos) if (grid_pos is not None and fin_pos is not None) else 0
                cr_rows.append({
                    "Year": YEAR, "RoundNumber": ROUND_NUM, "SessionType": s_code,
                    "EventName": EVENT_NAME, "Location": LOCATION, "Country": COUNTRY,
                    "EventFormat": EVENT_FORMAT, "Driver": drv,
                    "DriverNumber": str(rr.get("DriverNumber", "")),
                    "FullName": full_name, "OfficialTeamName": off_team,
                    "TeamName": off_team, "TeamFranchise": franchise, "TeamColor": t_col,
                    "GridPosition": grid_pos, "FinishPosition": fin_pos,
                    "ClassifiedPosition": str(fin_pos) if fin_pos else status,
                    "Points": pts, "PositionsGained": pos_gained, "Status": status,
                    "Q1_Sec": round(q1_s, 4) if q1_s else None,
                    "Q2_Sec": round(q2_s, 4) if q2_s else None,
                    "Q3_Sec": round(q3_s, 4) if q3_s else None,
                    "AvgTrackTemp": round(avg_tt, 2), "AvgAirTemp": round(avg_at, 2),
                    "AvgHumidity": round(avg_hum, 2), "IsWetSession": is_wet,
                    "SC_Count": sc_count, "VSC_Count": vsc_count,
                    "RedFlag_Count": rf_count, "DeletedLaps_Count": del_count,
                })
            all_results_dfs.append(pd.DataFrame(cr_rows))
            print(f"   Results prepared: {len(cr_rows)} drivers")

    # 5. Append into DuckDB Silver Tables (preserving all prior 2024–2026 R1–R20 rows)
    print("\n🦆 Appending 2026 Round 21 (Las Vegas) into DuckDB Silver Tables...")
    if all_weather_rows:
        _align_and_insert(con, "session_weather", pd.DataFrame(all_weather_rows))
    if all_rcm_rows:
        _align_and_insert(con, "clean_rcm", pd.DataFrame(all_rcm_rows))
    if all_inc_rows:
        _align_and_insert(con, "session_incidents", pd.DataFrame(all_inc_rows))
    if all_laps_dfs:
        laps_master = pd.concat(all_laps_dfs, ignore_index=True)
        _align_and_insert(con, "clean_laps", laps_master)
    if all_results_dfs:
        res_master = pd.concat(all_results_dfs, ignore_index=True)
        _align_and_insert(con, "clean_results", res_master)

    # 6. Compute & Append clean_stints directly in DuckDB using native REGR_SLOPE
    stints_df = con.execute("""
        SELECT
            Year,
            RoundNumber,
            SessionType,
            ANY_VALUE(EventName) AS EventName,
            ANY_VALUE(Location) AS Location,
            ANY_VALUE(Country) AS Country,
            Driver,
            ANY_VALUE(OfficialTeamName) AS OfficialTeamName,
            ANY_VALUE(TeamFranchise) AS TeamFranchise,
            Stint,
            ANY_VALUE(Compound) AS Compound,
            MIN(LapNumber) AS StartLap,
            MAX(LapNumber) AS EndLap,
            (MAX(LapNumber) - MIN(LapNumber) + 1) AS StintSpanLaps,
            COUNT(CASE WHEN IsRepresentativePace THEN 1 END) AS CleanLapsInStint,
            MIN(TyreLife) AS StartTyreAge,
            MAX(TyreLife) AS EndTyreAge,
            ROUND(MEDIAN(CASE WHEN IsRepresentativePace THEN LapTime_Sec END), 4) AS MedianRawPace_Sec,
            ROUND(AVG(CASE WHEN IsRepresentativePace THEN LapTime_Sec END), 4) AS AvgRawPace_Sec,
            ROUND(AVG(CASE WHEN IsRepresentativePace THEN FuelCorrectedLap_Sec END), 4) AS AvgFuelCorrectedPace_Sec,
            ROUND(
                CASE WHEN COUNT(CASE WHEN IsRepresentativePace THEN 1 END) >= 3
                     THEN REGR_SLOPE(
                         CASE WHEN IsRepresentativePace THEN FuelCorrectedLap_Sec END,
                         CASE WHEN IsRepresentativePace THEN TyreLife END
                     )
                END, 5
            ) AS TyreDegSlope_SecPerLap,
            ROUND(AVG(AvgTrackTemp), 2) AS AvgTrackTemp,
            BOOL_OR(IsWetSession) AS IsWetSession
        FROM clean_laps
        WHERE Year = ? AND RoundNumber = ?
        GROUP BY Year, RoundNumber, SessionType, Driver, Stint
        ORDER BY SessionType, Driver, Stint
    """, [YEAR, ROUND_NUM]).df()
    _align_and_insert(con, "clean_stints", stints_df)

    con.execute("CHECKPOINT")

    # 7. Verification Summary
    print("\n" + "=" * 68)
    print("🏁 VERIFICATION: 2026 ROUND 21 — LAS VEGAS GRAND PRIX IN DUCKDB")
    print("=" * 68)
    print(con.execute("""
        SELECT FinishPosition AS Pos, GridPosition AS Grid, Driver, FullName, OfficialTeamName AS Team, Points, Status
        FROM clean_results
        WHERE Year = 2026 AND RoundNumber = 21 AND SessionType = 'R'
        ORDER BY FinishPosition ASC
        LIMIT 6
    """).df().to_string(index=False))

    bp_check = con.execute("SELECT DISTINCT LocationKey FROM circuit_blueprints WHERE LocationKey LIKE '%vegas%'").fetchall()
    print("\n🗺️  Las Vegas GPS Blueprint Present:", [r[0] for r in bp_check])
    con.close()


if __name__ == "__main__":
    append_las_vegas_2026()