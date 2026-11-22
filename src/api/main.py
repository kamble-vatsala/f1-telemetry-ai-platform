from contextlib import asynccontextmanager
from functools import lru_cache
import math
import os
import sys
from pathlib import Path
from typing import Any, Optional

import duckdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")
DB_PATH = PROJECT_ROOT / "data" / "f1_lakehouse.duckdb"
FRONTEND_DIR = PROJECT_ROOT / "frontend"


def _get_db() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DB_PATH), read_only=True)


def _rotate_xy(x_arr: np.ndarray, y_arr: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    rad = np.deg2rad(angle_deg)
    rot_mat = np.array([[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]])
    xy = np.column_stack([x_arr, y_arr])
    rot = np.matmul(xy, rot_mat)
    return rot[:, 0], rot[:, 1]


def _to_json_safe(obj: Any, max_rows: int = 50) -> Any:
    if isinstance(obj, pd.DataFrame):
        trimmed = obj.head(max_rows).copy()
        return _to_json_safe(trimmed.to_dict(orient="records"), max_rows=max_rows)
    if isinstance(obj, tuple):
        return [_to_json_safe(item, max_rows=max_rows) for item in obj]
    if isinstance(obj, list):
        return [_to_json_safe(item, max_rows=max_rows) for item in obj]
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v, max_rows=max_rows) for k, v in obj.items()}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return None if (math.isnan(val) or math.isinf(val)) else round(val, 4)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if pd.isna(obj) if not isinstance(obj, (str, bytes)) else False:
        return None
    return obj


# ---------------------------------------------------------------------------
# 1. Global Catalog, Podium Banner & Roster Queries (< 20 ms)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=16)
def _cached_tracks_for_year(year: int) -> list:
    con = _get_db()
    try:
        df = con.execute(
            """
            SELECT
                RoundNumber,
                ANY_VALUE(EventName) AS EventName,
                ANY_VALUE(Location) AS Location,
                ANY_VALUE(Country) AS Country
            FROM clean_results
            WHERE Year = ?
            GROUP BY RoundNumber
            ORDER BY RoundNumber
            """,
            [year],
        ).df()
        return _to_json_safe(df, max_rows=30)
    finally:
        con.close()


@lru_cache(maxsize=128)
def _cached_session_overview(year: int, round_num: int, session_type: str = "R") -> dict:
    con = _get_db()
    try:
        res_df = con.execute(
            """
            SELECT
                Driver, ANY_VALUE(FullName) AS FullName,
                ANY_VALUE(OfficialTeamName) AS Team, ANY_VALUE(TeamColor) AS TeamColor,
                MIN(FinishPosition) AS FinishPosition, MIN(GridPosition) AS GridPosition,
                MAX(Points) AS Points, ANY_VALUE(Status) AS Status,
                ANY_VALUE(EventName) AS EventName, ANY_VALUE(Location) AS Location,
                ANY_VALUE(Country) AS Country, ROUND(AVG(AvgTrackTemp), 1) AS AvgTrackTemp,
                ROUND(AVG(AvgAirTemp), 1) AS AvgAirTemp,
                MAX(CASE WHEN IsWetSession THEN 1 ELSE 0 END) AS IsWetSession,
                MAX(SC_Count) AS SC_Count, MAX(VSC_Count) AS VSC_Count
            FROM clean_results
            WHERE Year = ? AND RoundNumber = ? AND SessionType = ?
            GROUP BY Driver
            ORDER BY FinishPosition ASC NULLS LAST, GridPosition ASC NULLS LAST
            """,
            [year, round_num, session_type],
        ).df()

        if res_df.empty and session_type != "R":
            res_df = con.execute(
                """
                SELECT
                    Driver, ANY_VALUE(FullName) AS FullName,
                    ANY_VALUE(OfficialTeamName) AS Team, ANY_VALUE(TeamColor) AS TeamColor,
                    MIN(FinishPosition) AS FinishPosition, MIN(GridPosition) AS GridPosition,
                    MAX(Points) AS Points, ANY_VALUE(Status) AS Status,
                    ANY_VALUE(EventName) AS EventName, ANY_VALUE(Location) AS Location,
                    ANY_VALUE(Country) AS Country, ROUND(AVG(AvgTrackTemp), 1) AS AvgTrackTemp,
                    ROUND(AVG(AvgAirTemp), 1) AS AvgAirTemp,
                    MAX(CASE WHEN IsWetSession THEN 1 ELSE 0 END) AS IsWetSession,
                    MAX(SC_Count) AS SC_Count, MAX(VSC_Count) AS VSC_Count
                FROM clean_results
                WHERE Year = ? AND RoundNumber = ? AND SessionType = 'R'
                GROUP BY Driver
                ORDER BY FinishPosition ASC NULLS LAST
                """,
                [year, round_num],
            ).df()

        laps_df = con.execute(
            """
            SELECT
                Driver,
                COUNT(*) AS TotalLaps,
                SUM(CASE WHEN IsRepresentativePace THEN 1 ELSE 0 END) AS CleanLaps,
                ROUND(MIN(LapTime_Sec), 3) AS BestLap_Sec
            FROM clean_laps
            WHERE Year = ? AND RoundNumber = ? AND SessionType = ?
              AND LapTime_Sec IS NOT NULL AND NOT isnan(LapTime_Sec)
            GROUP BY Driver
            """,
            [year, round_num, session_type],
        ).df()

        lap_map = {}
        if not laps_df.empty:
            for _, r in laps_df.iterrows():
                lap_map[str(r["Driver"])] = {
                    "TotalLaps": int(r["TotalLaps"] or 0),
                    "CleanLaps": int(r["CleanLaps"] or 0),
                    "BestLap_Sec": float(r["BestLap_Sec"]) if pd.notna(r["BestLap_Sec"]) else None,
                }

        roster = []
        for _, r in res_df.iterrows():
            drv = str(r["Driver"])
            l_info = lap_map.get(drv, {"TotalLaps": 0, "CleanLaps": 0, "BestLap_Sec": None})
            status_str = str(r["Status"] or "Finished")
            is_finished = status_str.lower() in ("finished", "lapped") or status_str.startswith("+")
            roster.append(
                {
                    "Driver": drv,
                    "FullName": str(r["FullName"] or drv),
                    "Team": str(r["Team"] or "Team"),
                    "TeamColor": str(r["TeamColor"] or "#38bdf8"),
                    "FinishPosition": int(r["FinishPosition"]) if pd.notna(r["FinishPosition"]) else None,
                    "GridPosition": int(r["GridPosition"]) if pd.notna(r["GridPosition"]) else None,
                    "Points": float(r["Points"] or 0.0),
                    "Status": status_str,
                    "IsClassified": is_finished,
                    "TotalLaps": l_info["TotalLaps"],
                    "CleanLaps": l_info["CleanLaps"],
                    "BestLap_Sec": l_info["BestLap_Sec"],
                }
            )

        podium = roster[:3] if len(roster) >= 3 else roster
        first_row = res_df.iloc[0].to_dict() if not res_df.empty else {}
        valid_drivers = [d["Driver"] for d in roster if d["TotalLaps"] > 0]
        default_drivers = valid_drivers[:3] if len(valid_drivers) >= 3 else ["VER", "NOR", "LEC"]

        return {
            "year": year,
            "round_num": round_num,
            "session_type": session_type,
            "event_name": str(first_row.get("EventName") or f"Round {round_num}"),
            "location": str(first_row.get("Location") or ""),
            "country": str(first_row.get("Country") or ""),
            "track_temp_c": first_row.get("AvgTrackTemp"),
            "air_temp_c": first_row.get("AvgAirTemp"),
            "is_wet": bool(first_row.get("IsWetSession")),
            "sc_count": int(first_row.get("SC_Count") or 0),
            "vsc_count": int(first_row.get("VSC_Count") or 0),
            "podium": _to_json_safe(podium, max_rows=3),
            "roster": _to_json_safe(roster, max_rows=25),
            "default_drivers": default_drivers,
        }
    finally:
        con.close()


LOCATION_ALIASES = {
    "monte carlo": "monaco",
    "monaco": "monte carlo",
    "madring": "madrid",
    "madrid": "madring",
    "kuala lumpur": "sepang",
    "sepang": "kuala lumpur",
    "barcelona": "montmeló",
}


def _load_real_circuit_blueprint(con: duckdb.DuckDBPyConnection, location: str) -> tuple[pd.DataFrame, list[dict], float]:
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    loc_key = location.lower().strip()
    candidates = [loc_key]
    if loc_key in LOCATION_ALIASES:
        candidates.append(LOCATION_ALIASES[loc_key])

    bp_df = pd.DataFrame()
    corners_list = []
    rot_deg = 0.0
    matched_key = loc_key

    if "circuit_blueprints" in tables:
        for cand in candidates:
            bp_df = con.execute(
                "SELECT * FROM circuit_blueprints WHERE LocationKey = ? ORDER BY Distance",
                [cand],
            ).df()
            if not bp_df.empty:
                matched_key = cand
                rot_deg = float(bp_df["RotationDeg"].iloc[0] or 0.0)
                break

    if "circuit_corners_v2" in tables:
        c_df = con.execute(
            "SELECT Number, CornerLabel, Distance, X, Y, LabelX, LabelY FROM circuit_corners_v2 WHERE LocationKey = ? ORDER BY Number",
            [matched_key],
        ).df()
        if not c_df.empty:
            corners_list = c_df.to_dict(orient="records")

    return bp_df, corners_list, rot_deg


# ---------------------------------------------------------------------------
# 2. Universal 2024-2026 Telemetry Engine (Real FIA GPS + Real 20Hz Templates)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=128)
def _cached_universal_telemetry(
    year: int, round_num: int, session_type: str, drivers_csv: str
) -> dict:
    overview = _cached_session_overview(year, round_num, session_type)
    roster_map = {r["Driver"]: r for r in overview["roster"]}

    requested = [d.strip().upper() for d in drivers_csv.split(",") if d.strip()]
    if not requested:
        requested = overview["default_drivers"]

    con = _get_db()
    try:
        diagnostics = []
        valid_drivers = []
        for drv in requested:
            r_info = roster_map.get(drv)
            if not r_info:
                diagnostics.append({
                    "Driver": drv,
                    "Severity": "error",
                    "Message": f"{drv} was not entered in the {year} {overview['event_name']}.",
                })
                continue

            if r_info["TotalLaps"] == 0 or r_info.get("BestLap_Sec") is None:
                diagnostics.append({
                    "Driver": drv,
                    "Severity": "error",
                    "Message": f"{r_info['FullName']} ({drv} — {r_info['Team']}) cannot be plotted: 0 timed laps recorded (Official FIA Status: {r_info['Status']}).",
                })
            else:
                valid_drivers.append(drv)
                if not r_info["IsClassified"]:
                    diagnostics.append({
                        "Driver": drv,
                        "Severity": "warning",
                        "Message": f"{r_info['FullName']} ({drv}) did not finish classified after {r_info['TotalLaps']} laps (Official FIA Status: {r_info['Status']}). Plotting fastest lap completed prior to incident.",
                    })

        if not valid_drivers:
            return {
                "year": year, "round_num": round_num,
                "event_name": overview["event_name"], "location": overview["location"],
                "country": overview["country"], "data_tier": "No Valid Laps",
                "diagnostics": diagnostics, "telemetry": [], "corners": [],
                "corner_speeds": [], "dominance": [], "driver_dna": [],
            }

        bp_df, corners_v2, rot_deg = _load_real_circuit_blueprint(con, overview["location"])
        placeholders = ", ".join(["?"] * len(valid_drivers))

        # Tier 1: Check pre-synced 20Hz driver-specific telemetry_cache
        t1_df = con.execute(
            f"""
            WITH fastest_laps AS (
                SELECT
                    Driver, LapNumber,
                    ROW_NUMBER() OVER (PARTITION BY Driver ORDER BY LapTime_Sec ASC NULLS LAST) AS rn
                FROM (
                    SELECT DISTINCT Driver, LapNumber, LapTime_Sec
                    FROM telemetry_cache
                    WHERE Year = ? AND RoundNumber = ? AND SessionType = ?
                      AND Driver IN ({placeholders})
                )
            )
            SELECT
                t.Driver, t.LapNumber, t.LapTime_Sec, t.Compound,
                t.Distance, t.Speed, t.Throttle, t.Brake_Pct, t.nGear,
                t.Time_Sec, t.X, t.Y
            FROM telemetry_cache t
            INNER JOIN fastest_laps f
              ON t.Driver = f.Driver AND t.LapNumber = f.LapNumber AND f.rn = 1
            WHERE t.Year = ? AND t.RoundNumber = ? AND t.SessionType = ?
            ORDER BY t.Driver, t.Distance
            """,
            [year, round_num, session_type] + valid_drivers + [year, round_num, session_type],
        ).df()

        if not t1_df.empty and t1_df["Driver"].nunique() == len(valid_drivers):
            # Rotate raw GPS X/Y to match official FIA broadcast orientation
            rx, ry = _rotate_xy(t1_df["X"].to_numpy(dtype=float), t1_df["Y"].to_numpy(dtype=float), rot_deg)
            t1_df["X"] = np.round(rx, 1)
            t1_df["Y"] = np.round(ry, 1)

            corners_use = corners_v2
            if not corners_use:
                c_old = con.execute(
                    "SELECT Number, CornerLabel, Distance, X, Y FROM circuit_corners_cache WHERE Year = ? AND RoundNumber = ? ORDER BY Number",
                    [year, round_num],
                ).df()
                if not c_old.empty:
                    rcx, rcy = _rotate_xy(c_old["X"].to_numpy(dtype=float), c_old["Y"].to_numpy(dtype=float), rot_deg)
                    c_old["X"], c_old["Y"] = np.round(rcx, 1), np.round(rcy, 1)
                    c_old["LabelX"], c_old["LabelY"] = c_old["X"], c_old["Y"]
                    corners_use = c_old.to_dict(orient="records")

            return _process_aligned_telemetry(
                t1_df, corners_use, roster_map, overview, diagnostics,
                "Tier 1: Official 20Hz Car Sensor Telemetry + FIA GPS Map",
            )

        # Tier 2: Modulate Real FIA 20Hz Circuit Blueprint with each driver's exact Sector Splits & Speed Traps
        best_laps_df = con.execute(
            f"""
            WITH ranked_laps AS (
                SELECT
                    Driver, OfficialTeamName, LapNumber, Compound, TyreLife,
                    LapTime_Sec, Sector1_Sec, Sector2_Sec, Sector3_Sec,
                    SpeedI1, SpeedI2, SpeedFL, SpeedST,
                    ROW_NUMBER() OVER (
                        PARTITION BY Driver
                        ORDER BY CASE WHEN IsRepresentativePace THEN 0 ELSE 1 END, LapTime_Sec ASC
                    ) AS rn
                FROM clean_laps
                WHERE Year = ? AND RoundNumber = ? AND SessionType = ?
                  AND Driver IN ({placeholders})
                  AND LapTime_Sec IS NOT NULL AND NOT isnan(LapTime_Sec)
            )
            SELECT * FROM ranked_laps WHERE rn = 1
            """,
            [year, round_num, session_type] + valid_drivers,
        ).df()

        if best_laps_df.empty or bp_df.empty:
            return {
                "year": year, "round_num": round_num,
                "event_name": overview["event_name"], "location": overview["location"],
                "country": overview["country"],
                "data_tier": "Run sync_all_circuit_maps.py to populate FIA GPS Blueprints",
                "diagnostics": diagnostics, "telemetry": [], "corners": [],
                "corner_speeds": [], "dominance": [], "driver_dna": [],
            }

        dist_grid = bp_df["Distance"].to_numpy(dtype=float)
        x_base = bp_df["X"].to_numpy(dtype=float)
        y_base = bp_df["Y"].to_numpy(dtype=float)
        ref_spd = bp_df["RefSpeed"].to_numpy(dtype=float)
        ref_thr = bp_df["RefThrottle"].to_numpy(dtype=float)
        ref_brk = bp_df["RefBrake"].to_numpy(dtype=float)
        ref_gear = bp_df["RefGear"].to_numpy(dtype=int)
        track_len_m = float(bp_df["TrackLength_m"].iloc[0])
        n_pts = len(dist_grid)

        ref_peak = max(float(np.max(ref_spd)), 280.0)
        is_straight = ref_spd >= (0.86 * ref_peak)

        fastest_row = best_laps_df.sort_values("LapTime_Sec").iloc[0]
        ref_lap = float(fastest_row["LapTime_Sec"])
        ref_s1 = float(fastest_row["Sector1_Sec"]) if pd.notna(fastest_row["Sector1_Sec"]) else ref_lap * 0.31
        ref_s2 = float(fastest_row["Sector2_Sec"]) if pd.notna(fastest_row["Sector2_Sec"]) else ref_lap * 0.38
        ref_s3 = float(fastest_row["Sector3_Sec"]) if pd.notna(fastest_row["Sector3_Sec"]) else (ref_lap - ref_s1 - ref_s2)

        s1_mask = dist_grid <= (track_len_m / 3.0)
        s2_mask = (dist_grid > (track_len_m / 3.0)) & (dist_grid <= (2.0 * track_len_m / 3.0))
        s3_mask = dist_grid > (2.0 * track_len_m / 3.0)

        synth_rows = []
        for idx_drv, (_, lap_row) in enumerate(best_laps_df.iterrows()):
            drv = str(lap_row["Driver"])
            lap_num = int(lap_row["LapNumber"] or 1)
            lap_time = float(lap_row["LapTime_Sec"])
            comp = str(lap_row["Compound"] or "MEDIUM")

            s1 = float(lap_row["Sector1_Sec"]) if pd.notna(lap_row["Sector1_Sec"]) else lap_time * 0.31
            s2 = float(lap_row["Sector2_Sec"]) if pd.notna(lap_row["Sector2_Sec"]) else lap_time * 0.38
            s3 = float(lap_row["Sector3_Sec"]) if pd.notna(lap_row["Sector3_Sec"]) else (lap_time - s1 - s2)

            v_st = float(lap_row["SpeedST"]) if pd.notna(lap_row["SpeedST"]) and lap_row["SpeedST"] > 120 else ref_peak
            v_i1 = float(lap_row["SpeedI1"]) if pd.notna(lap_row["SpeedI1"]) and lap_row["SpeedI1"] > 120 else v_st * 0.96
            v_i2 = float(lap_row["SpeedI2"]) if pd.notna(lap_row["SpeedI2"]) and lap_row["SpeedI2"] > 120 else v_st * 0.95

            # Straight-line scaling from real FIA Speed Traps; cornering scaling from real Sector Times
            st_mult = np.where(s1_mask, v_i1 / ref_peak, np.where(s2_mask, v_i2 / ref_peak, v_st / ref_peak))
            st_mult = np.clip(st_mult, 0.90, 1.08)

            c_mult = np.where(
                s1_mask, ref_s1 / max(s1, 1.0),
                np.where(s2_mask, ref_s2 / max(s2, 1.0), ref_s3 / max(s3, 1.0)),
            )
            c_mult = np.clip(c_mult, 0.93, 1.06)

            spd_trace = np.where(is_straight, ref_spd * st_mult, ref_spd * c_mult)
            spd_trace = np.clip(spd_trace, 65.0, max(v_st, ref_peak) + 3.0)

            # Exact cumulative elapsed time matching real S1, S2, S3 splits from clean_laps
            inv_v = 1.0 / np.maximum(spd_trace, 50.0)
            dt_raw = inv_v * np.gradient(dist_grid)
            dt_raw[0] = 0.0
            dt_raw[s1_mask] *= s1 / max(1e-6, float(dt_raw[s1_mask].sum()))
            dt_raw[s2_mask] *= s2 / max(1e-6, float(dt_raw[s2_mask].sum()))
            dt_raw[s3_mask] *= s3 / max(1e-6, float(dt_raw[s3_mask].sum()))
            t_trace = np.cumsum(dt_raw)

            # Preserve real circuit Throttle, Brake, and Gear profiles
            thr_trace = np.clip(ref_thr + (c_mult - 1.0) * 35.0, 0.0, 100.0)
            brk_trace = np.where(ref_brk > 50.0, 100.0, 0.0)
            gear_trace = np.clip(np.round(ref_gear + (spd_trace - ref_spd) / 38.0), 1, 8).astype(int)

            for i in range(n_pts):
                synth_rows.append({
                    "Driver": drv,
                    "LapNumber": lap_num,
                    "LapTime_Sec": round(lap_time, 3),
                    "Compound": comp,
                    "Distance": round(float(dist_grid[i]), 1),
                    "Speed": round(float(spd_trace[i]), 1),
                    "Throttle": round(float(thr_trace[i]), 1),
                    "Brake_Pct": round(float(brk_trace[i]), 1),
                    "nGear": int(gear_trace[i]),
                    "Time_Sec": round(float(t_trace[i]), 4),
                    "X": float(x_base[i]),
                    "Y": float(y_base[i]),
                })

        return _process_aligned_telemetry(
            pd.DataFrame(synth_rows),
            corners_v2,
            roster_map,
            overview,
            diagnostics,
            f"Tier 2: Official FIA GPS Blueprint + 20Hz Sector-Calibrated Telemetry ({overview['location']})",
        )
    finally:
        con.close()


def _process_aligned_telemetry(
    raw_df: pd.DataFrame,
    corners_list: list[dict],
    roster_map: dict,
    overview: dict,
    diagnostics: list,
    data_tier: str,
) -> dict:
    max_dist = float(raw_df["Distance"].max())
    n_grid = 380
    common_dist = np.linspace(0.0, max_dist, n_grid)

    drv_lap_times = (
        raw_df.groupby("Driver", as_index=False)["LapTime_Sec"]
        .first()
        .sort_values("LapTime_Sec")
    )
    ordered_drivers = list(drv_lap_times["Driver"])
    ref_driver = ordered_drivers[0]

    interp_by_drv = {}
    dna_rows = []

    for drv in ordered_drivers:
        grp = raw_df[raw_df["Driver"] == drv].sort_values("Distance")
        d_arr = grp["Distance"].to_numpy(dtype=float)
        spd_arr = np.interp(common_dist, d_arr, grp["Speed"].to_numpy(dtype=float))
        thr_arr = np.interp(common_dist, d_arr, grp["Throttle"].to_numpy(dtype=float))
        brk_arr = np.interp(common_dist, d_arr, grp["Brake_Pct"].to_numpy(dtype=float))
        gear_arr = np.round(np.interp(common_dist, d_arr, grp["nGear"].to_numpy(dtype=float))).astype(int)

        t_raw = grp["Time_Sec"].to_numpy(dtype=float)
        t_raw = t_raw - t_raw[0]
        time_arr = np.interp(common_dist, d_arr, t_raw)

        lap_time_sec = float(grp["LapTime_Sec"].iloc[0])
        if time_arr[-1] > 0:
            time_arr = time_arr * (lap_time_sec / time_arr[-1])

        x_arr = np.interp(common_dist, d_arr, grp["X"].to_numpy(dtype=float))
        y_arr = np.interp(common_dist, d_arr, grp["Y"].to_numpy(dtype=float))

        r_meta = roster_map.get(drv, {})
        interp_by_drv[drv] = {
            "LapNumber": int(grp["LapNumber"].iloc[0]),
            "LapTime_Sec": round(lap_time_sec, 3),
            "Compound": str(grp["Compound"].iloc[0]),
            "Speed": spd_arr,
            "Throttle": thr_arr,
            "Brake_Pct": brk_arr,
            "nGear": gear_arr,
            "Time_Sec": time_arr,
            "X": x_arr,
            "Y": y_arr,
        }

        dna_rows.append({
            "Driver": drv,
            "FullName": r_meta.get("FullName", drv),
            "Team": r_meta.get("Team", ""),
            "Status": r_meta.get("Status", "Finished"),
            "FastestLap": int(grp["LapNumber"].iloc[0]),
            "LapTime_Sec": round(lap_time_sec, 3),
            "GapToRef_Sec": round(lap_time_sec - float(drv_lap_times["LapTime_Sec"].iloc[0]), 3),
            "Compound": str(grp["Compound"].iloc[0]),
            "TopSpeed_kmh": round(float(np.max(spd_arr)), 1),
            "MinApexSpeed_kmh": round(float(np.min(spd_arr)), 1),
            "FullThrottle_Pct": round(float(np.mean(thr_arr >= 98.0) * 100.0), 1),
            "HeavyBraking_Pct": round(float(np.mean(brk_arr > 10.0) * 100.0), 1),
            "GearShifts": int(np.sum(np.abs(np.diff(gear_arr)) > 0)),
        })

    ref_time = interp_by_drv[ref_driver]["Time_Sec"]
    aligned_rows = []
    for drv in ordered_drivers:
        info = interp_by_drv[drv]
        delta_arr = info["Time_Sec"] - ref_time
        for i in range(n_grid):
            aligned_rows.append({
                "Driver": drv,
                "LapNumber": info["LapNumber"],
                "LapTime_Sec": info["LapTime_Sec"],
                "Compound": info["Compound"],
                "Distance": round(float(common_dist[i]), 1),
                "Speed": round(float(info["Speed"][i]), 1),
                "Delta_Sec": round(float(delta_arr[i]), 3),
                "Throttle": round(float(info["Throttle"][i]), 1),
                "Brake_Pct": round(float(info["Brake_Pct"][i]), 1),
                "nGear": int(info["nGear"][i]),
            })

    ref_x = interp_by_drv[ref_driver]["X"]
    ref_y = interp_by_drv[ref_driver]["Y"]
    dominance_rows = []
    for i in range(n_grid):
        best_drv = max(ordered_drivers, key=lambda d: interp_by_drv[d]["Speed"][i])
        dominance_rows.append({
            "Distance": round(float(common_dist[i]), 1),
            "X": round(float(ref_x[i]), 1),
            "Y": round(float(ref_y[i]), 1),
            "FastestDriver": best_drv,
            "Speed": round(float(interp_by_drv[best_drv]["Speed"][i]), 1),
        })

    corner_speed_rows = []
    for c in corners_list:
        c_dist = float(c["Distance"])
        mask = (common_dist >= c_dist - 45.0) & (common_dist <= c_dist + 45.0)
        if not np.any(mask):
            idx_closest = int(np.argmin(np.abs(common_dist - c_dist)))
            mask = np.zeros(n_grid, dtype=bool)
            mask[idx_closest] = True

        row_c = {"Corner": c.get("CornerLabel") or f"T{c['Number']}", "Distance_m": round(c_dist, 0)}
        for drv in ordered_drivers:
            row_c[drv] = round(float(np.min(interp_by_drv[drv]["Speed"][mask])), 1)
        corner_speed_rows.append(row_c)

    return {
        "year": overview["year"],
        "round_num": overview["round_num"],
        "event_name": overview["event_name"],
        "location": overview["location"],
        "country": overview["country"],
        "track_length_m": round(max_dist, 0),
        "corner_count": len(corners_list),
        "reference_driver": ref_driver,
        "ordered_drivers": ordered_drivers,
        "data_tier": data_tier,
        "diagnostics": diagnostics,
        "telemetry": aligned_rows,
        "corners": corners_list,
        "corner_speeds": corner_speed_rows,
        "dominance": dominance_rows,
        "driver_dna": dna_rows,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        for yr in (2024, 2025, 2026):
            tracks = _cached_tracks_for_year(yr)
            for t in tracks[:4]:
                ov = _cached_session_overview(yr, t["RoundNumber"], "R")
                _cached_universal_telemetry(yr, t["RoundNumber"], "R", ",".join(ov["default_drivers"]))
    except Exception:
        pass
    yield


app = FastAPI(
    title="Apex Racing F1 — Six Sigma Black Belt Pit-Wall Console v5.0",
    version="5.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
(FRONTEND_DIR / "tabs").mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

from src.api.analytics_router import router as analytics_router
app.include_router(analytics_router)

from src.api.ml_ai_router import router as ml_ai_router
app.include_router(ml_ai_router)

@app.get("/")
def serve_pitwall_ui():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "duckdb_lakehouse_ready": DB_PATH.exists(),
        "gemini_api_key_configured": bool(os.getenv("GEMINI_API_KEY")),
    }

@app.get("/api/tracks/{year}")
def endpoint_tracks_only(year: int):
    return {"year": year, "tracks": _cached_tracks_for_year(year)}

@app.get("/api/bootstrap")
def endpoint_bootstrap(
    year: int = 2025, round_num: Optional[int] = None, session_type: str = "R"
):
    tracks = _cached_tracks_for_year(year)
    active_round = round_num or (tracks[0]["RoundNumber"] if tracks else 1)
    valid_rounds = {t["RoundNumber"] for t in tracks}
    if active_round not in valid_rounds and tracks:
        active_round = tracks[0]["RoundNumber"]

    overview = _cached_session_overview(year, active_round, session_type)
    return {
        "year": year,
        "active_round": active_round,
        "session_type": session_type,
        "tracks": tracks,
        "overview": overview,
    }


@app.get("/api/telemetry/{year}/{round_num}")
def endpoint_universal_telemetry(
    year: int,
    round_num: int,
    drivers: str = "",
    session_type: str = "R",
):
    try:
        return _cached_universal_telemetry(year, round_num, session_type, drivers)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))