import os
import time
from pathlib import Path

import duckdb
import fastf1
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "f1_lakehouse.duckdb"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"

os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))


def rotate_xy(x_arr: np.ndarray, y_arr: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    rad = np.deg2rad(angle_deg)
    rot_mat = np.array([[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]])
    xy = np.column_stack([x_arr, y_arr])
    rot = np.matmul(xy, rot_mat)
    return rot[:, 0], rot[:, 1]


def auto_detect_corners(d_grid: np.ndarray, rot_x: np.ndarray, rot_y: np.ndarray, ref_spd: np.ndarray) -> list[dict]:
    """Fallback apex detector from real 20Hz speed dips if FastF1 lacks corner metadata on new tracks."""
    corners = []
    n = len(ref_spd)
    last_idx = -25
    num = 1
    for i in range(10, n - 10):
        window = ref_spd[i - 8 : i + 9]
        if ref_spd[i] == np.min(window) and (np.max(window) - ref_spd[i]) >= 18.0 and (i - last_idx) >= 20:
            corners.append({
                "Number": num,
                "CornerLabel": f"T{num}",
                "Distance": round(float(d_grid[i]), 1),
                "X": round(float(rot_x[i]), 1),
                "Y": round(float(rot_y[i]), 1),
                "LabelX": round(float(rot_x[i]), 1),
                "LabelY": round(float(rot_y[i]), 1),
            })
            last_idx = i
            num += 1
    return corners


def sync_all_circuit_blueprints():
    con = duckdb.connect(str(DB_PATH))

    con.execute("""
        CREATE TABLE IF NOT EXISTS circuit_blueprints (
            LocationKey VARCHAR, Location VARCHAR, EventName VARCHAR,
            SourceYear INTEGER, SourceRound INTEGER, RotationDeg DOUBLE,
            TrackLength_m DOUBLE, Distance DOUBLE, X DOUBLE, Y DOUBLE,
            RefSpeed DOUBLE, RefThrottle DOUBLE, RefBrake DOUBLE, RefGear INTEGER
        );
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS circuit_corners_v2 (
            LocationKey VARCHAR, Number INTEGER, CornerLabel VARCHAR,
            Distance DOUBLE, X DOUBLE, Y DOUBLE, LabelX DOUBLE, LabelY DOUBLE
        );
    """)

    candidates_df = con.execute("""
        SELECT DISTINCT Location, EventName, Year, RoundNumber
        FROM clean_results
        WHERE Location IS NOT NULL
        ORDER BY Location, Year DESC, RoundNumber ASC
    """).df()

    existing_keys = {
        r[0] for r in con.execute("SELECT DISTINCT LocationKey FROM circuit_blueprints").fetchall()
    }

    grouped = candidates_df.groupby("Location", sort=True)
    missing = [(loc, grp) for loc, grp in grouped if str(loc).lower().strip() not in existing_keys]
    print(f"🗺️  Already synced: {len(existing_keys)} circuits | Missing to sync now: {len(missing)}")

    for idx, (loc, grp) in enumerate(missing, start=1):
        loc_key = str(loc).lower().strip()
        synced_ok = False

        for _, row in grp.iterrows():
            yr, rnd, ev_name = int(row["Year"]), int(row["RoundNumber"]), str(row["EventName"])
            for sess_code in ("R", "Q"):
                print(f"[{idx}/{len(missing)}] 📡 Fetching {loc} ({yr} R{rnd} {ev_name} [{sess_code}])...")
                try:
                    session = fastf1.get_session(yr, rnd, sess_code)
                    session.load(laps=True, telemetry=True, weather=False, messages=False)
                    fast_lap = session.laps.pick_fastest()
                    if fast_lap is None or pd.isna(fast_lap["LapTime"]):
                        continue

                    tel = fast_lap.get_telemetry().add_distance()
                    if tel.empty or "X" not in tel.columns or tel["X"].isna().all():
                        continue

                    # Safely read circuit_info without failing the whole sync on brand-new 2026 circuits
                    rot_deg = 0.0
                    c_info = None
                    try:
                        c_info = session.get_circuit_info()
                        if c_info is not None and c_info.rotation is not None:
                            rot_deg = float(c_info.rotation)
                    except Exception as ci_err:
                        print(f"   ℹ️ circuit_info unavailable ({ci_err}); using 20Hz speed-dip corner detection.")

                    max_d = float(tel["Distance"].max())
                    d_grid = np.linspace(0.0, max_d, 500)
                    raw_x = np.interp(d_grid, tel["Distance"], tel["X"])
                    raw_y = np.interp(d_grid, tel["Distance"], tel["Y"])
                    rot_x, rot_y = rotate_xy(raw_x, raw_y, rot_deg)

                    ref_spd = np.interp(d_grid, tel["Distance"], tel["Speed"])
                    ref_thr = np.interp(d_grid, tel["Distance"], tel["Throttle"])
                    ref_brk = np.interp(d_grid, tel["Distance"], (tel["Brake"].astype(int) * 100.0).to_numpy())
                    ref_gear = np.round(np.interp(d_grid, tel["Distance"], tel["nGear"])).astype(int)

                    bp_df = pd.DataFrame({
                        "LocationKey": loc_key, "Location": str(loc), "EventName": ev_name,
                        "SourceYear": yr, "SourceRound": rnd, "RotationDeg": round(rot_deg, 2),
                        "TrackLength_m": round(max_d, 1), "Distance": np.round(d_grid, 2),
                        "X": np.round(rot_x, 1), "Y": np.round(rot_y, 1),
                        "RefSpeed": np.round(ref_spd, 1), "RefThrottle": np.round(ref_thr, 1),
                        "RefBrake": np.round(ref_brk, 1), "RefGear": ref_gear,
                    })
                    con.execute("DELETE FROM circuit_blueprints WHERE LocationKey = ?", [loc_key])
                    con.execute("INSERT INTO circuit_blueprints SELECT * FROM bp_df")

                    c_rows = []
                    if c_info is not None and c_info.corners is not None and not c_info.corners.empty:
                        for _, c in c_info.corners.iterrows():
                            cx, cy = float(c["X"]), float(c["Y"])
                            ang = float(c["Angle"]) if pd.notna(c.get("Angle")) else 0.0
                            off_x, off_y = rotate_xy(np.array([450.0]), np.array([0.0]), ang)
                            rcx, rcy = rotate_xy(np.array([cx]), np.array([cy]), rot_deg)
                            rlx, rly = rotate_xy(np.array([cx + off_x[0]]), np.array([cy + off_y[0]]), rot_deg)
                            letter = str(c["Letter"]) if pd.notna(c.get("Letter")) else ""
                            c_rows.append({
                                "LocationKey": loc_key, "Number": int(c["Number"]),
                                "CornerLabel": f"T{int(c['Number'])}{letter}",
                                "Distance": round(float(c["Distance"]), 1),
                                "X": round(float(rcx[0]), 1), "Y": round(float(rcy[0]), 1),
                                "LabelX": round(float(rlx[0]), 1), "LabelY": round(float(rly[0]), 1),
                            })
                    else:
                        c_rows = [{"LocationKey": loc_key, **cr} for cr in auto_detect_corners(d_grid, rot_x, rot_y, ref_spd)]

                    if c_rows:
                        c_df = pd.DataFrame(c_rows)
                        con.execute("DELETE FROM circuit_corners_v2 WHERE LocationKey = ?", [loc_key])
                        con.execute("INSERT INTO circuit_corners_v2 SELECT * FROM c_df")

                    con.commit()
                    print(f"   ✅ Saved real FIA GPS map for {loc} ({len(c_rows)} corners, rot={rot_deg:.0f}°)!")
                    synced_ok = True
                    time.sleep(1.5)
                    break
                except Exception as exc:
                    print(f"   ⚠️ {yr} R{rnd} [{sess_code}] failed: {exc}")
                    time.sleep(1.5)
            if synced_ok:
                break

    con.close()
    print("\n🎉 Missing circuit sync complete!")


if __name__ == "__main__":
    sync_all_circuit_blueprints()