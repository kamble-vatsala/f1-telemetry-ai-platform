import os
import duckdb
import numpy as np
import pandas as pd
import fastf1

CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

DB_PATH = "data/f1_lakehouse.duckdb"

# Showcase races to pre-pack with 100% real 20Hz FastF1 telemetry & GPS circuit maps
SHOWCASE_SESSIONS = [
    (2026, 20, 'R'),  # 2026 São Paulo GP
    (2026, 19, 'R'),  # 2026 Mexico City GP
    (2026, 1, 'R'),   # 2026 Australian GP
    (2025, 24, 'R'),  # 2025 Abu Dhabi GP
    (2024, 21, 'R'),  # 2024 São Paulo GP
]


def sync_real_telemetry_to_duckdb():
    con = duckdb.connect(DB_PATH)

    con.execute("""
        CREATE TABLE IF NOT EXISTS telemetry_cache (
            Year INTEGER,
            RoundNumber INTEGER,
            SessionType VARCHAR,
            Driver VARCHAR,
            LapNumber INTEGER,
            LapTime_Sec DOUBLE,
            Compound VARCHAR,
            TyreLife INTEGER,
            Distance DOUBLE,
            Speed DOUBLE,
            Throttle DOUBLE,
            Brake_Pct DOUBLE,
            nGear INTEGER,
            DRS_Active INTEGER,
            Time_Sec DOUBLE,
            X DOUBLE,
            Y DOUBLE
        );
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS circuit_corners_cache (
            Year INTEGER,
            RoundNumber INTEGER,
            Number INTEGER,
            Letter VARCHAR,
            Distance DOUBLE,
            X DOUBLE,
            Y DOUBLE,
            CornerLabel VARCHAR
        );
    """)

    for year, rnd, ses_type in SHOWCASE_SESSIONS:
        print(f"📡 Syncing real 20Hz FastF1 telemetry & GPS map for {year} Round {rnd} ({ses_type})...")
        try:
            session = fastf1.get_session(year, rnd, ses_type)
            session.load(laps=True, telemetry=True, weather=False, messages=False)
            laps = session.laps
        except Exception as e:
            print(f"⚠️ Skipping {year} R{rnd}: {e}")
            continue

        con.execute(f"DELETE FROM telemetry_cache WHERE Year={year} AND RoundNumber={rnd} AND SessionType='{ses_type}'")
        con.execute(f"DELETE FROM circuit_corners_cache WHERE Year={year} AND RoundNumber={rnd}")

        # 1. Save Official Circuit Corners
        try:
            c_info = session.get_circuit_info()
            if c_info is not None and c_info.corners is not None:
                c_df = c_info.corners[['Number', 'Letter', 'Distance', 'X', 'Y']].copy()
                c_df['Year'] = year
                c_df['RoundNumber'] = rnd
                c_df['Letter'] = c_df['Letter'].fillna('')
                c_df['CornerLabel'] = "T" + c_df['Number'].astype(str) + c_df['Letter']
                c_df = c_df[['Year', 'RoundNumber', 'Number', 'Letter', 'Distance', 'X', 'Y', 'CornerLabel']]
                con.execute("INSERT INTO circuit_corners_cache SELECT * FROM c_df")
        except Exception:
            pass

        # 2. Save Resampled 500-Point Fastest-Lap 20Hz Telemetry per Driver
        tel_rows = []
        for drv in laps['Driver'].unique():
            d_laps = laps.pick_drivers(drv)
            if d_laps.empty:
                continue
            fast_lap = d_laps.pick_fastest()
            if fast_lap is None or pd.isna(fast_lap['LapTime']):
                continue
            try:
                tel = fast_lap.get_telemetry().add_distance()
            except Exception:
                continue
            if tel.empty:
                continue

            max_d = float(tel['Distance'].max())
            d_grid = np.linspace(0, max_d, 500)
            t_sec = tel['Time'].dt.total_seconds().values
            brk_pct = (tel['Brake'].astype(int) * 100).values
            drs_act = tel['DRS'].isin([10, 12, 14]).astype(int).values

            df_drv = pd.DataFrame({
                'Year': year,
                'RoundNumber': rnd,
                'SessionType': ses_type,
                'Driver': drv,
                'LapNumber': int(fast_lap['LapNumber']),
                'LapTime_Sec': round(float(fast_lap['LapTime'].total_seconds()), 3),
                'Compound': str(fast_lap.get('Compound', 'UNK')),
                'TyreLife': int(fast_lap.get('TyreLife', 0)) if pd.notna(fast_lap.get('TyreLife')) else 0,
                'Distance': np.round(d_grid, 2),
                'Speed': np.round(np.interp(d_grid, tel['Distance'], tel['Speed']), 1),
                'Throttle': np.round(np.interp(d_grid, tel['Distance'], tel['Throttle']), 1),
                'Brake_Pct': np.round(np.interp(d_grid, tel['Distance'], brk_pct), 1),
                'nGear': np.round(np.interp(d_grid, tel['Distance'], tel['nGear'])).astype(int),
                'DRS_Active': np.round(np.interp(d_grid, tel['Distance'], drs_act)).astype(int),
                'Time_Sec': np.round(np.interp(d_grid, tel['Distance'], t_sec), 4),
                'X': np.round(np.interp(d_grid, tel['Distance'], tel['X']), 1),
                'Y': np.round(np.interp(d_grid, tel['Distance'], tel['Y']), 1)
            })
            tel_rows.append(df_drv)

        if tel_rows:
            batch_df = pd.concat(tel_rows, ignore_index=True)
            con.execute("INSERT INTO telemetry_cache SELECT * FROM batch_df")
            print(f"✅ Stored {len(tel_rows)} real driver telemetry traces & GPS map for {year} R{rnd}!")

    con.close()
    db_mb = os.path.getsize(DB_PATH) / (1024 * 1024)
    print(f"\n🎉 Telemetry Sync Complete! Updated DuckDB size: {db_mb:.2f} MB")


if __name__ == "__main__":
    sync_real_telemetry_to_duckdb()