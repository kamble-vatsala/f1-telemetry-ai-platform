import os
import numpy as np
import pandas as pd
import fastf1

CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


def analyze_session_telemetry(year: int, round_num: int, session_type: str, driver_lap_list: list):
    session = fastf1.get_session(year, round_num, session_type)
    session.load(laps=True, telemetry=True, weather=False, messages=False)

    telemetry_frames = []
    loaded_laps = []
    summary_rows = []

    for drv, lap_choice in driver_lap_list:
        drv_laps = session.laps.pick_drivers(drv)
        if drv_laps.empty:
            continue

        if lap_choice is None or str(lap_choice).lower() == 'fastest':
            lap = drv_laps.pick_fastest()
            if lap is None or pd.isna(lap['LapTime']):
                continue
            lap_num = int(lap['LapNumber'])
        else:
            lap_num = int(lap_choice)
            lap_subset = drv_laps[drv_laps['LapNumber'] == lap_num]
            if lap_subset.empty:
                continue
            lap = lap_subset.iloc[0]

        tel = lap.get_telemetry().add_distance()
        if tel.empty:
            continue

        lap_sec = lap['LapTime'].total_seconds()
        label = f"{drv} (L{lap_num} - {lap_sec:.3f}s)"

        tel['Driver'] = drv
        tel['LapNumber'] = lap_num
        tel['Label'] = label
        tel['Time_Sec'] = tel['Time'].dt.total_seconds()
        tel['Brake_Pct'] = tel['Brake'].astype(int) * 100
        tel['DRS_Active'] = tel['DRS'].isin([10, 12, 14]).astype(int)

        full_throttle_pct = (tel['Throttle'] >= 98).mean() * 100
        braking_pct = (tel['Brake_Pct'] > 0).mean() * 100
        coasting_pct = ((tel['Throttle'] < 98) & (tel['Brake_Pct'] == 0)).mean() * 100
        gear_shifts = int((tel['nGear'].diff().abs() > 0).sum())
        drs_pct = (tel['DRS_Active'] == 1).mean() * 100

        summary_rows.append({
            'Driver': drv,
            'LapNumber': lap_num,
            'Label': label,
            'LapTime_Sec': round(lap_sec, 3),
            'Compound': str(lap.get('Compound', 'UNK')),
            'TyreLife': int(lap.get('TyreLife', 0)) if pd.notna(lap.get('TyreLife')) else 0,
            'TopSpeed_kmh': round(float(tel['Speed'].max()), 1),
            'AvgSpeed_kmh': round(float(tel['Speed'].mean()), 1),
            'FullThrottle_Pct': round(float(full_throttle_pct), 1),
            'Braking_Pct': round(float(braking_pct), 1),
            'Coasting_Pct': round(float(coasting_pct), 1),
            'GearShifts': gear_shifts,
            'DRS_Open_Pct': round(float(drs_pct), 1)
        })

        telemetry_frames.append(tel)
        loaded_laps.append((drv, lap_num, label, lap, tel))

    if not telemetry_frames:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    combined_tel = pd.concat(telemetry_frames, ignore_index=True)
    driver_summary_df = pd.DataFrame(summary_rows)

    ref_drv, ref_lap_num, ref_label, ref_lap, ref_tel = loaded_laps[0]
    max_distance = min([t['Distance'].max() for _, _, _, _, t in loaded_laps])
    dist_grid = np.linspace(0, max_distance, 1000)
    ref_time_interp = np.interp(dist_grid, ref_tel['Distance'], ref_tel['Time_Sec'])

    delta_records = []
    for drv, lap_num, label, lap, tel in loaded_laps:
        drv_time_interp = np.interp(dist_grid, tel['Distance'], tel['Time_Sec'])
        drv_speed_interp = np.interp(dist_grid, tel['Distance'], tel['Speed'])
        drv_x_interp = np.interp(dist_grid, tel['Distance'], tel['X'])
        drv_y_interp = np.interp(dist_grid, tel['Distance'], tel['Y'])

        delta_sec = drv_time_interp - ref_time_interp
        delta_records.append(pd.DataFrame({
            'Distance': dist_grid,
            'Driver': drv,
            'Label': label,
            'Delta_Sec': np.round(delta_sec, 4),
            'Speed_Interp': np.round(drv_speed_interp, 1),
            'X': drv_x_interp,
            'Y': drv_y_interp
        }))

    delta_df = pd.concat(delta_records, ignore_index=True)

    corners_df = pd.DataFrame()
    corner_speeds_rows = []
    try:
        circuit_info = session.get_circuit_info()
        if circuit_info is not None and circuit_info.corners is not None:
            corners_df = circuit_info.corners[['Number', 'Letter', 'Distance', 'X', 'Y']].copy()
            corners_df['CornerLabel'] = "T" + corners_df['Number'].astype(str) + corners_df['Letter'].fillna('')

            for _, c_row in corners_df.iterrows():
                c_dist = c_row['Distance']
                c_label = c_row['CornerLabel']
                for drv, lap_num, label, lap, tel in loaded_laps:
                    window = tel[(tel['Distance'] >= c_dist - 45) & (tel['Distance'] <= c_dist + 45)]
                    if not window.empty:
                        min_spd = float(window['Speed'].min())
                        max_entry_spd = float(window['Speed'].max())
                        min_gear = int(window['nGear'].min())
                        corner_speeds_rows.append({
                            'Corner': c_label,
                            'CornerNumber': int(c_row['Number']),
                            'CornerDistance_m': round(c_dist, 1),
                            'Driver': drv,
                            'Label': label,
                            'ApexMinSpeed_kmh': round(min_spd, 1),
                            'EntrySpeed_kmh': round(max_entry_spd, 1),
                            'MinGear': min_gear
                        })
    except Exception:
        pass

    corner_speeds_df = pd.DataFrame(corner_speeds_rows)

    fastest_row = driver_summary_df.sort_values('LapTime_Sec').iloc[0]
    top_spd_row = driver_summary_df.sort_values('TopSpeed_kmh', ascending=False).iloc[0]

    notes = {
        "f1_context": (
            "In Formula 1, raw lap times only tell you WHO was faster, not WHERE or WHY. "
            "Engineers overlay 20Hz car sensor telemetry (Speed, Throttle, Brake, Gear, DRS) against Track Distance. "
            "Higher 'Apex Minimum Speed' in corners indicates superior aerodynamic downforce and mechanical grip, "
            "while higher 'Top Speed' on straights indicates lower aerodynamic drag or stronger hybrid deployment."
        ),
        "ds_methodology": (
            "1D Spatial Signal Interpolation (np.interp): Because a faster car finishes a lap in fewer seconds than a slower car, "
            "plotting sensors against elapsed time misaligns corners. We integrate velocity over time to map every sample to physical "
            "Distance (meters) and resample all drivers onto a uniform 1,000-point spatial grid to calculate continuous Time Delta (Δt)."
        ),
        "key_insight": (
            f"Fastest analysed lap: {fastest_row['Label']} ({fastest_row['FullThrottle_Pct']}% full throttle, "
            f"{fastest_row['Braking_Pct']}% braking). Highest speed trap reached by {top_spd_row['Driver']} at "
            f"{top_spd_row['TopSpeed_kmh']} km/h."
        )
    }

    return combined_tel, delta_df, corner_speeds_df, driver_summary_df, corners_df, notes