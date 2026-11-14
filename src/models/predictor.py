import numpy as np
import pandas as pd


def simulate_pit_strategies(
    total_laps: int = 60,
    base_pace_sec: float = 75.0,
    deg_rates: dict = None,
    compound_offsets: dict = None,
    pit_loss_sec: float = 22.0
) -> pd.DataFrame:
    if deg_rates is None:
        deg_rates = {"SOFT": 0.12, "MEDIUM": 0.07, "HARD": 0.04}
    if compound_offsets is None:
        compound_offsets = {"SOFT": 0.0, "MEDIUM": 0.45, "HARD": 0.95}

    lap_1_stop_mh = int(total_laps * 0.42)
    lap_1_stop_sm = int(total_laps * 0.32)
    lap_2_stop_a = int(total_laps * 0.28)
    lap_2_stop_b = int(total_laps * 0.64)

    strategies = {
        f"1-Stop: MEDIUM -> HARD (Pit L{lap_1_stop_mh})": [
            (1, lap_1_stop_mh, "MEDIUM"),
            (lap_1_stop_mh + 1, total_laps, "HARD")
        ],
        f"1-Stop: SOFT -> HARD (Pit L{lap_1_stop_sm})": [
            (1, lap_1_stop_sm, "SOFT"),
            (lap_1_stop_sm + 1, total_laps, "HARD")
        ],
        f"2-Stop: SOFT -> MEDIUM -> MEDIUM (Pit L{lap_2_stop_a}, L{lap_2_stop_b})": [
            (1, lap_2_stop_a, "SOFT"),
            (lap_2_stop_a + 1, lap_2_stop_b, "MEDIUM"),
            (lap_2_stop_b + 1, total_laps, "MEDIUM")
        ],
        f"2-Stop Aggressive: SOFT -> MEDIUM -> SOFT (Pit L{lap_2_stop_a}, L{int(total_laps*0.72)})": [
            (1, lap_2_stop_a, "SOFT"),
            (lap_2_stop_a + 1, int(total_laps * 0.72), "MEDIUM"),
            (int(total_laps * 0.72) + 1, total_laps, "SOFT")
        ]
    }

    rows = []
    for strat_name, stints in strategies.items():
        cum_time = 0.0
        for stint_idx, (start_l, end_l, comp) in enumerate(stints):
            deg = deg_rates.get(comp, 0.08)
            offset = compound_offsets.get(comp, 0.5)
            for lap_num in range(start_l, end_l + 1):
                tyre_age = lap_num - start_l + 1
                fuel_penalty = (total_laps - lap_num) * 0.055
                lap_time = base_pace_sec + offset + (deg * tyre_age) + fuel_penalty
                if stint_idx > 0 and lap_num == start_l:
                    lap_time += pit_loss_sec
                cum_time += lap_time
                rows.append({
                    "Strategy": strat_name,
                    "LapNumber": lap_num,
                    "Compound": comp,
                    "TyreAge": tyre_age,
                    "LapTime_Sec": round(lap_time, 3),
                    "CumulativeTime_Sec": round(cum_time, 3)
                })

    sim_df = pd.DataFrame(rows)
    fastest_per_lap = sim_df.groupby("LapNumber")["CumulativeTime_Sec"].transform("min")
    sim_df["GapToFastestStrategy_Sec"] = np.round(sim_df["CumulativeTime_Sec"] - fastest_per_lap, 3)
    return sim_df