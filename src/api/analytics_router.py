from functools import lru_cache
import math
from pathlib import Path
from typing import Any, Optional

import duckdb
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "f1_lakehouse.duckdb"

router = APIRouter(prefix="/api/analytics", tags=["Engineering & Strategy Analytics"])


def _to_json_safe(obj: Any, max_rows: Optional[int] = 2500) -> Any:
    if isinstance(obj, pd.DataFrame):
        df = obj.head(max_rows).copy() if max_rows else obj.copy()
        return _to_json_safe(df.to_dict(orient="records"), max_rows=max_rows)
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
# 1. Track Profile & Pit-Stop Strategy Crossover Simulator
# ---------------------------------------------------------------------------
@lru_cache(maxsize=64)
def _cached_track_profile(year: int, round_num: int) -> dict:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        meta_df = con.execute(
            """
            SELECT
                ANY_VALUE(EventName) AS EventName,
                ANY_VALUE(Location) AS Location,
                MAX(TotalSessionLaps) AS TotalLaps,
                ROUND(MEDIAN(CASE WHEN IsRepresentativePace THEN LapTime_Sec END), 2) AS BasePaceSec
            FROM clean_laps
            WHERE Year = ? AND RoundNumber = ? AND SessionType = 'R'
            """,
            [year, round_num],
        ).df()

        deg_df = con.execute(
            """
            SELECT
                UPPER(Compound) AS Compound,
                ROUND(MEDIAN(TyreDegSlope_SecPerLap), 4) AS MedianDegSlope
            FROM clean_stints
            WHERE Year = ? AND RoundNumber = ? AND SessionType = 'R'
              AND TyreDegSlope_SecPerLap IS NOT NULL
              AND NOT isnan(TyreDegSlope_SecPerLap)
              AND TyreDegSlope_SecPerLap > 0
            GROUP BY UPPER(Compound)
            """,
            [year, round_num],
        ).df()

        row = meta_df.iloc[0].to_dict() if not meta_df.empty else {}
        deg_map = dict(zip(deg_df["Compound"], deg_df["MedianDegSlope"])) if not deg_df.empty else {}
        total_laps = int(row["TotalLaps"]) if pd.notna(row.get("TotalLaps")) else 58
        base_pace = float(row["BasePaceSec"]) if pd.notna(row.get("BasePaceSec")) else 90.0

        return {
            "event_name": str(row.get("EventName") or f"Round {round_num}"),
            "location": str(row.get("Location") or ""),
            "total_laps": max(35, min(78, total_laps)),
            "base_pace_sec": round(max(65.0, min(115.0, base_pace)), 1),
            "soft_deg": round(float(deg_map.get("SOFT", 0.11)), 3),
            "medium_deg": round(float(deg_map.get("MEDIUM", 0.06)), 3),
            "hard_deg": round(float(deg_map.get("HARD", 0.035)), 3),
        }
    finally:
        con.close()


@lru_cache(maxsize=64)
def _cached_pit_simulator(
    total_laps: int,
    base_pace_sec: float,
    soft_deg: float,
    medium_deg: float,
    hard_deg: float,
    pit_loss_sec: float,
) -> dict:
    from src.models.predictor import simulate_pit_strategies

    deg_rates = {"SOFT": soft_deg, "MEDIUM": medium_deg, "HARD": hard_deg}
    compound_offsets = {"SOFT": 0.0, "MEDIUM": 0.45, "HARD": 0.95}
    sim_df = simulate_pit_strategies(
        total_laps=total_laps,
        base_pace_sec=base_pace_sec,
        deg_rates=deg_rates,
        compound_offsets=compound_offsets,
        pit_loss_sec=pit_loss_sec,
    )

    if sim_df is None or sim_df.empty:
        return {"lap_traces": [], "summary": []}

    final_totals = (
        sim_df.sort_values("LapNumber")
        .groupby("Strategy", as_index=False)
        .last()
        .sort_values("CumulativeTime_Sec")
    )
    best_time = float(final_totals["CumulativeTime_Sec"].min())
    ref_lap_pace = best_time / max(1, total_laps)

    sim_df = sim_df.copy()
    sim_df["GapToVirtualWinner_Sec"] = np.round(
        sim_df["CumulativeTime_Sec"] - (sim_df["LapNumber"] * ref_lap_pace), 3
    )
    final_totals["GapToFastest_Sec"] = np.round(final_totals["CumulativeTime_Sec"] - best_time, 2)

    trace_cols = [c for c in ["Strategy", "LapNumber", "Compound", "GapToVirtualWinner_Sec"] if c in sim_df.columns]
    sum_cols = [c for c in ["Strategy", "LapNumber", "Compound", "CumulativeTime_Sec", "GapToFastest_Sec"] if c in final_totals.columns]

    return {
        "lap_traces": _to_json_safe(sim_df[trace_cols], max_rows=1500),
        "summary": _to_json_safe(final_totals[sum_cols], max_rows=15),
    }


class PitSimRequest(BaseModel):
    total_laps: int = Field(58, ge=30, le=80)
    base_pace_sec: float = Field(90.0, ge=60.0, le=130.0)
    soft_deg: float = 0.11
    medium_deg: float = 0.06
    hard_deg: float = 0.035
    pit_loss_sec: float = 22.0


@router.get("/track-profile/{year}/{round_num}")
def endpoint_track_profile(year: int, round_num: int):
    return _cached_track_profile(year, round_num)


@router.post("/pit-simulator")
def endpoint_pit_simulator(payload: PitSimRequest):
    return _cached_pit_simulator(
        payload.total_laps,
        payload.base_pace_sec,
        payload.soft_deg,
        payload.medium_deg,
        payload.hard_deg,
        payload.pit_loss_sec,
    )


# ---------------------------------------------------------------------------
# 2. Race Trace ('Worm Chart'), Position Flow & Overtake Decomposition
# ---------------------------------------------------------------------------
@lru_cache(maxsize=64)
def _cached_race_flow(year: int, round_num: int, session_type: str) -> dict:
    from src.analytics.race_analytics import analyze_race_flow_and_trace

    trace_df, pos_flow_df, overtake_df, notes = analyze_race_flow_and_trace(
        year, round_num, session_type
    )
    t_cols = [c for c in ["Driver", "LapNumber", "GapToVirtualCar_Sec", "Position", "Compound", "TyreLife", "LapTime_Sec"] if c in trace_df.columns]
    p_cols = [c for c in ["Driver", "LapNumber", "Position"] if c in pos_flow_df.columns]
    return {
        "trace": _to_json_safe(trace_df[t_cols], max_rows=2000) if not trace_df.empty else [],
        "pos_flow": _to_json_safe(pos_flow_df[p_cols], max_rows=2000) if not pos_flow_df.empty else [],
        "overtakes": _to_json_safe(overtake_df, max_rows=30) if not overtake_df.empty else [],
        "notes": _to_json_safe(notes),
    }


@router.get("/race-flow/{year}/{round_num}")
def endpoint_race_flow(year: int, round_num: int, session_type: str = "R"):
    try:
        return _cached_race_flow(year, round_num, session_type)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# 3. Tyre Stint Strategy, Fuel-Corrected OLS Degradation & Pace Distribution
# ---------------------------------------------------------------------------
@lru_cache(maxsize=64)
def _cached_tyres(year: int, round_num: int, session_type: str) -> dict:
    from src.analytics.tyre_analytics import analyze_tyres_and_pace_distributions

    stints_df, pace_dist_df, comp_sum_df, clean_laps_df, notes = analyze_tyres_and_pace_distributions(
        year, round_num, session_type
    )
    cl_cols = [c for c in ["Driver", "LapNumber", "Stint", "Compound", "TyreLife", "LapTime_Sec", "FuelCorrectedLap_Sec"] if c in clean_laps_df.columns]
    return {
        "stints": _to_json_safe(stints_df, max_rows=250) if not stints_df.empty else [],
        "pace_dist": _to_json_safe(pace_dist_df, max_rows=30) if not pace_dist_df.empty else [],
        "compound_summary": _to_json_safe(comp_sum_df, max_rows=15) if not comp_sum_df.empty else [],
        "clean_laps": _to_json_safe(clean_laps_df[cl_cols], max_rows=2000) if not clean_laps_df.empty else [],
        "notes": _to_json_safe(notes),
    }


@router.get("/tyres/{year}/{round_num}")
def endpoint_tyres(year: int, round_num: int, session_type: str = "R"):
    try:
        return _cached_tyres(year, round_num, session_type)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# 4. Weather Crossovers ('Was It Worth It?'), SC/VSC & DNF/DNS Reliability
# ---------------------------------------------------------------------------
@lru_cache(maxsize=64)
def _cached_weather_incidents(year: int, round_num: int, session_type: str) -> dict:
    from src.analytics.incident_weather_analytics import analyze_weather_vsc_and_dnfs

    dnf_df, attr_df, neut_df, pit_dec_df, weather_tl_df, notes = analyze_weather_vsc_and_dnfs(
        year, round_num, session_type
    )
    return {
        "dnf": _to_json_safe(dnf_df, max_rows=30) if not dnf_df.empty else [],
        "attrition": _to_json_safe(attr_df, max_rows=15) if not attr_df.empty else [],
        "neutralizations": _to_json_safe(neut_df, max_rows=25) if not neut_df.empty else [],
        "pit_decisions": _to_json_safe(pit_dec_df, max_rows=100) if not pit_dec_df.empty else [],
        "weather_timeline": _to_json_safe(weather_tl_df, max_rows=500) if not weather_tl_df.empty else [],
        "notes": _to_json_safe(notes),
    }


@router.get("/weather-incidents/{year}/{round_num}")
def endpoint_weather_incidents(year: int, round_num: int, session_type: str = "R"):
    try:
        return _cached_weather_incidents(year, round_num, session_type)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# 5. Constructor Aero Philosophy Quadrant & Pit-Wall Strategy Audit
# ---------------------------------------------------------------------------
@lru_cache(maxsize=64)
def _cached_constructors(year: int, round_num: int, session_type: str) -> dict:
    from src.analytics.constructor_analytics import analyze_constructors_and_strategy

    ev_teams_df, strat_audit_df, season_wcc_df, notes = analyze_constructors_and_strategy(
        year, round_num, session_type
    )
    return {
        "event_teams": _to_json_safe(ev_teams_df, max_rows=20) if not ev_teams_df.empty else [],
        "strategy_audit": _to_json_safe(strat_audit_df, max_rows=20) if not strat_audit_df.empty else [],
        "season_wcc": _to_json_safe(season_wcc_df, max_rows=20) if not season_wcc_df.empty else [],
        "notes": _to_json_safe(notes),
    }


@router.get("/constructors/{year}/{round_num}")
def endpoint_constructors(year: int, round_num: int, session_type: str = "R"):
    try:
        return _cached_constructors(year, round_num, session_type)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# 6. 6-Axis Driver DNA Radar, Theoretical Optimal Sectors & Teammates
# ---------------------------------------------------------------------------
@lru_cache(maxsize=64)
def _cached_driver_dna(year: int, round_num: int, session_type: str) -> dict:
    from src.analytics.driver_dna import analyze_driver_dna_and_sectors

    dna_df, sectors_df, h2h_df, notes = analyze_driver_dna_and_sectors(
        year, round_num, session_type
    )
    return {
        "dna": _to_json_safe(dna_df, max_rows=30) if not dna_df.empty else [],
        "sectors": _to_json_safe(sectors_df, max_rows=30) if not sectors_df.empty else [],
        "h2h": _to_json_safe(h2h_df, max_rows=30) if not h2h_df.empty else [],
        "notes": _to_json_safe(notes),
    }


@router.get("/driver-dna/{year}/{round_num}")
def endpoint_driver_dna(year: int, round_num: int, session_type: str = "R"):
    try:
        return _cached_driver_dna(year, round_num, session_type)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))