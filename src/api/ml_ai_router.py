from functools import lru_cache
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

import duckdb
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from google import genai
from google.genai import types
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "f1_lakehouse.duckdb"

router = APIRouter(prefix="/api", tags=["10-Model AI/ML Suite & Gemini Race Engineer"])


def _to_json_safe(obj: Any, max_rows: Optional[int] = 100) -> Any:
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
# 1. Cached Individual 10-Model ML Runners (Models 2, 3 & 5 Cleanly Separated)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=48)
def _cached_models_235_bundle(year: int, round_num: int):
    from src.models.ml_supervised_suite import run_models_2_3_5_race_prediction_and_xai

    return run_models_2_3_5_race_prediction_and_xai(year, round_num)


@lru_cache(maxsize=64)
def _cached_run_individual_model(
    model_key: str,
    year: int,
    round_num: int,
    session_type: str,
    circuit_name: str,
    rain_prob: float,
    sc_prob: float,
) -> dict:
    from src.models.ml_supervised_suite import (
        run_model1_practice_to_quali_regression,
        run_model4_neural_net_tyre_deg,
    )
    from src.models.ml_unsupervised_bayesian_sim import (
        run_model6_pca_kmeans_driver_clustering,
        run_model7_isolation_forest_anomalies,
        run_model8_timeseries_track_evolution,
        run_model9_bayesian_driver_car_ratings,
        run_model10_monte_carlo_upcoming_race_simulator,
    )

    if model_key == "model1_quali":
        pred_df, metrics_df, notes = run_model1_practice_to_quali_regression(year, round_num)
        return {
            "model_key": model_key,
            "context": _to_json_safe(notes),
            "primary": _to_json_safe(pred_df, max_rows=25),
            "secondary": _to_json_safe(metrics_df, max_rows=10),
        }

    if model_key in ("model2_ranker", "model3_probabilities", "model5_xai"):
        race_df, cal_df, xai_df, notes = _cached_models_235_bundle(year, round_num)

        if model_key == "model2_ranker":
            cols = [
                c
                for c in [
                    "PredictedPosition",
                    "FinishPosition",
                    "Driver",
                    "OfficialTeamName",
                    "GridPosition",
                    "ExpectedFinishScore",
                    "DriverRollingFinish_5R",
                    "TeamRollingFinish_5R",
                    "FP_LongRunDelta_Pct",
                    "Status",
                ]
                if c in race_df.columns
            ]
            out_df = race_df[cols].sort_values("PredictedPosition") if not race_df.empty else race_df
            ctx = {
                "f1_context": "Forecasts Sunday's full grid finishing order prior to lights out by integrating starting grid slot, Friday long-run heavy-fuel pace, Sprint performance, and 5-race rolling driver/team form.",
                "ds_methodology": "Walk-Forward Time-Series XGBoost Regressor (trained strictly on past races before the target round to eliminate temporal data leakage, using 85% row/feature subsampling).",
                "key_insight": notes.get("key_insight", ""),
            }
            return {"model_key": model_key, "context": ctx, "primary": _to_json_safe(out_df, 25), "secondary": []}

        if model_key == "model3_probabilities":
            cols = [
                c
                for c in [
                    "Driver",
                    "OfficialTeamName",
                    "Podium_Prob_Pct",
                    "Points_Prob_Pct",
                    "GridPosition",
                    "PredictedPosition",
                    "FinishPosition",
                ]
                if c in race_df.columns
            ]
            out_df = (
                race_df[cols].sort_values("Podium_Prob_Pct", ascending=False)
                if not race_df.empty
                else race_df
            )
            ctx = {
                "f1_context": "Translates raw pace rankings into exact percentage probabilities of scoring a Top-3 Podium or Top-10 Points finish to guide aggressive vs. conservative pit-wall risk tolerance.",
                "ds_methodology": "Platt-Calibrated Gradient Boosting Classifier (CalibratedClassifierCV, sigmoid scaling) evaluated via ROC-AUC discrimination and Brier Score calibration loss.",
                "key_insight": notes.get("key_insight", ""),
            }
            return {
                "model_key": model_key,
                "context": ctx,
                "primary": _to_json_safe(out_df, 25),
                "secondary": _to_json_safe(cal_df, 10),
            }

        # model5_xai
        local_cols = [
            c
            for c in [
                "PredictedPosition",
                "Driver",
                "OfficialTeamName",
                "GridPosition",
                "Podium_Prob_Pct",
                "XAI_KeyDrivers",
            ]
            if c in race_df.columns
        ]
        local_df = race_df[local_cols].sort_values("PredictedPosition") if not race_df.empty else race_df
        ctx = {
            "f1_context": "Opens the machine learning black box so race engineers see the exact performance factors pushing each driver up or down the forecast.",
            "ds_methodology": "Global XGBoost Gain Importance paired with Model-Agnostic Permutation Importance and per-driver local feature attributions.",
            "key_insight": notes.get("key_insight", ""),
        }
        return {
            "model_key": model_key,
            "context": ctx,
            "primary": _to_json_safe(xai_df, 15),
            "secondary": _to_json_safe(local_df, 25),
        }

    if model_key == "model4_nn_tyre":
        nn_df, notes = run_model4_neural_net_tyre_deg(year)
        return {"model_key": model_key, "context": _to_json_safe(notes), "primary": _to_json_safe(nn_df, 60), "secondary": []}

    if model_key == "model6_clustering":
        cl_df, notes = run_model6_pca_kmeans_driver_clustering(year)
        return {"model_key": model_key, "context": _to_json_safe(notes), "primary": _to_json_safe(cl_df, 30), "secondary": []}

    if model_key == "model7_anomalies":
        anom_df, notes = run_model7_isolation_forest_anomalies(year, round_num, session_type)
        return {"model_key": model_key, "context": _to_json_safe(notes), "primary": _to_json_safe(anom_df, 30), "secondary": []}

    if model_key == "model8_evolution":
        ev_df, notes = run_model8_timeseries_track_evolution(year, round_num)
        return {"model_key": model_key, "context": _to_json_safe(notes), "primary": _to_json_safe(ev_df, 15), "secondary": []}

    if model_key == "model9_bayesian":
        bay_df, notes = run_model9_bayesian_driver_car_ratings(year)
        return {"model_key": model_key, "context": _to_json_safe(notes), "primary": _to_json_safe(bay_df, 30), "secondary": []}

    if model_key == "model10_montecarlo":
        mc_df, notes = run_model10_monte_carlo_upcoming_race_simulator(
            year, circuit_name, 1000, rain_prob, sc_prob
        )
        return {"model_key": model_key, "context": _to_json_safe(notes), "primary": _to_json_safe(mc_df, 25), "secondary": []}

    raise ValueError(f"Unsupported model_key: {model_key}")


class MLRunRequest(BaseModel):
    model_key: str
    year: int = 2025
    round_num: int = 1
    session_type: str = "R"
    circuit_name: str = "Australian Grand Prix"
    rain_prob: float = 0.2
    sc_prob: float = 0.4


@router.post("/ml/run")
def endpoint_run_ml(payload: MLRunRequest):
    try:
        return _cached_run_individual_model(
            payload.model_key,
            payload.year,
            payload.round_num,
            payload.session_type,
            payload.circuit_name,
            round(payload.rain_prob, 2),
            round(payload.sc_prob, 2),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# 2. 2024–2026 Multi-Factor Grounding Engine for AI Race Engineer (< 25 ms)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=2)
def _cached_cross_season_regimes() -> list:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        df = con.execute(
            """
            SELECT
                Year,
                CASE
                    WHEN IsWetSession THEN 'Wet / Rainy'
                    WHEN AvgTrackTemp >= 38 THEN 'Hot Track (>=38C)'
                    WHEN AvgTrackTemp <= 26 THEN 'Cool Track (<=26C)'
                    ELSE 'Moderate (27-37C)'
                END AS WeatherRegime,
                OfficialTeamName AS Team,
                COUNT(DISTINCT RoundNumber) AS Races,
                ROUND(AVG(SpeedST), 1) AS AvgEngineSpeedST_kmh,
                ROUND(AVG(DeltaToBest_Pct), 3) AS AvgPaceDelta_Pct
            FROM clean_laps
            WHERE SessionType = 'R' AND IsRepresentativePace = TRUE
              AND SpeedST IS NOT NULL AND NOT isnan(SpeedST)
            GROUP BY Year, WeatherRegime, OfficialTeamName
            HAVING COUNT(*) >= 30
            ORDER BY Year DESC, AvgPaceDelta_Pct ASC
            LIMIT 18
            """
        ).df()
        return _to_json_safe(df, 18)
    finally:
        con.close()

def _build_multi_factor_evidence(question: str, default_year: int, default_round: int) -> dict:
    """Queries 2024–2026 DuckDB tables (weather, engine traps, sectors, corners, tyres, pit sim) in < 25 ms."""
    from src.api.analytics_router import _cached_pit_simulator, _cached_track_profile

    q_low = question.lower()
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        # 1. Detect explicit years mentioned in question (supports 2024, 2025, 2026, or multi-year)
        found_years = [int(y) for y in re.findall(r"\b(2024|2025|2026)\b", q_low)]
        multi_year = len(set(found_years)) > 1 or any(
            k in q_low for k in ("all years", "2024 to 2026", "2024-2026", "multi-year", "across seasons", "hot temp", "rainy")
        )
        target_year = found_years[0] if found_years else default_year

        # 2. Detect explicit circuit/track name from DuckDB catalog
        tracks_df = con.execute(
            """
            SELECT DISTINCT Year, RoundNumber, EventName, Location, Country
            FROM clean_results WHERE SessionType = 'R'
            ORDER BY Year DESC, RoundNumber ASC
            """
        ).df()

        target_round = default_round
        for _, t in tracks_df[tracks_df["Year"] == target_year].iterrows():
            ev = str(t["EventName"]).lower().replace("grand prix", "").strip()
            loc = str(t["Location"]).lower().strip()
            ctry = str(t["Country"]).lower().strip()
            if (ev and ev in q_low) or (loc and loc in q_low) or (ctry and ctry in q_low):
                target_round = int(t["RoundNumber"])
                break

        # 3. Detect specific driver code or surname mentioned
        drv_df = con.execute("SELECT DISTINCT Driver, FullName FROM clean_results").df()
        mentioned_drivers = []
        for _, dr in drv_df.iterrows():
            code = str(dr["Driver"]).upper()
            full = str(dr["FullName"] or "").lower()
            last = full.split()[-1] if full else ""
            if (last and len(last) >= 4 and last in q_low) or re.search(rf"\b{code}\b", question):            
                if code not in mentioned_drivers:
                    mentioned_drivers.append(code)

        # 4. Event Context & Weather
        profile = _cached_track_profile(target_year, target_round)
        event_ctx_df = con.execute(
            """
            SELECT
                ANY_VALUE(EventName) AS EventName,
                ANY_VALUE(Location) AS Location,
                ROUND(AVG(AvgTrackTemp), 1) AS AvgTrackTemp_C,
                ROUND(AVG(AvgAirTemp), 1) AS AvgAirTemp_C,
                MAX(CASE WHEN IsWetSession THEN 1 ELSE 0 END) AS IsWetSession,
                MAX(SC_Count) AS SC_Count,
                MAX(VSC_Count) AS VSC_Count
            FROM clean_results
            WHERE Year = ? AND RoundNumber = ? AND SessionType = 'R'
            """,
            [target_year, target_round],
        ).df()

        # 5. Tyre Compound Pace & OLS Deg Slopes
        comp_df = con.execute(
            """
            SELECT
                UPPER(Compound) AS Compound,
                COUNT(*) AS Stints,
                ROUND(MEDIAN(MedianRawPace_Sec), 3) AS MedianRawPace_Sec,
                ROUND(MEDIAN(AvgFuelCorrectedPace_Sec), 3) AS FuelCorrPace_Sec,
                ROUND(MEDIAN(CASE WHEN NOT isnan(TyreDegSlope_SecPerLap) THEN TyreDegSlope_SecPerLap END), 4) AS MedianDeg_SecPerLap
            FROM clean_stints
            WHERE Year = ? AND RoundNumber = ? AND SessionType = 'R'
            GROUP BY UPPER(Compound)
            ORDER BY FuelCorrPace_Sec ASC NULLS LAST
            """,
            [target_year, target_round],
        ).df()

        # 6. Engine / Power Unit Speed Traps & Sector Splits for Active Race (or Mentioned Drivers)
        drv_filter = ""
        params_eng = [target_year, target_round]
        if mentioned_drivers:
            ph = ", ".join(["?"] * len(mentioned_drivers))
            drv_filter = f"AND Driver IN ({ph})"
            params_eng += mentioned_drivers

        engine_sector_df = con.execute(
            f"""
            SELECT
                Driver,
                ANY_VALUE(OfficialTeamName) AS Team,
                ROUND(MIN(LapTime_Sec), 3) AS BestLap_Sec,
                ROUND(MIN(Sector1_Sec), 3) AS Best_S1,
                ROUND(MIN(Sector2_Sec), 3) AS Best_S2,
                ROUND(MIN(Sector3_Sec), 3) AS Best_S3,
                ROUND(MAX(SpeedST), 1) AS PeakSpeedST_kmh,
                ROUND(MAX(SpeedFL), 1) AS SpeedFL_kmh,
                ROUND(AVG(SpeedI1), 1) AS AvgSpeedI1_kmh
            FROM clean_laps
            WHERE Year = ? AND RoundNumber = ? AND SessionType = 'R'
              AND LapTime_Sec IS NOT NULL AND NOT isnan(LapTime_Sec) {drv_filter}
            GROUP BY Driver
            ORDER BY BestLap_Sec ASC NULLS LAST
            LIMIT 10
            """,
            params_eng,
        ).df()

        # 7. Cross-Season (2024–2026) Thermal & Wet Engine / Team Regime Sensitivity
        weather_engine_regimes = _cached_cross_season_regimes()
        
        # 8. Circuit Corner Benchmarks (from real FIA circuit_corners_v2 / blueprints)
        corners_summary = []
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        loc_key = profile["location"].lower().strip()
        if "circuit_corners_v2" in tables and "circuit_blueprints" in tables:
            c_df = con.execute(
                """
                SELECT c.CornerLabel, ROUND(c.Distance, 0) AS Distance_m,
                       ROUND(MIN(b.RefSpeed), 1) AS RefApexSpeed_kmh
                FROM circuit_corners_v2 c
                JOIN circuit_blueprints b
                  ON c.LocationKey = b.LocationKey
                 AND ABS(b.Distance - c.Distance) <= 45
                WHERE c.LocationKey = ?
                GROUP BY c.CornerLabel, c.Distance, c.Number
                ORDER BY c.Number
                """,
                [loc_key],
            ).df()
            corners_summary = _to_json_safe(c_df, max_rows=20)

        pit_sim = _cached_pit_simulator(
            profile["total_laps"],
            profile["base_pace_sec"],
            profile["soft_deg"],
            profile["medium_deg"],
            profile["hard_deg"],
            22.0,
        )

        return {
            "resolved_year": target_year,
            "resolved_round": target_round,
            "resolved_event": f"{target_year} R{target_round}: {profile['event_name']} ({profile['location']})",
            "multi_year_scope": multi_year,
            "mentioned_drivers": mentioned_drivers,
            "event_context": _to_json_safe(event_ctx_df, 1),
            "profile": profile,
            "compound_summary": _to_json_safe(comp_df, 10),
            "engine_sectors": _to_json_safe(engine_sector_df, 10),
            "weather_engine_regimes": weather_engine_regimes,
            "corner_benchmarks": corners_summary,
            "pit_simulation": pit_sim.get("summary", []),
        }
    finally:
        con.close()


RACE_ENGINEER_SYSTEM_PROMPT = (
    "You are the Chief Race Strategist & AI Race Engineer for Apex Racing F1 (Black Belt Six Sigma Analytics Team). "
    "You are given live, verified 2024–2026 DuckDB evidence covering Track Weather, Pirelli Tyre Degradation Slopes, "
    "Engine Speed Traps & Sector Splits, Hot vs. Wet Engine Regime Performance, Corner Apex Benchmarks, and Pit-Stop Simulations. "
    "STRICT RULES: "
    "1. Quote exact numbers directly from the provided DuckDB evidence—never invent lap times or speeds. "
    "2. Never use LaTeX math syntax or dollar signs (no $\\rightarrow$, no \\approx). Use plain Unicode arrows (→) and clean text. "
    "3. Keep your response sharp, scannable, and structured in 3 sections: "
    "### (1) Telemetry, Engine & Weather Evidence, "
    "### (2) Optimal Pit-Wall Strategy & Setup Call, and "
    "### (3) Risk Factors, Crossover Windows & Driver/Corner Notes."
)


class EngineerQuestionRequest(BaseModel):
    question: str
    year: int = 2025
    round_num: int = 1


@router.post("/race-engineer/ask")
def endpoint_ask_race_engineer(payload: EngineerQuestionRequest):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not found in .env.")

    t_db0 = time.perf_counter()
    evidence = _build_multi_factor_evidence(payload.question, payload.year, payload.round_num)
    db_ms = round((time.perf_counter() - t_db0) * 1000.0, 1)

    grounded_prompt = (
        f"ENGINEER RADIO QUERY: {payload.question}\n\n"
        f"VERIFIED DUCKDB EVIDENCE ({evidence['resolved_event']} | Multi-Year 2024-2026 Active):\n"
        f"- Event & Weather Context: {evidence['event_context']}\n"
        f"- Pirelli Compound Pace & Deg Slopes: {evidence['compound_summary']}\n"
        f"- Engine Speed Traps & Sector Splits (Top Drivers): {evidence['engine_sectors']}\n"
        f"- 2024-2026 Hot vs. Cool vs. Wet Engine & Pace Regimes: {evidence['weather_engine_regimes']}\n"
        f"- Official Corner Apex Benchmarks: {evidence['corner_benchmarks']}\n"
        f"- Pit-Stop Strategy Simulation (22s pit loss): {evidence['pit_simulation']}\n"
    )

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=45000))
    errors = []

    # Active Gemini 3.x models with low/zero thinking latency so responses finish in ~2-5 seconds
    for model_id in ("gemini-3.5-flash", "gemini-3.8-flash", "gemini-3.5-flash-lite"):
        try:
            try:
                cfg = types.GenerateContentConfig(
                    system_instruction=RACE_ENGINEER_SYSTEM_PROMPT,
                    temperature=0.2,
                    max_output_tokens=2200,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                )
            except Exception:
                cfg = types.GenerateContentConfig(
                    system_instruction=RACE_ENGINEER_SYSTEM_PROMPT,
                    temperature=0.2,
                    max_output_tokens=2200,
                )

            response = client.models.generate_content(
                model=model_id,
                contents=grounded_prompt,
                config=cfg,
            )
            return {
                "model_used": model_id,
                "db_query_ms": db_ms,
                "resolved_event": evidence["resolved_event"],
                "evidence": evidence,
                "strategy_brief": response.text,
            }
        except Exception as exc:
            errors.append(f"{model_id}: {exc}")

    # Graceful offline Six Sigma fallback so Tab 4 still delivers full grounded stats even during Gemini 503 spikes
    return {
        "model_used": "DuckDB Grounded Fallback (Gemini Busy)",
        "db_query_ms": db_ms,
        "resolved_event": evidence["resolved_event"],
        "evidence": evidence,
        "strategy_brief": (
            f"### (1) Telemetry, Engine & Weather Evidence\n"
            f"Live DuckDB data retrieved in **{db_ms} ms** for **{evidence['resolved_event']}** "
            f"(Base Pace: **{evidence['profile']['base_pace_sec']}s**, Laps: **{evidence['profile']['total_laps']}**).\n\n"
            f"### (2) Optimal Pit-Wall Strategy Call\n"
            + (
                f"Fastest simulated call: **{evidence['pit_simulation'][0]['Strategy']}** "
                f"(Total: **{evidence['pit_simulation'][0]['CumulativeTime_Sec']}s**).\n\n"
                if evidence["pit_simulation"]
                else "Review the grounded tables above for compound pace and engine traps.\n\n"
            )
            + f"### (3) API Notice\nGemini API returned a temporary error ({' | '.join(errors[:1])}). "
            f"All verified DuckDB statistical evidence tables above are live and ready for engineering review."
        ),
    }