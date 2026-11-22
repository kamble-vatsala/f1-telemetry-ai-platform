# Phase 3: DMAIC Analyze — Root Cause Analysis & Solution Mapping

## 1. Pareto Analysis of UI Latency ($Y_1$)

Based on Phase 2 empirical measurements (`178.21 ms` mean DuckDB latency vs. `~3,000 ms` Streamlit rerun latency):

1. **Monolithic Script Re-execution (~90.7% of delay):** Streamlit re-runs the entire Python script and redraws all UI components on every widget interaction.
2. **Unpruned `SELECT *` Queries (~5.9% of delay):** Wide tables such as `clean_results` (`404.19 ms` for 6,996 rows) transfer unused columns on every query.
3. **Hosting & Cold-Start Overhead (~3.4% warm / up to 30s cold):** Community hosting suspends inactive containers.

---

## 2. Value-Stream Solution Mapping (Linking Root Causes to Phase 4 Upgrades)

| Vital Few Root Cause | Target Architecture Fix (Phase 4 Improve) | Expected KPI Impact |
| :--- | :--- | :--- |
| Full-script UI reruns on every slider move | Decouple backend via **Python FastAPI** + interactive **React** frontend on **Google Cloud Run** | Reduces UI update latency from `~3,000 ms` to **under 250 ms** |
| Slow `SELECT *` queries on wide tables | Write column-pruned SQL views (`SELECT driver, lap_time, ...`) inside FastAPI endpoints | Reduces SQL fetch time from `404 ms` to **under 30 ms** |
| Multi-tab manual pit-stop decision workflow | Deploy **Gemini API Function Calling ("AI Race Engineer")** connected to ML & simulator functions | Reduces strategy synthesis from `~48 sec (7 clicks)` to **under 5 sec (1 prompt)** |
| Lack of executive season-level reporting | Sync aggregated DuckDB tables to **Google BigQuery** + interactive **Data Studio** dashboard | Reduces executive filtering from `8 clicks` to **1 click** at **$0 cost** |

---

## 3. Codebase & DuckDB Schema Inventory (Output from `analyze_architecture.py`)

(venv) PS D:\f1-platform-fresh> python docs_consulting/analyze_architecture.py
======================================================================
SIX SIGMA DMAIC - PHASE 3: CODE & SCHEMA ARCHITECTURE AUDIT
======================================================================

[1] EXISTING PYTHON MODULES IN REPOSITORY:
----------------------------------------------------------------------
  - app.py                                        (1477 lines)
  - src\analytics\constructor_analytics.py        (182 lines)
  - src\analytics\driver_dna.py                   (149 lines)
  - src\analytics\incident_weather_analytics.py   (238 lines)
  - src\analytics\query_data.py                   (0 lines)
  - src\analytics\race_analytics.py               (115 lines)
  - src\analytics\telemetry.py                    (0 lines)
  - src\analytics\telemetry_engine.py             (451 lines)
  - src\analytics\tyre_analytics.py               (133 lines)
  - src\analytics\tyres.py                        (0 lines)
  - src\models\ml_supervised_suite.py             (363 lines)
  - src\models\ml_unsupervised_bayesian_sim.py    (372 lines)
  - src\models\predict_race.py                    (0 lines)
  - src\models\predictor.py                       (68 lines)
  - src\pipeline\audit_data.py                    (0 lines)
  - src\pipeline\build_clean_database.py          (0 lines)
  - src\pipeline\bulk_download.py                 (0 lines)
  - src\pipeline\fetch_data.py                    (0 lines)
  - src\pipeline\ingest_all_seasons.py            (0 lines)
  - src\pipeline\sync_telemetry_to_duckdb.py      (142 lines)

[2] SAVED MODEL / CONFIG ARTIFACTS:
----------------------------------------------------------------------
  - None found on disk (models are likely trained dynamically in Python).

[3] DUCKDB TABLE SCHEMAS (FOR BIGQUERY & FASTAPI MAPPING):
----------------------------------------------------------------------

* circuit_corners_cache [8 columns]:
  Year (INTEGER), RoundNumber (INTEGER), Number (INTEGER), Letter (VARCHAR), Distance (DOUBLE), X (DOUBLE), Y (DOUBLE), CornerLabel (VARCHAR)

* clean_laps [40 columns]:
  Year (BIGINT), RoundNumber (BIGINT), EventName (VARCHAR), Country (VARCHAR), Location (VARCHAR), EventFormat (VARCHAR), SessionType (VARCHAR), Driver (VARCHAR), DriverNumber (VARCHAR), TeamFranchise (VARCHAR), OfficialTeamName (VARCHAR), LapNumber (INTEGER), Stint (INTEGER), Compound (VARCHAR), TyreLife (INTEGER), FreshTyre (BOOLEAN), Position (INTEGER), TrackStatus (VARCHAR), IsAccurate (BOOLEAN), IsPitLap (BOOLEAN), LapTime_Sec (DOUBLE), Sector1_Sec (DOUBLE), Sector2_Sec (DOUBLE), Sector3_Sec (DOUBLE), SpeedI1 (DOUBLE), SpeedI2 (DOUBLE), SpeedFL (DOUBLE), SpeedST (DOUBLE), SessionBestLap_Sec (DOUBLE), SessionMedianLap_Sec (DOUBLE), TotalSessionLaps (INTEGER), DeltaToBest_Sec (DOUBLE), DeltaToBest_Pct (DOUBLE), IsGreenFlag (BOOLEAN), IsRepresentativePace (BOOLEAN), FuelCorrectedLap_Sec (DOUBLE), AvgTrackTemp (DOUBLE), AvgAirTemp (DOUBLE), AvgHumidity (DOUBLE), IsWetSession (BOOLEAN)

* clean_rcm [16 columns]:
  Year (BIGINT), RoundNumber (BIGINT), EventName (VARCHAR), SessionType (VARCHAR), Time (TIMESTAMP_NS), Category (VARCHAR), Message (VARCHAR), Status (VARCHAR), Flag (VARCHAR), Scope (VARCHAR), Sector (DOUBLE), RacingNumber (VARCHAR), Lap (INTEGER), IncidentType (VARCHAR), PrevIncidentTime (TIMESTAMP_NS), IsNewDeployment (INTEGER)

* clean_results [30 columns]:
  Year (BIGINT), RoundNumber (BIGINT), EventName (VARCHAR), Country (VARCHAR), Location (VARCHAR), EventFormat (VARCHAR), SessionType (VARCHAR), Driver (VARCHAR), DriverNumber (VARCHAR), FullName (VARCHAR), OfficialTeamName (VARCHAR), TeamFranchise (VARCHAR), TeamColor (VARCHAR), FinishPosition (INTEGER), GridPosition (INTEGER), PositionsGained (INTEGER), Points (DOUBLE), Status (VARCHAR), Time_Sec (DOUBLE), Q1_Sec (DOUBLE), Q2_Sec (DOUBLE), Q3_Sec (DOUBLE), AvgTrackTemp (DOUBLE), AvgAirTemp (DOUBLE), IsWetSession (BOOLEAN), SC_Count (HUGEINT), VSC_Count (HUGEINT), RedFlag_Count (HUGEINT), Penalty_Msgs (BIGINT), DeletedLaps_Count (BIGINT)

* clean_stints [18 columns]:
  Year (BIGINT), RoundNumber (BIGINT), EventName (VARCHAR), SessionType (VARCHAR), Driver (VARCHAR), OfficialTeamName (VARCHAR), TeamFranchise (VARCHAR), Stint (INTEGER), Compound (VARCHAR), StartLap (INTEGER), EndLap (INTEGER), StintSpanLaps (INTEGER), TotalStintLaps (BIGINT), CleanStintLaps (BIGINT), AvgRawPace_Sec (DOUBLE), MedianRawPace_Sec (DOUBLE), AvgFuelCorrectedPace_Sec (DOUBLE), TyreDegSlope_SecPerLap (DOUBLE)

* session_incidents [8 columns]:
  Year (BIGINT), RoundNumber (BIGINT), SessionType (VARCHAR), SC_Count (HUGEINT), VSC_Count (HUGEINT), RedFlag_Count (HUGEINT), Penalty_Msgs (BIGINT), DeletedLaps_Count (BIGINT)

* session_weather [9 columns]:
  Year (BIGINT), RoundNumber (BIGINT), SessionType (VARCHAR), AvgTrackTemp (DOUBLE), MaxTrackTemp (DOUBLE), AvgAirTemp (DOUBLE), AvgHumidity (DOUBLE), AvgWindSpeed (DOUBLE), IsWetSession (BOOLEAN)

* telemetry_cache [17 columns]:
  Year (INTEGER), RoundNumber (INTEGER), SessionType (VARCHAR), Driver (VARCHAR), LapNumber (INTEGER), LapTime_Sec (DOUBLE), Compound (VARCHAR), TyreLife (INTEGER), Distance (DOUBLE), Speed (DOUBLE), Throttle (DOUBLE), Brake_Pct (DOUBLE), nGear (INTEGER), DRS_Active (INTEGER), Time_Sec (DOUBLE), X (DOUBLE), Y (DOUBLE)

======================================================================
(venv) PS D:\f1-platform-fresh> 