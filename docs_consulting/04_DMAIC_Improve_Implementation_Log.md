# Phase 4: DMAIC Improve — Engineering & Cloud Implementation Log

## Workstream 1: Executive BI Warehouse (BigQuery + Data Studio)
- **Root Cause Addressed:** Team Principals required 6–8 clicks across technical ML tabs to review season performance; DuckDB `HUGEINT` columns required type casting for cloud warehouse compatibility.
- **Solution Implemented:** Exported type-safe Apache Parquet tables (`dim_results`, `fact_stints`, `dim_weather`) from DuckDB into Google BigQuery Sandbox (`$0` cost) and connected an interactive Data Studio Executive Scorecard.
- **Verification Result:** `vw_executive_scorecard` deployed in BigQuery (`asia-south1`) with `IS_NAN()` null-handling; live Data Studio Executive Scorecard verified across 2024–2026 seasons (`7,535` total points, `204` podiums, `-0.08` sec/lap mean tyre degradation slope).
- **KPI Impact:** Executive filtering cycle reduced from **8 clicks to 1 click** (Defect Rate: **0%**).
Workstream 2's FastAPI + Gemini gemini-3.8-flash Race Engineer is live and verified with Code 200   # Phase 4: DMAIC Improve — Engineering & Cloud Implementation Log

## Workstream 1: Executive BI Warehouse (BigQuery + Data Studio)
- **Root Cause Addressed:** Team Principals required 6–8 clicks across technical ML tabs to review season performance; DuckDB `HUGEINT` columns required type casting for cloud warehouse compatibility.
- **Solution Implemented:** Exported type-safe Apache Parquet tables (`dim_results`, `fact_stints`, `dim_weather`) from DuckDB into Google BigQuery Sandbox (`$0` cost) and connected an interactive Data Studio Executive Scorecard.
- **Verification Result:** `vw_executive_scorecard` deployed in BigQuery (`asia-south1`) with `IS_NAN()` null-handling; live Data Studio Executive Scorecard verified across 2024–2026 seasons (`7,535` total points, `204` podiums, `-0.08` sec/lap mean tyre degradation slope).
- **KPI Impact:** Executive filtering cycle reduced from **8 clicks to 1 click** (Defect Rate: **0%**).

---

## Workstream 2: Decoupled FastAPI Backend & Grounded AI Race Engineer
- **Root Cause Addressed:** Monolithic Streamlit full-script reruns caused ~3,200 ms UI latency (~90.7% of total delay), and manual multi-tab pit-stop synthesis took ~48–50 seconds under race conditions.
- **Solution Implemented:**
  - Built a modular REST API layer (`src/api/main.py`, `src/api/analytics_router.py`, `src/api/ml_ai_router.py`) with column-pruned DuckDB queries wrapping all existing telemetry, tyre degradation, driver DNA, race flow, and 10-model ML modules (`src/analytics/` and `src/models/`).
  - Integrated the `google-genai` SDK for single-pass DuckDB-grounded strategy briefings, paired with a resilient **DuckDB Grounded Fallback** engine so Tab 4 still computes optimal pit-stop calls even when external Gemini API quota is depleted.
- **Verification Result:** All FastAPI analytics, ML, and AI Race Engineer endpoints verified with HTTP `200 OK`; live warm API latency measured at **169 ms** in Cloud Run and DuckDB grounded strategy retrieval verified at **329.4 ms**.
- **KPI Impact:** UI/API response latency reduced by **94.7%** (`~3,200 ms` → `169 ms`, beating the `< 250 ms` CTQ target); pit-wall strategy synthesis reduced from ~48 seconds across 3 tabs to a single prompt.

---

## Workstream 3: Pit-Wall Console v5.0, Data Expansion & Cloud Run Production Deployment
- **Root Cause Addressed:** Streamlit Community Cloud sleep timeouts, memory crashes under XGBoost/DuckDB workloads, and lack of a production-grade CI/CD deployment pipeline.
- **Solution Implemented:**
  - Built the dark-mode, zero-reload **Apex Racing F1 Pit-Wall Console v5.0** (`frontend/index.html` + modular `tab_telemetry.js`, `tab_analytics.js`, `tab_ml_suite.js`, `tab_ai_engineer.js`) using Tailwind CSS, Plotly.js, and Marked.js with an embedded real-time **Six Sigma API Latency Meter**.
  - Appended **2026 Round 21: Las Vegas Grand Prix** (`src/pipeline/append_las_vegas_2026.py`), synchronized GPS circuit maps (`src/pipeline/sync_all_circuit_maps.py`), and precomputed lakehouse assets (`src/precompute_lakehouse.py`).
  - Containerized the platform via `Dockerfile` (with Linux `libgomp1` and `requirements-cloud.txt`) and connected GitHub (`main`) to **Google Cloud Build** for automated CI/CD.
  - Deployed `f1-telemetry-ai-platform` to **Google Cloud Run (`asia-south1`, Mumbai)** with **2 vCPU**, **2 GiB RAM**, Request-based billing, `min-instances = 0` / `max-instances = 2` scaling guardrails, a **₹100/month billing alert** (`Apex-F1-100-INR-Limit`), and custom HTTPS URL **`https://apex-f1-pitwall-vatsala.cloud.run`**.
- **Verification Result:** Verified public HTTPS accessibility across desktop Incognito and mobile devices; Tabs 1–3 verified live on 2026 Monaco (R6) and Las Vegas (R21) sessions, and Tab 4 verified with `GEMINI_API_KEY` injected via Cloud Run environment variables and active DuckDB Grounded Fallback.
- **KPI Impact:** Eliminated community-hosting sleep timeouts and OOM crashes while maintaining **₹0 idle compute cost** within the ₹100–₹200/month budget envelope.