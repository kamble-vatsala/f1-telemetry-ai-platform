# Phase 2: DMAIC Measure — Data Collection Plan & Baseline Capability

## 1. Data Collection Plan & Measurement System Analysis (MSA)

| Metric Name | Variable Type | Operational Definition | Measurement Method (MSA) | Sample Size |
| :--- | :--- | :--- | :--- | :--- |
| **DuckDB Query Latency ($X_1$)** | Input ($X$) — Continuous | Time required to query up to 5,000 rows per table from `data\f1_lakehouse.duckdb`. | Automated Python `time.perf_counter()` script (`measure_baseline.py`). | 10 trials per table |
| **UI Interaction Latency ($Y_1$)** | Output ($Y$) — Continuous | Time from clicking a Streamlit filter/slider until the top-right `Running...` status completes. | Live browser test on `f1-telemetry-ai-platform-kamblevatsala.streamlit.app`. | 3 trials per widget |
| **Strategy Decision Cycle ($Y_2$)** | Output ($Y$) — Continuous | Total time & clicks required to evaluate Tyre Deg + Pit Simulator + ML Predictor. | End-to-end user workflow walkthrough. | 3 scenario runs |
| **Executive Filter Clicks ($Y_3$)** | Output ($Y$) — Discrete | Number of clicks required for a Team Principal to isolate season KPIs for 1 driver. | Direct UI click count audit. | 1 audit |

---

## 2. Database Asset Inventory (`data\f1_lakehouse.duckdb` — 17.26 MB)

* **Total Records Across Lakehouse:** 293,634 rows across 8 tables
* **Overall Mean DuckDB Query Latency ($X_1$):** **178.21 ms**

| Table Name | Row Count | Mean Query Time ($X_1$) | Std Dev ($\pm\text{ms}$) | Architectural Role in v2.0 Upgrade |
| :--- | :--- | :--- | :--- | :--- |
| `circuit_corners_cache` | 77 | 30.72 ms | $\pm 40.96\text{ ms}$ | FastAPI Track Map Endpoint (Low-latency JSON) |
| `clean_laps` | 188,917 | 362.39 ms | $\pm 125.06\text{ ms}$ | BigQuery Executive BI + FastAPI Pace Analytics |
| `clean_rcm` | 19,887 | 173.25 ms | $\pm 112.99\text{ ms}$ | FastAPI Race Control / Safety Car Context for Gemini AI |
| `clean_results` | 6,996 | 404.19 ms | $\pm 158.97\text{ ms}$ | BigQuery Executive BI (Driver/Team Season Scorecard) |
| `clean_stints` | 26,577 | 172.18 ms | $\pm 66.95\text{ ms}$ | BigQuery BI + FastAPI Tyre Degradation Engine |
| `session_incidents` | 340 | 47.61 ms | $\pm 26.52\text{ ms}$ | BigQuery Executive BI (Reliability & DNF Summary) |
| `session_weather` | 340 | 63.65 ms | $\pm 83.51\text{ ms}$ | BigQuery BI + FastAPI Track Temp Feature Input |
| `telemetry_cache` | 50,500 | 171.66 ms | $\pm 90.72\text{ ms}$ | FastAPI High-Frequency 20Hz Telemetry Traces |

---

## 3. Baseline Process Performance Summary ("As-Is" State)

| Process Step / Action | Measured Baseline | Upper Spec Limit (USL) | Six Sigma Defect Status |
| :--- | :--- | :--- | :--- |
| **DuckDB Mean SQL Fetch ($X_1$)** | **178.21 ms** | 250 ms | **PASS** (Capable — well within spec) |
| **Wide-Table `SELECT *` Fetch (`clean_results`)** | **404.19 ms** | 250 ms | **DEFECT** (Requires column pruning in FastAPI) |
| **Streamlit UI Rerun Latency ($Y_1$)** | **~3,000 ms** *(live browser rerun)* | 500 ms | **DEFECT** (6x slower than pit-wall spec) |
| **Multi-Tab Pit Strategy Check ($Y_2$)** | **~48 sec / 7 clicks** | 10 sec / 2 clicks | **DEFECT** (Requires AI Agent consolidation) |