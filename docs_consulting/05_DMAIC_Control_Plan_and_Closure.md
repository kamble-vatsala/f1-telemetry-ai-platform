# Phase 5: DMAIC Control — Process Control Plan & Project Closure

* **Project Title:** Apex Racing F1 — Next-Gen Telemetry, Executive BI & Agentic Strategy Platform
* **Methodology:** Lean Six Sigma DMAIC + Agile Full-Stack Data Science Delivery
* **Project Lead:** Lead Data Scientist & Six Sigma Black Belt Consultant
* **Live Production Console:** [https://apex-f1-pitwall-vatsala.cloud.run](https://apex-f1-pitwall-vatsala.cloud.run)
* **Cloud Infrastructure:** Google Cloud Run (`asia-south1`, Mumbai) | Google BigQuery (`asia-south1`) | Looker Studio | Cloud Build CI/CD
* **Project Status:** **CLOSED — IN STATISTICAL & OPERATIONAL CONTROL**

---

## 1. Before vs. After KPI Scorecard (DMAIC Impact Verification)

All four Critical-to-Quality (CTQ) metrics defined in `01_DMAIC_Project_Charter.md` and benchmarked in `02_DMAIC_Measure_Baseline_Report.md` were re-measured in production on Google Cloud Run (`f1-telemetry-ai-platform`, Revision `00004+`):

| KPI Metric (CTQ) | Phase 2 Baseline (Streamlit v1.0) | Target (USL / Defect Limit) | Verified Production Result (v5.0 on GCP) | Improvement | Defect Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. UI / API Response Latency** | ~3,200 ms (full script rerun) | `< 250 ms` (Defect: `> 500 ms`) | **169 ms** warm API latency (live Cloud Run meter) | **94.7% faster** | **0% Defect (In Control)** |
| **2. Pit Strategy Decision Cycle** | ~48–50 sec (manual across 3 tabs) | `< 5 sec` (Defect: `> 10 sec`) | **329.4 ms** DuckDB grounded synthesis (1 prompt) | **99.3% faster** | **0% Defect (In Control)** |
| **3. Executive Report Filtering** | 6–8 clicks + manual dropdowns | `1 click` (Defect: `> 2 clicks`) | **1 click** cross-filtered BigQuery + Looker Studio canvas (`7,535` pts, `204` podiums) | **87.5% fewer clicks** | **0% Defect (In Control)** |
| **4. Monthly Cloud Operating Cost** | $0 (with sleep timeouts & OOM crashes) | `<= ₹100–₹200/mo` (Always Free optimized) | **₹0 idle compute** (`min-instances=0`, Request-based); est. **~₹10–₹30/mo** container storage | **Production-grade SLA at near-₹0 cost** | **In Control (Budget Guardrail Active)** |

---

## 2. Ongoing Process Control Plan

To prevent performance regression, cost drift, or service downtime after project handover, the following control mechanisms are enforced in production:

| Process / Control Item | Specification / Target Limit | Measurement & Monitoring Tool | Frequency | Reaction Plan (Out-of-Control SOP) |
| :--- | :--- | :--- | :--- | :--- |
| **API Latency (`Y1`)** | Warm response `< 250 ms` | Built-in **Six Sigma API Latency Meter** in top navbar + Cloud Run **Observability** tab | Continuous (every request) | Inspect column pruning in `src/api/analytics_router.py`; verify no `SELECT *` queries were introduced. |
| **Cloud Spend & Billing (`Y4`)** | `<= ₹100/month` soft alert ceiling | GCP Billing Budget **`Apex-F1-100-INR-Limit`** (email alerts at 50%, 90%, 100%) | Automated real-time tracking | Verify `min-instances = 0` and `max-instances = 2` in Cloud Run; prune old Artifact Registry images (see SOP-2). |
| **Container Memory & Stability** | Peak RAM `< 1.6 GiB` (Limit: `2 GiB`, `2 vCPU`) | Cloud Run **Metrics** (Container memory utilization & 5xx error rate) | Weekly / Post-deploy | Review XGBoost/DuckDB memory spikes in Cloud Run Logs; confirm `requirements-cloud.txt` remains lean. |
| **AI Strategist Availability (Tab 4)** | Valid strategy brief on 100% of queries | Live Tab 4 output badge (`Gemini` vs. `DuckDB Grounded Fallback`) | Per test / Monthly | If `402 RESOURCE_EXHAUSTED` appears, DuckDB Grounded Fallback maintains 100% uptime; rotate `GEMINI_API_KEY` via SOP-3. |
| **CI/CD Deployment Integrity** | Green Cloud Build status on `main` | GitHub → **Cloud Build Trigger** (`^main$`, `/Dockerfile`) | On every `git push` | Roll back traffic to the previous green revision in Cloud Run **Revision History** (zero downtime) and inspect build logs. |
| **Lakehouse Data Currency** | Includes latest completed 2026 Grand Prix | DuckDB table row counts (`clean_results`, `clean_laps`, `session_weather`) | Post-race weekend | Execute SOP-1 to append new race data and refresh BigQuery executive views. |

---

## 3. Standard Operating Procedures (SOPs)

### SOP-1: Post-Race Data Ingestion & Warehouse Sync
When a new Grand Prix weekend finishes:
1. Run the incremental ingestion script (following the `src/pipeline/append_las_vegas_2026.py` pattern) to pull FastF1 session results, laps, stints, and weather into `data/f1_lakehouse.duckdb`.
2. Run `python src/pipeline/sync_all_circuit_maps.py` (if a new circuit layout is added) and `python src/precompute_lakehouse.py` to refresh cached telemetry/ML features.
3. Run `python src/pipeline/export_for_bigquery.py` and upload refreshed Parquet files (`dim_results`, `fact_stints`, `dim_weather`) to BigQuery (`f1_executive_warehouse`) so `vw_executive_scorecard` and Looker Studio update automatically.
4. Commit and push to `main` to trigger an automated Cloud Build rollout to `https://apex-f1-pitwall-vatsala.cloud.run`.

### SOP-2: Cost & Artifact Registry Hygiene
1. **Batch Documentation Commits:** When updating only Markdown files (`docs_consulting/`, `README.md`), batch changes into a single commit or include `[skip ci]` in the commit message so Cloud Build does not generate unnecessary container images.
2. **Prune Old Container Images:** Periodically open **Artifact Registry → `cloud-run-source-deploy` → `f1-telemetry-ai-platform`** in Google Cloud Console and delete superseded image digests (keeping only the latest active revision) to keep storage near the `0.5 GiB` Always Free allowance.

### SOP-3: Rotating `GEMINI_API_KEY` Without Rebuilding
1. Generate or copy an active key from [Google AI Studio](https://ai.studio/projects).
2. In Google Cloud Run, open **`f1-telemetry-ai-platform` → Containers → Variables & Secrets**.
3. Update the value of **`GEMINI_API_KEY`** (exact uppercase key name), click **Done**, and click **Redeploy**. Cloud Run injects the new key into the existing image in ~30 seconds with no Docker rebuild.

---

## 4. DMAIC Lifecycle Deliverables & Project Closure

| DMAIC Phase | Governance Artifact | Key Milestone Delivered |
| :--- | :--- | :--- |
| **1. Define** | `docs_consulting/01_DMAIC_Project_Charter.md` | VOC → CTQ translation, SIPOC, problem statement, and 4 measurable KPIs. |
| **2. Measure** | `docs_consulting/02_DMAIC_Measure_Baseline_Report.md` | Automated MSA benchmark across 8 DuckDB tables (`293,634` rows, `178.21 ms` mean SQL vs. `~3,200 ms` Streamlit UI). |
| **3. Analyze** | `docs_consulting/03_DMAIC_Analyze_Root_Cause.md` | Pareto proof that monolithic UI reruns caused ~90.7% of latency; full codebase & schema inventory. |
| **4. Improve** | `docs_consulting/04_DMAIC_Improve_Implementation_Log.md` | BigQuery + Looker Studio Executive BI, FastAPI + 10-Model ML Suite, Grounded AI Strategist, Pit-Wall Console v5.0, and live Cloud Run CI/CD. |
| **5. Control** | `docs_consulting/05_DMAIC_Control_Plan_and_Closure.md` | Verified before/after KPI scorecard, production control plan, cost guardrails, and maintenance SOPs. |

**Engagement Sign-Off:** All defined business defects have been eliminated, production stability and public HTTPS access are verified across desktop and mobile, and automated cost/operational controls are active. The engagement is formally closed.