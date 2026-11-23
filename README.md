# 🏎️ Apex Racing F1 — Six Sigma Pit-Wall Console v5.0

**An end-to-end Formula 1 Telemetry, Machine Learning, Executive BI & Grounded AI Strategy Platform — engineered and optimized using Lean Six Sigma DMAIC.**

[![Live App](https://img.shields.io/badge/Live_Console-apex--f1--pitwall--vatsala.cloud.run-e10600?style=for-the-badge&logo=googlecloud&logoColor=white)](https://apex-f1-pitwall-vatsala.cloud.run)
![Python](https://img.shields.io/badge/Python-FastAPI%20%7C%20DuckDB%20%7C%20XGBoost-3776AB?style=for-the-badge&logo=python&logoColor=white)
![GCP](https://img.shields.io/badge/Google_Cloud-Cloud_Run%20%7C%20BigQuery%20%7C%20Cloud_Build-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white)
![Six Sigma](https://img.shields.io/badge/Methodology-Lean_Six_Sigma_DMAIC-10B981?style=for-the-badge)

> **🚀 Live Production Console:** [**https://apex-f1-pitwall-vatsala.cloud.run**](https://apex-f1-pitwall-vatsala.cloud.run)  
> *(Hosted on Google Cloud Run in Mumbai `asia-south1` with scale-to-zero cost guardrails. First load after idle may take ~5–15 seconds to cold-start; subsequent API responses run in <250 ms.)*

<!-- Optional: Add a screenshot of your dark-mode Pit-Wall Console by saving an image to docs_consulting/pitwall_preview.png and uncommenting the line below -->
<!-- ![Apex Racing F1 Pit-Wall Console](docs_consulting/pitwall_preview.png) -->

---

## 📊 Lean Six Sigma DMAIC Business Impact

Originally built as a monolithic Streamlit app, the platform was re-architected through a formal **5-phase DMAIC consulting engagement** to eliminate UI bottlenecks, manual strategy synthesis delays, and hosting instability while maintaining near-zero operating cost.

| KPI Metric (CTQ) | Baseline (Streamlit v1.0) | Target (USL) | Verified Production (v5.0 on GCP) | Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **UI / API Response Latency** | ~3,200 ms *(full script rerun)* | `< 250 ms` | **169 ms** *(live Cloud Run meter)* | **94.7% faster** |
| **Pit Strategy Decision Cycle** | ~48–50 sec *(manual across 3 tabs)* | `< 5 sec` | **329.4 ms** *(single-prompt DuckDB grounded)* | **99.3% faster** |
| **Executive Report Filtering** | 6–8 clicks + manual dropdowns | `1 click` | **1 click** *(BigQuery + Looker Studio)* | **87.5% fewer clicks** |
| **Cloud Operating Cost** | $0 *(sleep timeouts & OOM crashes)* | `<= ₹100/mo` | **₹0 idle compute** (`min-instances=0`) | **Production SLA within free tier** |

---

## 🛠️ Four Production Pit-Wall Workspaces

1. **20Hz Telemetry, Corner Apex & GPS Dominance (`tab_telemetry.js`)**  
   Interactive, zero-reload Plotly.js telemetry overlays comparing driver speed traces, throttle/brake inputs, corner apex speeds, and GPS circuit dominance maps across the 2024–2026 seasons (including **2026 R21: Las Vegas GP**).
2. **Race Engineering, Tyres, Weather & Constructors (`tab_analytics.js`)**  
   Pirelli stint degradation slopes, track temperature/weather impact, constructor aerodynamic philosophy quadrants (low-drag vs. high-downforce), and pit-wall strategy audits.
3. **10-Model AI / Machine Learning Suite (`tab_ml_suite.js`)**  
   Supervised, unsupervised, Bayesian, and simulation models powered by **XGBoost** and **Scikit-Learn**: Practice-to-Qualifying regression, Race Pace & XAI predictors, Neural Net tyre degradation, PCA/K-Means driver clustering, Isolation Forest anomaly detection, time-series track evolution, Bayesian driver/car ratings, and Monte Carlo race simulation.
4. **AI Race Engineer Strategist (`tab_ai_engineer.js`)**  
   Single-pass **DuckDB Grounded Inference** pairing sub-second statistical evidence (weather, speed traps, sector splits, and tyre degradation) with **Google Gemini** strategy briefings—backed by a resilient **DuckDB Grounded Fallback** engine that maintains 100% uptime even when external LLM quota is unavailable.

---

## 🏗️ System Architecture

```mermaid
flowchart LR
    A[FastF1 Telemetry & Weather<br/>2024–2026 Seasons] --> B[(DuckDB Lakehouse<br/>8 Tables • 293K+ Rows)]
    B -->|Parquet Export| C[(Google BigQuery<br/>asia-south1)]
    C --> D[Looker Studio<br/>Executive Scorecard]
    B -->|Column-Pruned SQL| E[FastAPI Backend<br/>Analytics + 10-Model ML + Gemini AI]
    E -->|JSON < 250ms| F[Pit-Wall Console v5.0<br/>Tailwind • Plotly.js • Marked.js]
    G[GitHub main Branch] -->|CI/CD Trigger| H[Google Cloud Build<br/>Dockerfile]
    H --> I[Google Cloud Run<br/>2 vCPU • 2 GiB • Min=0]
    I --> F
    ```

---

## 📂 Lean Six Sigma DMAIC Consulting Portfolio

Full project governance, empirical measurements, root-cause diagnostics, and operational SOPs are documented in [`docs_consulting/`](docs_consulting/):

| Phase | Document | Key Deliverable |
| :--- | :--- | :--- |
| **1. Define** | [`01_DMAIC_Project_Charter.md`](docs_consulting/01_DMAIC_Project_Charter.md) | VOC → CTQ translation, SIPOC, problem statement & KPI charter |
| **2. Measure** | [`02_DMAIC_Measure_Baseline_Report.md`](docs_consulting/02_DMAIC_Measure_Baseline_Report.md) | Automated MSA benchmark across 8 DuckDB tables (`178.21 ms` mean SQL vs. `~3,200 ms` UI) |
| **3. Analyze** | [`03_DMAIC_Analyze_Root_Cause.md`](docs_consulting/03_DMAIC_Analyze_Root_Cause.md) | Pareto root-cause proof (~90.7% delay from monolithic reruns) & schema inventory |
| **4. Improve** | [`04_DMAIC_Improve_Implementation_Log.md`](docs_consulting/04_DMAIC_Improve_Implementation_Log.md) | BigQuery BI, FastAPI + Gemini AI, Pit-Wall Console v5.0 & Cloud Run deployment |
| **5. Control** | [`05_DMAIC_Control_Plan_and_Closure.md`](docs_consulting/05_DMAIC_Control_Plan_and_Closure.md) | Before/After KPI sign-off, production control plan, ₹100 budget guardrail & SOPs |

---

## 🗂️ Repository Structure

```text
f1-telemetry-ai-platform/
├── data/
│   ├── f1_lakehouse.duckdb              # Precomputed 2024–2026 DuckDB lakehouse
│   └── raw_weather/                     # Session weather Parquet archives (incl. 2026 R21)
├── docs_consulting/                     # Complete 5-phase Lean Six Sigma DMAIC documentation
├── frontend/
│   ├── index.html                       # Dark-mode Pit-Wall Console v5.0 shell + latency meter
│   └── tabs/                            # Modular JS views (telemetry, analytics, ML, AI engineer)
├── src/
│   ├── analytics/                       # Telemetry, tyres, driver DNA & race flow engines
│   ├── api/                             # FastAPI entrypoint, analytics & ML/AI routers
│   ├── models/                          # 10-model supervised, unsupervised, Bayesian & MC suite
│   ├── pipeline/                        # Ingestion scripts, circuit sync & BigQuery SQL/Parquet
│   └── precompute_lakehouse.py          # Lakehouse feature precomputation script
├── Dockerfile                           # Production container (Python + libgomp1 for XGBoost)
├── requirements-cloud.txt               # Lean production dependencies
└── README.md
```

---

## ⚡ Local & Docker Quickstart

### Run Locally (Python)
```bash
git clone [https://github.com/kamble-vatsala/f1-telemetry-ai-platform.git](https://github.com/kamble-vatsala/f1-telemetry-ai-platform.git)
cd f1-telemetry-ai-platform

python -m venv venv
# Windows: venv\Scripts\activate  |  macOS/Linux: source venv/bin/activate
pip install -r requirements-cloud.txt

# Optional: enable full Gemini narrative responses in Tab 4
echo "GEMINI_API_KEY=your_api_key_here" > .env

uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --reload
```
Open **`http://127.0.0.1:8080`** in your browser.

### Run via Docker
```bash
docker build -t apex-f1-pitwall .
docker run -p 8080:8080 -e GEMINI_API_KEY=your_api_key_here apex-f1-pitwall
```

---

## 👤 Author

**Vatsala Kamble**  
*Built with FastAPI, DuckDB, XGBoost, Plotly.js, Google Gemini, BigQuery, Looker Studio & Google Cloud Run.*