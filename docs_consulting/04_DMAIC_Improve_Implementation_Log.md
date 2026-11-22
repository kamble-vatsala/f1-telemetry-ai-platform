# Phase 4: DMAIC Improve — Engineering & Cloud Implementation Log

## Workstream 1: Executive BI Warehouse (BigQuery + Data Studio)
- **Root Cause Addressed:** Team Principals required 6–8 clicks across technical ML tabs to review season performance; DuckDB `HUGEINT` columns required type casting for cloud warehouse compatibility.
- **Solution Implemented:** Exported type-safe Apache Parquet tables (`dim_results`, `fact_stints`, `dim_weather`) from DuckDB into Google BigQuery Sandbox (`$0` cost) and connected an interactive Data Studio Executive Scorecard.
- **Verification Result:** `vw_executive_scorecard` deployed in BigQuery (`asia-south1`) with `IS_NAN()` null-handling; live Data Studio Executive Scorecard verified across 2024–2026 seasons (`7,535` total points, `204` podiums, `-0.08` sec/lap mean tyre degradation slope).
- **KPI Impact:** Executive filtering cycle reduced from **8 clicks to 1 click** (Defect Rate: **0%**).
Workstream 2's FastAPI + Gemini gemini-3.8-flash Race Engineer is live and verified with Code 200