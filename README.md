# 🏎️ F1 Telemetry, Race Engineering & 10-Model AI/ML Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://f1-telemetry-ai-platform-kamblevatsala.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![DuckDB](https://img.shields.io/badge/Database-DuckDB_Lakehouse-FFF000)
![Scikit-Learn](https://img.shields.io/badge/AI%2FML-Scikit--Learn_%7C_XGBoost-F7931E)

🔗 **Live Dashboard:** [https://f1-telemetry-ai-platform-kamblevatsala.streamlit.app/](https://f1-telemetry-ai-platform-kamblevatsala.streamlit.app/)

An end-to-end Formula 1 Telemetry, Pit-Wall Strategy, and Predictive AI Lakehouse platform covering the **2024, 2025, and 2026 Formula 1 World Championships** (~188,000+ laps). Built to bridge motorsport race engineering with a complete 10-paradigm Data Science & Machine Learning curriculum.

---

## 🏛️ System Architecture

* **Storage & Analytical Engine (`DuckDB`):** Medallion Silver-layer database (`data/f1_lakehouse.duckdb`) executing vectorized SQL window functions (`LAG`, `QUANTILE_CONT`, `REGR_SLOPE`), compound-aware 107% outlier filtering, and fuel-burn lap time corrections (`0.055s/lap`).
* **Hybrid 20Hz Telemetry Engine (`FastF1` + `DuckDB`):** Resamples car sensor channels (*Speed, Throttle, Brake, Gear, DRS, GPS X/Y*) onto a uniform spatial distance grid using 1D linear interpolation (`np.interp`) to compute continuous Time Delta ($\Delta t$) and corner-by-corner minimum apex speeds.
* **Executive Dark-Mode UI (`Streamlit` + `Plotly`):** Every module includes a 3-part **Domain, Methodology & Actionable Insight Guide**, plus plain-English **Graph Summaries** and **Table Summaries** for technical and non-technical audiences alike.

---

## 📊 Workspace 1: Race Engineering & Constructor Analytics

1. **High-Frequency 20Hz Telemetry & Corner Apex Analysis:** 5-panel synchronized sensor traces, continuous $\Delta t$, Corner-by-Corner Minimum Apex Speed bar charts, and 2D GPS Circuit Dominance maps.
2. **Pit-Wall Race Trace & Position Flow:** Linearly detrended Virtual Reference Car "Worm Chart", Lap 0-to-Flag position bump charts, and Lap 1 launch vs. post-Lap 1 overtake decompositions.
3. **Tyre Stint Strategy & Fuel-Corrected Degradation:** Non-overlapping horizontal compound stint timelines, compound-conditional 107% pace boxplots (Median & IQR vs. Mean & StdDev), and fuel-corrected Ordinary Least Squares (OLS) wear slopes.
4. **Weather Crossovers, Safety Car Impact & Reliability:** Counterfactual pit-stop evaluator $[t-2 \text{ vs. } t+5]$ grading every tyre change (*Masterstroke* vs. *Costly Undercut*), deduplicated SC/VSC/Red Flag episode trackers, and DNF/DNS mechanical attrition breakdowns.
5. **Constructor Aero Philosophy & Pit-Wall Audit:** 2D median quadrant partitioning (*Peak Speed Trap* vs. *Clean Pace*) classifying teams into High-Efficiency, High-Downforce, or Low-Drag profiles, alongside pit-wall strategy success rates.
6. **Multi-Axis Driver DNA Radar & Optimal Sectors:** 6-axis percentile-normalized radar polygons, Theoretical Optimal Lap (`Best S1 + S2 + S3`) execution gaps, and season Teammate Head-to-Head benchmarks.

---

## 🤖 Workspace 2: 10-Model AI & Machine Learning Suite

| Module | AI / ML Paradigm | Motorsport Application & Evaluation Metrics |
| :--- | :--- | :--- |
| **1. Practice-to-Quali Predictor** | **Ridge ($L_2$), Lasso ($L_1$), Random Forest, XGBoost** | Predicts Saturday Qualifying grid positions from Friday Practice data & rolling form (*MAE, RMSE, $R^2$*). |
| **2. Pre-Race Outcome Ranker** | **Walk-Forward Gradient Boosted Trees (`XGBRegressor`)** | Chronological out-of-sample race finish ranker preventing lookahead leakage. |
| **3. Podium & Points Classifier** | **Platt Sigmoid Calibration (`CalibratedClassifierCV`)** | Outputs calibrated `P(Podium)` and `P(Top 10)` probabilities (*ROC-AUC, Brier Score Loss*). |
| **4. Thermal Tyre Degradation** | **Deep Learning Neural Network (`MLPRegressor 64->32 ReLU`)** | Models non-linear `Compound × TrackTemp` thermal wear response surfaces (*Adam optimizer*). |
| **5. Explainable AI (XAI)** | **Permutation Importance & Local Attribution** | Quantifies global feature gain vs. out-of-sample permutation drop and per-driver position attribution. |
| **6. Driver Style Archetypes** | **Unsupervised `StandardScaler` + `PCA (2D)` + `K-Means`** | Projects 6D Driver DNA onto 2 orthogonal eigenvectors with Silhouette Score cluster optimization. |
| **7. Green-Flag Lap Anomalies** | **Unsupervised `Isolation Forest`** | Detects multivariate sector and speed-trap anomalies caused by lockups, damage, or damp patches. |
| **8. Weekend Track Evolution** | **Time-Series Trend Decomposition** | Quantifies progressive circuit rubbering-in (`Cumulative Grip Gain` in sec) from `FP1` to `R`. |
| **9. Driver vs. Car Separation** | **Sequential Bayesian Elo / TrueSkill Rating** | Decouples Pure Driver Skill ($\mu, \sigma$ from Teammate H2H & race craft) from Constructor Car Power. |
| **10. Upcoming Race Simulator** | **1,000-Run Monte Carlo Stochastic Engine** | Simulates upcoming Grands Prix with Gaussian pace noise, wet-weather skill, SC compression & DNF shocks. |