import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

print("=== 1. EXTRACTING EXPLANATIONS & SUMMARIES FROM app.py ===")
app_text = (project_root / "app.py").read_text(encoding="utf-8", errors="ignore")
for line in app_text.splitlines():
    if any(k in line for k in ["st.subheader", "st.caption", "st.info", "st.markdown", "Purpose", "Terminolog", "Summary"]):
        clean = line.strip()
        if len(clean) < 160:
            print(" ", clean)

print("\n=== 2. CHECKING EXACT OUTPUT COLUMNS OF ALL 10 ML MODELS ===")
from src.models.ml_supervised_suite import (
    run_model1_practice_to_quali_regression,
    run_models_2_3_5_race_prediction_and_xai,
    run_model4_neural_net_tyre_deg,
)
from src.models.ml_unsupervised_bayesian_sim import (
    run_model6_pca_kmeans_driver_clustering,
    run_model7_isolation_forest_anomalies,
    run_model8_timeseries_track_evolution,
    run_model9_bayesian_driver_car_ratings,
    run_model10_monte_carlo_upcoming_race_simulator,
)

tests = [
    ("Model 1 (Quali)", lambda: run_model1_practice_to_quali_regression(2025, 1)),
    ("Models 2,3,5 (Race+XAI)", lambda: run_models_2_3_5_race_prediction_and_xai(2025, 1)),
    ("Model 4 (NN Tyre)", lambda: run_model4_neural_net_tyre_deg(2025)),
    ("Model 6 (Clustering)", lambda: run_model6_pca_kmeans_driver_clustering(2025)),
    ("Model 7 (Anomalies)", lambda: run_model7_isolation_forest_anomalies(2025, 1, "R")),
    ("Model 8 (Track Evol)", lambda: run_model8_timeseries_track_evolution(2025, 1)),
    ("Model 9 (Bayesian)", lambda: run_model9_bayesian_driver_car_ratings(2025)),
    ("Model 10 (Monte Carlo)", lambda: run_model10_monte_carlo_upcoming_race_simulator(2025, "Australian Grand Prix", 300, 0.2, 0.4)),
]

for name, fn in tests:
    try:
        res = fn()
        if isinstance(res, tuple):
            parts = [
                f"DF({list(x.columns)})" if hasattr(x, "columns")
                else f"Dict({list(x.keys())})" if isinstance(x, dict)
                else str(type(x))
                for x in res
            ]
            print(f"  {name}: " + " | ".join(parts))
        elif hasattr(res, "columns"):
            print(f"  {name}: DF({list(res.columns)})")
        else:
            print(f"  {name}: {type(res)}")
    except Exception as e:
        print(f"  {name} ERROR: {e}")