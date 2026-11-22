import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.analytics.tyre_analytics import analyze_tyres_and_pace_distributions
from src.models.predictor import simulate_pit_strategies
from src.models.ml_supervised_suite import run_models_2_3_5_race_prediction_and_xai

print("1. Testing tyre_analytics...")
res1 = analyze_tyres_and_pace_distributions(2025, 1, "R")
print("   Type:", type(res1), "| Tuple len:" if isinstance(res1, tuple) else "", len(res1) if isinstance(res1, tuple) else "")
if isinstance(res1, tuple):
    for i, item in enumerate(res1):
        print(f"   Item {i}: {type(item)}")

print("\n2. Testing simulate_pit_strategies...")
res2 = simulate_pit_strategies(57, 92.0, {"SOFT": 0.09, "MEDIUM": 0.05, "HARD": 0.03}, {"SOFT": 0.0, "MEDIUM": 0.6, "HARD": 1.2}, 22.0)
print("   Type:", type(res2))

print("\n3. Testing run_models_2_3_5_race_prediction_and_xai...")
res3 = run_models_2_3_5_race_prediction_and_xai(2025, 1)
print("   Type:", type(res3), "| Tuple len:" if isinstance(res3, tuple) else "", len(res3) if isinstance(res3, tuple) else "")
if isinstance(res3, tuple):
    for i, item in enumerate(res3):
        print(f"   Item {i}: {type(item)}")