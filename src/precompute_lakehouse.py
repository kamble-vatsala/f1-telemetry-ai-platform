import sys
from pathlib import Path
import duckdb
import pandas as pd

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

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

DB_PATH = project_root / "data" / "f1_lakehouse.duckdb"

def precompute_and_index():
    print("=" * 60)
    print("APEX RACING F1 — LAKEHOUSE INDEXING & PRE-COMPUTATION")
    print("=" * 60)
    
    con = duckdb.connect(str(DB_PATH))
    
    # 1. Create high-performance indexes on DuckDB so telemetry & stint queries take < 10ms
    print("\n[1/3] Creating indexes on telemetry and stint lakehouse tables...")
    con.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_lookup ON telemetry_cache(Year, RoundNumber, SessionType, Driver);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_stints_lookup ON clean_stints(Year, RoundNumber, SessionType, Driver);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_laps_lookup ON clean_laps(Year, RoundNumber, SessionType, Driver);")
    con.commit()
    print("  -> Indexes created successfully.")

    # 2. Pre-compute ML models for 2025 Round 1 (Australian GP) & Round 20 (São Paulo GP)
    print("\n[2/3] Pre-computing ML models for active race weekends...")
    target_rounds = [(2025, 1, "Australian Grand Prix"), (2025, 20, "São Paulo Grand Prix")]
    
    for year, round_num, circ_name in target_rounds:
        print(f"  -> Processing {year} Round {round_num} ({circ_name})...")
        
        # Model 2, 3, 5: Separate Race Winner, Podium Probabilities, and XAI
        try:
            m235 = run_models_2_3_5_race_prediction_and_xai(year, round_num)
            if isinstance(m235, tuple) and len(m235) >= 3:
                race_df, cal_df, xai_df = m235[0], m235[1], m235[2]
                con.execute("CREATE TABLE IF NOT EXISTS precomputed_model2_race_pred (year INT, round_num INT, data VARCHAR);")
                con.execute("DELETE FROM precomputed_model2_race_pred WHERE year = ? AND round_num = ?;", [year, round_num])
                con.execute("INSERT INTO precomputed_model2_race_pred VALUES (?, ?, ?);", [year, round_num, race_df.to_json(orient="records")])
                
                con.execute("CREATE TABLE IF NOT EXISTS precomputed_model3_probabilities (year INT, round_num INT, data VARCHAR);")
                con.execute("DELETE FROM precomputed_model3_probabilities WHERE year = ? AND round_num = ?;", [year, round_num])
                con.execute("INSERT INTO precomputed_model3_probabilities VALUES (?, ?, ?);", [year, round_num, cal_df.to_json(orient="records")])
                
                con.execute("CREATE TABLE IF NOT EXISTS precomputed_model5_xai (year INT, round_num INT, data VARCHAR);")
                con.execute("DELETE FROM precomputed_model5_xai WHERE year = ? AND round_num = ?;", [year, round_num])
                con.execute("INSERT INTO precomputed_model5_xai VALUES (?, ?, ?);", [year, round_num, xai_df.to_json(orient="records")])
        except Exception as e:
            print(f"     Warning on Models 2/3/5: {e}")

        # Model 10: Monte Carlo
        try:
            mc_df = run_model10_monte_carlo_upcoming_race_simulator(year, circ_name, 300, 0.2, 0.4)
            if isinstance(mc_df, pd.DataFrame):
                con.execute("CREATE TABLE IF NOT EXISTS precomputed_model10_montecarlo (year INT, round_num INT, data VARCHAR);")
                con.execute("DELETE FROM precomputed_model10_montecarlo WHERE year = ? AND round_num = ?;", [year, round_num])
                con.execute("INSERT INTO precomputed_model10_montecarlo VALUES (?, ?, ?);", [year, round_num, mc_df.to_json(orient="records")])
        except Exception as e:
            print(f"     Warning on Model 10: {e}")

    con.commit()
    con.close()
    print("\n[3/3] Pre-computation & Indexing complete! All tables stored locally in DuckDB.")

if __name__ == "__main__":
    precompute_and_index()