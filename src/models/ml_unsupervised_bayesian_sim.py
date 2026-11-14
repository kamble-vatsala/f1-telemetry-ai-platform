import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import duckdb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest

from src.analytics.driver_dna import analyze_driver_dna_and_sectors
from src.models.ml_supervised_suite import build_master_ml_feature_store

DB_PATH = "data/f1_lakehouse.duckdb"


def run_model6_pca_kmeans_driver_clustering(year: int = 2026):
    dna_df, _, _, _ = analyze_driver_dna_and_sectors(year, round_num=None, session_type='R')
    if dna_df.empty or len(dna_df) < 6:
        return pd.DataFrame(), {}

    feature_cols = [
        'Quali_Pace_Score', 'Race_Pace_Score', 'Tyre_Mgmt_Score',
        'Overtake_Craft_Score', 'Top_Speed_Score', 'Consistency_Score'
    ]
    X = dna_df[feature_cols].fillna(50.0)
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    var_explained = pca.explained_variance_ratio_ * 100

    dna_df['PCA_1'] = np.round(coords[:, 0], 3)
    dna_df['PCA_2'] = np.round(coords[:, 1], 3)

    best_k, best_sil, best_km = 4, -1.0, None
    for k in [3, 4]:
        if len(dna_df) > k:
            km = KMeans(n_clusters=k, n_init=15, random_state=42)
            labels = km.fit_predict(X_scaled)
            sil = silhouette_score(X_scaled, labels)
            if sil > best_sil:
                best_sil, best_k, best_km = sil, k, km

    dna_df['Cluster_ID'] = best_km.labels_

    archetype_names = {}
    for cid, grp in dna_df.groupby('Cluster_ID'):
        avg_overall = grp['Overall_DNA_Rating'].mean()
        avg_quali = grp['Quali_Pace_Score'].mean()
        avg_craft = grp['Overtake_Craft_Score'].mean()
        avg_tyre = grp['Tyre_Mgmt_Score'].mean()

        if avg_overall >= 78:
            archetype_names[cid] = "🏆 Championship Frontrunners"
        elif avg_craft >= avg_quali and avg_tyre >= 70:
            archetype_names[cid] = "🛞 Sunday Race & Tyre Specialists"
        elif avg_quali > avg_craft:
            archetype_names[cid] = "⚡ Saturday Quali / Top-Speed Chargers"
        else:
            archetype_names[cid] = "🛡️ Midfield Metronomes & Defenders"

    dna_df['Driver_Archetype'] = dna_df['Cluster_ID'].map(archetype_names)

    notes = {
        "f1_context": (
            "Not all F1 drivers extract lap time the same way. Some excel over a single explosive Qualifying lap, "
            "while others qualify modestly but climb through the pack on Sunday via tyre preservation and race craft."
        ),
        "ds_methodology": (
            f"Unsupervised Learning (Z-Score Standardization + PCA + K-Means): PCA projects the 6-dimensional Driver DNA "
            f"matrix onto 2 orthogonal eigenvectors capturing {var_explained[0]+var_explained[1]:.1f}% of total variance "
            f"(PC1: {var_explained[0]:.1f}%, PC2: {var_explained[1]:.1f}%). K-Means (k={best_k}, Silhouette Score={best_sil:.3f}) "
            "partitions drivers by minimizing within-cluster sum-of-squares (WCSS)."
        ),
        "key_insight": (
            f"PCA + K-Means automatically segmented {len(dna_df)} drivers into {best_k} distinct racing archetypes "
            f"with a Silhouette separation score of {best_sil:.3f}."
        )
    }

    return dna_df, notes


def run_model7_isolation_forest_anomalies(year: int, round_num: int, session_type: str = 'R'):
    con = duckdb.connect(DB_PATH, read_only=True)
    laps = con.execute(f"""
        SELECT Driver, OfficialTeamName AS Team, LapNumber, Compound, TyreLife,
               LapTime_Sec, Sector1_Sec, Sector2_Sec, Sector3_Sec, SpeedST, TrackStatus, IsPitLap
        FROM clean_laps
        WHERE Year = {year} AND RoundNumber = {round_num} AND SessionType = '{session_type}'
              AND TrackStatus = '1' AND NOT COALESCE(IsPitLap, FALSE) AND LapNumber > 1
              AND LapTime_Sec IS NOT NULL AND Sector1_Sec IS NOT NULL AND Sector2_Sec IS NOT NULL AND Sector3_Sec IS NOT NULL
    """).df()
    con.close()

    if len(laps) < 20:
        return pd.DataFrame(), {}

    for col in ['LapTime_Sec', 'Sector1_Sec', 'Sector2_Sec', 'Sector3_Sec']:
        med = laps.groupby(['Driver', 'Compound'])[col].transform('median')
        laps[f'{col}_Dev'] = (laps[col] - med).round(3)

    features = ['LapTime_Sec_Dev', 'Sector1_Sec_Dev', 'Sector2_Sec_Dev', 'Sector3_Sec_Dev']
    X = laps[features].fillna(0)

    iso = IsolationForest(contamination=0.06, random_state=42)
    laps['Anomaly_Flag'] = iso.fit_predict(X)
    laps['Anomaly_Score'] = np.round(-iso.score_samples(X), 3)

    anomalies_df = laps[(laps['Anomaly_Flag'] == -1) & (laps['LapTime_Sec_Dev'] > 0.8)].copy()
    anomalies_df['Worst_Sector'] = anomalies_df[['Sector1_Sec_Dev', 'Sector2_Sec_Dev', 'Sector3_Sec_Dev']].idxmax(axis=1).str.replace('_Sec_Dev', '')
    anomalies_df = anomalies_df.sort_values('LapTime_Sec_Dev', ascending=False).reset_index(drop=True)

    notes = {
        "f1_context": (
            "During a 70-lap race, how do engineers immediately spot if a rival suffered a front-wing endplate loss, "
            "a brake lockup in Sector 2, or a slow puncture on a green-flag lap? By monitoring multivariate deviations "
            "across Sector 1, Sector 2, Sector 3, and Speed Traps relative to the driver's own stint baseline."
        ),
        "ds_methodology": (
            "Isolation Forest (Unsupervised Tree-Based Anomaly Detection): Randomly partitions the 4D feature space "
            "(ΔLap, ΔS1, ΔS2, ΔS3). True anomalies require significantly fewer random splits to isolate (shorter path length "
            "in the binary tree) than normal racing laps."
        ),
        "key_insight": (
            f"Scanned {len(laps)} green-flag racing laps and isolated {len(anomalies_df)} significant on-track pace anomalies. "
            + (f"Largest single-lap green-flag loss: {anomalies_df.iloc[0]['Driver']} on Lap {int(anomalies_df.iloc[0]['LapNumber'])} "
               f"(+{anomalies_df.iloc[0]['LapTime_Sec_Dev']:.2f}s lost primarily in {anomalies_df.iloc[0]['Worst_Sector']})."
               if not anomalies_df.empty else "")
        )
    }

    return anomalies_df, notes


def run_model8_timeseries_track_evolution(year: int, round_num: int):
    con = duckdb.connect(DB_PATH, read_only=True)
    sessions_df = con.execute(f"""
        SELECT SessionType,
               ROUND(MIN(LapTime_Sec) FILTER (WHERE LapTime_Sec > 40 AND NOT COALESCE(IsPitLap, FALSE)), 3) AS BestLap_Sec,
               ROUND(QUANTILE_CONT(LapTime_Sec, 0.10) FILTER (WHERE LapTime_Sec > 40 AND NOT COALESCE(IsPitLap, FALSE)), 3) AS Top10PctLap_Sec,
               ROUND(AVG(AvgTrackTemp), 2) AS TrackTemp_C,
               BOOL_OR(IsWetSession) AS IsWet
        FROM clean_laps
        WHERE Year = {year} AND RoundNumber = {round_num}
        GROUP BY SessionType
    """).df()
    con.close()

    order_map = {'FP1': 1, 'FP2': 2, 'FP3': 3, 'SQ': 4, 'S': 5, 'Q': 6, 'R': 7}
    sessions_df['ChronologicalOrder'] = sessions_df['SessionType'].map(lambda x: order_map.get(x, 9))
    sessions_df = sessions_df.dropna(subset=['BestLap_Sec']).sort_values('ChronologicalOrder').reset_index(drop=True)

    if len(sessions_df) >= 2:
        first_lap = sessions_df['BestLap_Sec'].iloc[0]
        sessions_df['CumulativeGripGain_Sec'] = np.round(first_lap - sessions_df['BestLap_Sec'], 3)
        dry_sub = sessions_df[sessions_df['IsWet'] == False]
        if len(dry_sub) >= 2:
            slope, _ = np.polyfit(range(len(dry_sub)), dry_sub['BestLap_Sec'], 1)
            evo_rate = round(float(-slope), 3)
        else:
            evo_rate = round(float((first_lap - sessions_df['BestLap_Sec'].min()) / max(1, len(sessions_df) - 1)), 3)
    else:
        sessions_df['CumulativeGripGain_Sec'] = 0.0
        evo_rate = 0.0

    notes = {
        "f1_context": (
            "An F1 circuit is 'green' (dusty and low-grip) on Friday morning in FP1. As 20+ cars lay down heated Pirelli "
            "rubber over the weekend, the asphalt grip increases dramatically ('Track Evolution'). If a driver doesn't "
            "improve by at least the Track Evolution rate between Q1 and Q3, they effectively went backwards!"
        ),
        "ds_methodology": (
            "Chronological Time-Series Trend Decomposition: Tracks the 1st-order difference ΔY_t = Y_t - Y_{t-1} and "
            "ordinary least squares trend slope across sequential weekend sessions, controlling for wet-weather regime shifts."
        ),
        "key_insight": (
            f"Average Track Evolution rate across the weekend: ~{evo_rate:+.3f}s faster per session progression. "
            f"Fastest weekend benchmark: {sessions_df['BestLap_Sec'].min():.3f}s." if not sessions_df.empty else "Track evolution computed."
        )
    }

    return sessions_df, notes


def run_model9_bayesian_driver_car_ratings(up_to_year: int = 2026):
    df = build_master_ml_feature_store()
    df = df[df['Year'] <= up_to_year].sort_values(['Year', 'RoundNumber']).reset_index(drop=True)

    driver_skill = {}
    driver_uncertainty = {}
    team_power = {}

    for (_, _), race_grp in df.groupby(['Year', 'RoundNumber'], sort=False):
        for team, t_grp in race_grp.groupby('TeamFranchise'):
            curr_t = team_power.get(team, 1500.0)
            team_avg_fin = t_grp['FinishPosition'].mean()
            perf_delta = (11.0 - team_avg_fin) * 4.5
            team_power[team] = 0.94 * curr_t + 0.06 * (1500.0 + perf_delta * 18.0)

        for team, t_grp in race_grp.groupby('OfficialTeamName'):
            drvs = t_grp.to_dict('records')
            for d_row in drvs:
                d = d_row['Driver']
                if d not in driver_skill:
                    driver_skill[d] = 1500.0
                    driver_uncertainty[d] = 120.0

                tm_bonus = 0.0
                if len(drvs) == 2:
                    other = drvs[1] if drvs[0]['Driver'] == d else drvs[0]
                    r_self = driver_skill[d]
                    r_tm = driver_skill.get(other['Driver'], 1500.0)
                    expected_win = 1.0 / (1.0 + 10.0 ** ((r_tm - r_self) / 400.0))
                    actual_race_win = 1.0 if d_row['FinishPosition'] < other['FinishPosition'] else 0.0
                    actual_quali_win = 1.0 if d_row['QualiPosition'] < other['QualiPosition'] else 0.0
                    combined_actual = 0.65 * actual_race_win + 0.35 * actual_quali_win
                    tm_bonus = 24.0 * (combined_actual - expected_win)

                craft_bonus = np.clip((d_row['CleanGrid'] - d_row['FinishPosition']) * 1.2, -8.0, 12.0)

                sigma = max(35.0, driver_uncertainty[d] * 0.96)
                driver_uncertainty[d] = sigma
                k_scale = sigma / 85.0
                driver_skill[d] = driver_skill[d] + k_scale * (tm_bonus + 0.35 * craft_bonus)

    latest_drivers = df[df['Year'] == up_to_year].groupby('Driver').last().reset_index()
    rows = []
    for _, r in latest_drivers.iterrows():
        d = r['Driver']
        t_fran = r['TeamFranchise']
        d_elo = driver_skill.get(d, 1500.0)
        t_elo = team_power.get(t_fran, 1500.0)
        sig = driver_uncertainty.get(d, 60.0)
        combined = 0.55 * t_elo + 0.45 * d_elo
        rows.append({
            'Driver': d,
            'Team': r['OfficialTeamName'],
            'PureDriverSkill_Elo': round(d_elo, 1),
            'SkillUncertainty_Sigma': round(sig, 1),
            'ConstructorCar_Elo': round(t_elo, 1),
            'Combined_WinPower_Index': round(combined, 1)
        })

    ratings_df = pd.DataFrame(rows).sort_values('PureDriverSkill_Elo', ascending=False).reset_index(drop=True)
    ratings_df['DriverSkill_Rank'] = ratings_df.index + 1

    notes = {
        "f1_context": (
            "In Formula 1, raw race results confound two variables: 80% Car Performance and 20% Driver Skill. "
            "A brilliant driver in a 9th-place car might finish P12 every week, while an average driver in a dominant "
            "car finishes P2. By evaluating drivers strictly against their Teammate (identical machinery) plus grid-to-flag "
            "overperformance, we isolate Pure Driver Skill from Constructor Car Rating."
        ),
        "ds_methodology": (
            "Sequential Bayesian Elo / TrueSkill Updating: Each driver starts with a Gaussian prior N(μ=1500, σ=120). "
            "After every Grand Prix, ratings update via Bayes' rule proportional to prediction surprise (Actual Teammate H2H "
            "minus Logistic Expected Win Probability), weighted by decaying posterior uncertainty σ_t."
        ),
        "key_insight": (
            f"#1 Ranked Pure Driver Skill (Isolated from Car): {ratings_df.iloc[0]['Driver']} "
            f"(Skill Elo: {ratings_df.iloc[0]['PureDriverSkill_Elo']} ± {ratings_df.iloc[0]['SkillUncertainty_Sigma']}). "
            f"Highest Combined Driver+Car Index: {ratings_df.sort_values('Combined_WinPower_Index', ascending=False).iloc[0]['Driver']}."
            if not ratings_df.empty else "Bayesian ratings computed."
        )
    }

    return ratings_df, notes


def run_model10_monte_carlo_upcoming_race_simulator(
    season_year: int = 2026,
    circuit_name: str = "Las Vegas Grand Prix",
    n_simulations: int = 1000,
    rain_probability: float = 0.10,
    safety_car_probability: float = 0.65
):
    df = build_master_ml_feature_store()
    ratings_df, _ = run_model9_bayesian_driver_car_ratings(season_year)

    latest_year = df['Year'].max()
    latest_round = df[df['Year'] == latest_year]['RoundNumber'].max()
    active_grid = df[(df['Year'] == latest_year) & (df['RoundNumber'] == latest_round)].copy()

    sim_base = active_grid.merge(
        ratings_df[['Driver', 'PureDriverSkill_Elo', 'ConstructorCar_Elo', 'Combined_WinPower_Index']],
        on='Driver', how='left'
    )
    sim_base['Combined_WinPower_Index'] = sim_base['Combined_WinPower_Index'].fillna(1500.0)

    max_power = sim_base['Combined_WinPower_Index'].max()
    sim_base['BasePaceDeficit_Sec'] = ((max_power - sim_base['Combined_WinPower_Index']) / 180.0) + (sim_base['DriverRollingFinish_5R'] * 0.08)

    n_drivers = len(sim_base)
    rng = np.random.default_rng(42)

    win_counts = np.zeros(n_drivers)
    podium_counts = np.zeros(n_drivers)
    points_counts = np.zeros(n_drivers)
    dnf_counts = np.zeros(n_drivers)
    all_positions = np.zeros((n_simulations, n_drivers))

    base_pace = sim_base['BasePaceDeficit_Sec'].values
    wet_skill_bonus = ((10.0 - sim_base['DriverWetSkill_AvgFinish'].values) * 0.12)

    for sim_idx in range(n_simulations):
        is_wet = rng.random() < rain_probability
        has_late_sc = rng.random() < safety_car_probability

        noise_sigma = 1.35 if is_wet else 0.85
        if has_late_sc:
            noise_sigma *= 1.25

        perf_score = base_pace.copy()
        if is_wet:
            perf_score -= wet_skill_bonus

        perf_score += rng.normal(0, noise_sigma, size=n_drivers)

        dnf_mask = rng.random(size=n_drivers) < 0.06
        dnf_counts += dnf_mask.astype(int)
        perf_score[dnf_mask] += 100.0 + rng.uniform(0, 10, size=dnf_mask.sum())

        order = np.argsort(perf_score)
        ranks = np.empty(n_drivers, dtype=int)
        ranks[order] = np.arange(1, n_drivers + 1)

        all_positions[sim_idx, :] = ranks
        win_counts[ranks == 1] += 1
        podium_counts[ranks <= 3] += 1
        points_counts[ranks <= 10] += 1

    sim_results = pd.DataFrame({
        'Driver': sim_base['Driver'].values,
        'Team': sim_base['OfficialTeamName'].values,
        'Win_Prob_Pct': np.round((win_counts / n_simulations) * 100, 1),
        'Podium_Prob_Pct': np.round((podium_counts / n_simulations) * 100, 1),
        'Points_Prob_Pct': np.round((points_counts / n_simulations) * 100, 1),
        'Expected_Finish_Mean': np.round(all_positions.mean(axis=0), 1),
        'Median_Finish_P50': np.percentile(all_positions, 50, axis=0).astype(int),
        'Ceiling_P10': np.percentile(all_positions, 10, axis=0).astype(int),
        'Floor_P90': np.percentile(all_positions, 90, axis=0).astype(int),
        'Simulated_DNF_Risk_Pct': np.round((dnf_counts / n_simulations) * 100, 1)
    }).sort_values(['Win_Prob_Pct', 'Podium_Prob_Pct', 'Expected_Finish_Mean'], ascending=[False, False, True]).reset_index(drop=True)

    sim_results['Predicted_Rank'] = sim_results.index + 1
    top_fav = sim_results.iloc[0]

    notes = {
        "f1_context": (
            f"How do we predict an upcoming race like the {circuit_name} BEFORE cars even hit the track? "
            "Deterministic models fail because a single Safety Car, sudden rain shower, or DNF completely reshuffles the grid. "
            "F1 strategists run thousands of Monte Carlo race simulations to calculate each driver's Win %, Podium %, and "
            "10th-to-90th percentile finish range [P10 Ceiling, P90 Floor]."
        ),
        "ds_methodology": (
            f"Monte Carlo Stochastic Simulation ({n_simulations:,} iterations): Combines Bayesian Driver+Car priors with "
            f"Bernoulli weather regimes (P(Rain)={rain_probability*100:.0f}%), Safety Car bunching variance multipliers "
            f"(P(SC)={safety_car_probability*100:.0f}%), and Poisson/Bernoulli DNF shocks."
        ),
        "key_insight": (
            f"Across {n_simulations:,} simulated runnings of {circuit_name}: {top_fav['Driver']} ({top_fav['Team']}) "
            f"is the #1 favorite with a {top_fav['Win_Prob_Pct']}% Win probability and {top_fav['Podium_Prob_Pct']}% Podium probability "
            f"(80% Confidence Interval: P{top_fav['Ceiling_P10']} to P{top_fav['Floor_P90']})."
        )
    }

    return sim_results, notes