import os
import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.analytics.telemetry_engine import analyze_session_telemetry
from src.analytics.race_analytics import analyze_race_flow_and_trace
from src.analytics.tyre_analytics import analyze_tyres_and_pace_distributions
from src.analytics.driver_dna import analyze_driver_dna_and_sectors
from src.analytics.incident_weather_analytics import analyze_weather_vsc_and_dnfs
from src.analytics.constructor_analytics import analyze_constructors_and_strategy
from src.models.ml_supervised_suite import (
    run_model1_practice_to_quali_regression,
    run_models_2_3_5_race_prediction_and_xai,
    run_model4_neural_net_tyre_deg
)
from src.models.ml_unsupervised_bayesian_sim import (
    run_model6_pca_kmeans_driver_clustering,
    run_model7_isolation_forest_anomalies,
    run_model8_timeseries_track_evolution,
    run_model9_bayesian_driver_car_ratings,
    run_model10_monte_carlo_upcoming_race_simulator
)
from src.models.predictor import simulate_pit_strategies

# ==============================================================================
#PAGE CONFIG & EXECUTIVE DARK THEME CSS
# ==============================================================================
st.set_page_config(
    page_title="F1 Telemetry, Engineering & AI Platform",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .kpi-card {
        background: linear-gradient(145deg, #161922, #1E222D);
        border: 1px solid #2D3343;
        border-left: 4px solid #E80020;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 6px;
    }
    .kpi-label { font-size: 0.72rem; color: #9BA3B0; text-transform: uppercase; letter-spacing: 0.6px; }
    .kpi-value { font-size: 1.10rem; font-weight: 700; color: #FFFFFF; margin-top: 2px; }
    .podium-card {
        background: linear-gradient(135deg, #181C26, #12151E);
        border: 1px solid #2E3646;
        border-radius: 10px;
        padding: 10px 16px;
        margin-bottom: 10px;
    }
    .podium-gold { border-top: 4px solid #FFD700; }
    .podium-silver { border-top: 4px solid #C0C0C0; }
    .podium-bronze { border-top: 4px solid #CD7F32; }
    .podium-rank { font-size: 0.76rem; font-weight: 700; color: #A0AAB8; text-transform: uppercase; }
    .podium-driver { font-size: 1.08rem; font-weight: 800; color: #FFFFFF; margin: 2px 0; }
    .podium-meta { font-size: 0.82rem; color: #00F5D4; }
    .note-box {
        background-color: #151922;
        border: 1px solid #283040;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 8px;
        font-size: 0.90rem;
        line-height: 1.5;
    }
    .note-title { font-weight: 700; color: #00F5D4; margin-bottom: 5px; font-size: 0.95rem; }
    .graph-summary {
        background-color: #131822;
        border-left: 4px solid #00F5D4;
        border-radius: 6px;
        padding: 10px 16px;
        margin-top: -6px;
        margin-bottom: 24px;
        font-size: 0.92rem;
        color: #E2E8F0;
    }
    .table-summary {
        background-color: #161824;
        border-left: 4px solid #FFB703;
        border-radius: 6px;
        padding: 10px 16px;
        margin-top: 4px;
        margin-bottom: 26px;
        font-size: 0.91rem;
        color: #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

DB_PATH = "data/f1_lakehouse.duckdb"

TYRE_COLORS = {
    "SOFT": "#FF3333",
    "MEDIUM": "#FFE600",
    "HARD": "#FFFFFF",
    "INTERMEDIATE": "#39B54A",
    "WET": "#00AEEF",
    "UNKNOWN": "#888888"
}

TEAM_COLORS = {
    "Red Bull Racing": "#3671C6",
    "Ferrari": "#E80020",
    "Mercedes": "#27F4D2",
    "McLaren": "#FF8000",
    "Aston Martin": "#229971",
    "Alpine": "#0093CC",
    "Williams": "#64C4FF",
    "Racing Bulls": "#6692FF",
    "RB": "#6692FF",
    "Haas F1 Team": "#B6BABD",
    "Kick Sauber": "#52E252",
    "Audi": "#F50537",
    "Sauber / Audi": "#52E252",
    "Cadillac": "#D4AF37"
}


# ==============================================================================
# CACHED DATA & MODEL WRAPPERS (INSTANT < 0.5s SWITCHING)
# ==============================================================================
@st.cache_data(ttl=600)
def query_df(sql: str) -> pd.DataFrame:
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute(sql).df()
    con.close()
    return df


@st.cache_data(ttl=600)
def cached_telemetry(year: int, round_num: int, session_type: str, driver_lap_tuple: tuple, v: int = 3):
    return analyze_session_telemetry(year, round_num, session_type, list(driver_lap_tuple))


@st.cache_data(ttl=600)
def cached_race_flow(year: int, round_num: int, session_type: str):
    return analyze_race_flow_and_trace(year, round_num, session_type)


@st.cache_data(ttl=600)
def cached_tyre_analytics(year: int, round_num: int, session_type: str):
    return analyze_tyres_and_pace_distributions(year, round_num, session_type)


@st.cache_data(ttl=600)
def cached_weather_incidents(year: int, round_num: int, session_type: str):
    return analyze_weather_vsc_and_dnfs(year, round_num, session_type)


@st.cache_data(ttl=600)
def cached_constructors(year: int, round_num: int, session_type: str):
    return analyze_constructors_and_strategy(year, round_num, session_type)


@st.cache_data(ttl=600)
def cached_driver_dna(year: int, round_num: int, session_type: str):
    return analyze_driver_dna_and_sectors(year, round_num, session_type)


@st.cache_data(ttl=600)
def cached_model1_quali(year: int, round_num: int):
    return run_model1_practice_to_quali_regression(year, round_num)


@st.cache_data(ttl=600)
def cached_models_2_3_5_race(year: int, round_num: int):
    return run_models_2_3_5_race_prediction_and_xai(year, round_num)


@st.cache_data(ttl=600)
def cached_model4_nn(year: int):
    return run_model4_neural_net_tyre_deg(year)


@st.cache_data(ttl=600)
def cached_model6_pca(year: int):
    return run_model6_pca_kmeans_driver_clustering(year)


@st.cache_data(ttl=600)
def cached_model7_anomalies(year: int, round_num: int, session_type: str):
    return run_model7_isolation_forest_anomalies(year, round_num, session_type)


@st.cache_data(ttl=600)
def cached_model8_evo(year: int, round_num: int):
    return run_model8_timeseries_track_evolution(year, round_num)


@st.cache_data(ttl=600)
def cached_model9_bayes(year: int):
    return run_model9_bayesian_driver_car_ratings(year)


@st.cache_data(ttl=600)
def cached_model10_mc(year: int, target_name: str, n_sims: int, rain_p: float, sc_p: float):
    return run_model10_monte_carlo_upcoming_race_simulator(year, target_name, n_sims, rain_p, sc_p)


# ==============================================================================
# REUSABLE STORYTELLING, GRAPH SUMMARY & TABLE SUMMARY HELPERS
# ==============================================================================
def render_storytelling_notes(notes: dict, title_prefix: str = "Executive & Methodology Guide"):
    if not notes:
        return
    with st.expander(f"📖 {title_prefix} — Context, Methodology & Key Findings", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="note-box"><div class="note-title">🏎️ F1 Domain & Engineering Concept</div>{notes.get("f1_context", "")}</div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="note-box"><div class="note-title">🧮 Statistical / AI / ML Methodology</div>{notes.get("ds_methodology", "")}</div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="note-box"><div class="note-title">💡 Actionable Result & Key Takeaway</div>{notes.get("key_insight", "")}</div>', unsafe_allow_html=True)


def render_graph_summary(line1: str, line2: str = ""):
    text = f"📊 <b>Graph Summary:</b> {line1}" + (f" {line2}" if line2 else "")
    st.markdown(f'<div class="graph-summary">{text}</div>', unsafe_allow_html=True)


def render_table_summary(line1: str, line2: str = ""):
    text = f"📋 <b>Table Summary:</b> {line1}" + (f" {line2}" if line2 else "")
    st.markdown(f'<div class="table-summary">{text}</div>', unsafe_allow_html=True)


# ==============================================================================
# SIDEBAR: GLOBAL EVENT & SESSION SELECTOR
# ==============================================================================
st.sidebar.title("🏎️ F1 Telemetry & AI Hub")
st.sidebar.caption("Engineering Analytics & Predictive AI")

if not os.path.exists(DB_PATH):
    st.error("Database not found! Run `python src/pipeline/build_clean_database.py` first.")
    st.stop()

years_df = query_df("SELECT DISTINCT Year FROM clean_laps ORDER BY Year DESC")
selected_year = st.sidebar.selectbox("📅 Select Season", years_df['Year'].tolist(), index=0)

events_df = query_df(f"""
    SELECT DISTINCT RoundNumber, EventName, Country, Location, EventFormat
    FROM clean_laps WHERE Year = {selected_year} ORDER BY RoundNumber DESC
""")
event_options = {
    f"R{int(r.RoundNumber):02d} - {r.EventName}": (int(r.RoundNumber), r.EventName, r.Location, r.EventFormat)
    for _, r in events_df.iterrows()
}
selected_event_label = st.sidebar.selectbox("🏁 Select Grand Prix", list(event_options.keys()), index=0)
selected_round, selected_event_name, selected_location, selected_format = event_options[selected_event_label]

sessions_df = query_df(f"""
    SELECT DISTINCT SessionType FROM clean_laps 
    WHERE Year = {selected_year} AND RoundNumber = {selected_round}
""")
session_order = ['R', 'Q', 'S', 'SQ', 'FP3', 'FP2', 'FP1']
avail_sessions = sorted(
    sessions_df['SessionType'].tolist(),
    key=lambda x: session_order.index(x) if x in session_order else 99
)
selected_session = st.sidebar.selectbox("⏱️ Select Session", avail_sessions, index=0)

# Query Top 5 Season Championship Drivers dynamically for default comparisons
season_top5_df = query_df(f"""
    SELECT Driver, SUM(Points) AS Pts
    FROM clean_results
    WHERE Year = {selected_year} AND SessionType IN ('R', 'S')
    GROUP BY Driver
    ORDER BY Pts DESC
    LIMIT 5
""")
season_top5_drivers = season_top5_df['Driver'].tolist()

st.sidebar.divider()
st.sidebar.markdown(f"**🏆 {selected_year} Season Top 5 Drivers:**")
st.sidebar.caption(", ".join(season_top5_drivers) if season_top5_drivers else "Auto-detected from session")

laps_df = query_df(f"""
    SELECT * FROM clean_laps 
    WHERE Year = {selected_year} AND RoundNumber = {selected_round} AND SessionType = '{selected_session}'
    ORDER BY Driver, LapNumber
""")

podium_df = query_df(f"""
    SELECT FinishPosition, Driver, FullName, OfficialTeamName AS Team, GridPosition, Points, Status, Time_Sec
    FROM clean_results
    WHERE Year = {selected_year} AND RoundNumber = {selected_round} AND SessionType = '{selected_session}'
          AND FinishPosition IN (1, 2, 3)
    ORDER BY FinishPosition ASC
""")

if podium_df.empty and not laps_df.empty and laps_df['LapTime_Sec'].notna().any():
    fast3 = (
        laps_df[laps_df['LapTime_Sec'].notna()]
        .sort_values('LapTime_Sec')
        .groupby('Driver', as_index=False)
        .first()
        .sort_values('LapTime_Sec')
        .head(3)
        .reset_index(drop=True)
    )
    fast3['FinishPosition'] = fast3.index + 1
    fast3['FullName'] = fast3['Driver']
    fast3['Team'] = fast3['OfficialTeamName']
    fast3['Points'] = 0.0
    fast3['Status'] = fast3['LapTime_Sec'].apply(lambda s: f"{s:.3f}s")
    podium_df = fast3[['FinishPosition', 'Driver', 'FullName', 'Team', 'Points', 'Status']]

all_drivers = sorted(laps_df['Driver'].unique().tolist())
default_top5_in_session = [d for d in season_top5_drivers if d in all_drivers]
if len(default_top5_in_session) < 5:
    for d in all_drivers:
        if d not in default_top5_in_session:
            default_top5_in_session.append(d)
        if len(default_top5_in_session) == 5:
            break

driver_color_map = {}
palette = px.colors.qualitative.Bold
for idx, drv in enumerate(all_drivers):
    t_vals = laps_df.loc[laps_df['Driver'] == drv, 'OfficialTeamName'].dropna()
    t_name = t_vals.iloc[0] if not t_vals.empty else ""
    driver_color_map[drv] = TEAM_COLORS.get(t_name, palette[idx % len(palette)])


# ==============================================================================
# TOP HEADER, KPI CARDS & OFFICIAL PODIUM BANNER
# ==============================================================================
st.markdown(f"## 🏁 {selected_year} {selected_event_name} — Session: {selected_session}")

if not laps_df.empty:
    best_lap = laps_df['SessionBestLap_Sec'].iloc[0]
    track_temp = laps_df['AvgTrackTemp'].iloc[0] or 0.0
    air_temp = laps_df['AvgAirTemp'].iloc[0] or 0.0
    is_wet = bool(laps_df['IsWetSession'].iloc[0])
    fastest_row = laps_df.loc[laps_df['LapTime_Sec'].idxmin()] if laps_df['LapTime_Sec'].notna().any() else None
    fastest_str = f"{fastest_row['Driver']} ({best_lap:.3f}s)" if fastest_row is not None and pd.notna(best_lap) else "N/A"

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown(f'<div class="kpi-card"><div class="kpi-label">📍 Circuit & Format</div><div class="kpi-value">{selected_location} ({selected_format})</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="kpi-card"><div class="kpi-label">⚡ Fastest Session Lap</div><div class="kpi-value">{fastest_str}</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="kpi-card"><div class="kpi-label">🌡️ Track / Air Temp</div><div class="kpi-value">{track_temp:.1f}°C / {air_temp:.1f}°C</div></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="kpi-card"><div class="kpi-label">🌧️ Surface Condition</div><div class="kpi-value">{"WET / INTERS" if is_wet else "DRY SLICKS"}</div></div>', unsafe_allow_html=True)
    k5.markdown(f'<div class="kpi-card"><div class="kpi-label">📊 Laps & Active Cars</div><div class="kpi-value">{len(laps_df):,} Laps ({len(all_drivers)} Cars)</div></div>', unsafe_allow_html=True)

if len(podium_df) >= 3:
    p1, p2, p3 = podium_df.iloc[0], podium_df.iloc[1], podium_df.iloc[2]
    pod_c1, pod_c2, pod_c3 = st.columns(3)
    pod_c1.markdown(
        f'<div class="podium-card podium-gold"><div class="podium-rank">🥇 P1 — SESSION WINNER / LEADER</div>'
        f'<div class="podium-driver">{p1["FullName"]} ({p1["Driver"]})</div>'
        f'<div class="podium-meta">{p1["Team"]} • {p1["Points"]:.0f} Pts • Status: {p1["Status"]}</div></div>',
        unsafe_allow_html=True
    )
    pod_c2.markdown(
        f'<div class="podium-card podium-silver"><div class="podium-rank">🥈 P2 — SECOND PLACE</div>'
        f'<div class="podium-driver">{p2["FullName"]} ({p2["Driver"]})</div>'
        f'<div class="podium-meta">{p2["Team"]} • {p2["Points"]:.0f} Pts • Status: {p2["Status"]}</div></div>',
        unsafe_allow_html=True
    )
    pod_c3.markdown(
        f'<div class="podium-card podium-bronze"><div class="podium-rank">🥉 P3 — THIRD PLACE</div>'
        f'<div class="podium-driver">{p3["FullName"]} ({p3["Driver"]})</div>'
        f'<div class="podium-meta">{p3["Team"]} • {p3["Points"]:.0f} Pts • Status: {p3["Status"]}</div></div>',
        unsafe_allow_html=True
    )


# ==============================================================================
# COMPACT SINGLE-ROW DROPDOWN NAVIGATION BAR (NO INTERNAL NUMBERS OR CLUTTER!)
# ==============================================================================
nav_col1, nav_col2 = st.columns([1, 2])

with nav_col1:
    workspace = st.selectbox(
        "🧭 Select Workspace",
        [
            "📊 Race Engineering & Telemetry Analytics",
            "🤖 AI & Machine Learning Suite"
        ]
    )

with nav_col2:
    if workspace == "📊 Race Engineering & Telemetry Analytics":
        selected_module = st.selectbox(
            "🔎 Select Engineering Analysis View",
            [
                "📡 High-Frequency Telemetry & Corner Apex Analysis",
                "📈 Pit-Wall Race Trace, Position Flow & Overtakes",
                "🛞 Tyre Stint Strategy, Fuel-Corrected Degradation & Pace",
                "🌧️ Weather Crossovers, Safety Car Impact & Reliability",
                "🏗️ Constructor Aero Philosophy & Pit-Wall Strategy Audit",
                "🧬 Multi-Axis Driver DNA Radar, Optimal Sectors & Teammates"
            ]
        )
    else:
        selected_module = st.selectbox(
            "🧠 Select AI / Machine Learning Model",
            [
                "⚡ Practice-to-Qualifying Pace Predictor (Regularized & Tree Regression)",
                "🏁 Pre-Race Classification Ranker (Walk-Forward Gradient Boosting)",
                "🎯 Podium & Points Probability Classifier (Platt Sigmoid Calibration)",
                "🧠 Thermal Tyre Degradation Surface (Deep Learning MLP Neural Network)",
                "🔎 Explainable AI & Feature Attribution (Permutation & Gain XAI)",
                "🧩 Driver Style & Performance Archetypes (PCA + K-Means Clustering)",
                "🚨 Green-Flag Lap Anomaly Detection (Unsupervised Isolation Forest)",
                "📉 Weekend Track Evolution & Grip Forecasting (Time-Series Trend)",
                "⚖️ Pure Driver Skill vs. Constructor Car Power (Bayesian Elo Rating)",
                "🎲 Upcoming Race & Pit Strategy Simulator (Monte Carlo Engine)"
            ]
        )

st.markdown("---")


# ==============================================================================
# WORKSPACE 1: RACE ENGINEERING & TELEMETRY ANALYTICS (ALL CHARTS & TABLES STACKED)
# ==============================================================================
if workspace == "📊 Race Engineering & Telemetry Analytics":

    # --------------------------------------------------------------------------
    # VIEW: HIGH-FREQUENCY TELEMETRY & CORNER APEX ANALYSIS
    # --------------------------------------------------------------------------
    if selected_module == "📡 High-Frequency Telemetry & Corner Apex Analysis":
        st.subheader("📡 High-Frequency Telemetry Comparison, Corner Apex Speeds & Track Dominance")

        c_sel1, c_sel2 = st.columns([2, 3])
        with c_sel1:
            comp_drivers = st.multiselect(
                "Select Drivers to Compare (Default = Season Top 5; 1st Driver = Reference Baseline)",
                all_drivers, default=default_top5_in_session
            )
            lap_mode = st.radio("Lap Selection Mode", ["Fastest Lap per Driver", "Custom Specific Lap Numbers"], horizontal=True)

        driver_lap_requests = []
        with c_sel2:
            if lap_mode == "Fastest Lap per Driver":
                driver_lap_requests = [(d, "fastest") for d in comp_drivers]
                st.info(f"Comparing fastest session laps for: {', '.join(comp_drivers)}")
            else:
                lap_cols = st.columns(max(1, len(comp_drivers)))
                for idx, d in enumerate(comp_drivers):
                    d_laps = laps_df[(laps_df['Driver'] == d) & (laps_df['LapTime_Sec'].notna())].sort_values('LapNumber')
                    l_choices = d_laps['LapNumber'].tolist()
                    fast_l = int(d_laps.loc[d_laps['LapTime_Sec'].idxmin(), 'LapNumber']) if not d_laps.empty else 1
                    def_idx = l_choices.index(fast_l) if fast_l in l_choices else 0
                    chosen_l = lap_cols[idx].selectbox(f"{d} Lap #", l_choices, index=def_idx, key=f"tel_l_{d}")
                    driver_lap_requests.append((d, chosen_l))

        if driver_lap_requests:
            with st.spinner("Aligning 20Hz car sensor telemetry onto spatial distance grid..."):
                tel_df, delta_df, corner_spd_df, drv_sum_df, corners_df, tel_notes = cached_telemetry(
                    selected_year, selected_round, selected_session, tuple(driver_lap_requests)
                )

            render_storytelling_notes(tel_notes, "Spatial Signal Interpolation & 20Hz Telemetry Guide")

            if not tel_df.empty:
                # CHART 1: 5-Row Synchronized Telemetry Chart (Full Width)
                fig_tel = make_subplots(
                    rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.035,
                    row_heights=[0.32, 0.18, 0.18, 0.16, 0.16],
                    subplot_titles=(
                        "Speed Trace (km/h) with Official Circuit Corner Markers",
                        f"Continuous Time Delta Δt (s) vs {driver_lap_requests[0][0]} (Lower/Negative = Faster)",
                        "Throttle Application (%)",
                        "Brake Pressure Indicator (%)",
                        "Gear Selection (1–8)"
                    )
                )
                neon_palette = ["#00F5D4", "#FF2A85", "#FFE600", "#9B5DE5", "#00BBF9", "#FF9F1C"]
                drv_trace_colors = {}

                for idx, label in enumerate(tel_df['Label'].unique()):
                    sub = tel_df[tel_df['Label'] == label]
                    c = neon_palette[idx % len(neon_palette)]
                    drv_trace_colors[sub['Driver'].iloc[0]] = c
                    fig_tel.add_trace(go.Scatter(x=sub['Distance'], y=sub['Speed'], name=label, line=dict(color=c, width=2), legendgroup=label), row=1, col=1)
                    fig_tel.add_trace(go.Scatter(x=sub['Distance'], y=sub['Throttle'], name=label, line=dict(color=c, width=1.5), showlegend=False, legendgroup=label), row=3, col=1)
                    fig_tel.add_trace(go.Scatter(x=sub['Distance'], y=sub['Brake_Pct'], name=label, line=dict(color=c, width=1.5), showlegend=False, legendgroup=label), row=4, col=1)
                    fig_tel.add_trace(go.Scatter(x=sub['Distance'], y=sub['nGear'], name=label, line=dict(color=c, width=1.5), showlegend=False, legendgroup=label), row=5, col=1)

                for idx, label in enumerate(delta_df['Label'].unique()):
                    sub_d = delta_df[delta_df['Label'] == label]
                    c = neon_palette[idx % len(neon_palette)]
                    fig_tel.add_trace(go.Scatter(x=sub_d['Distance'], y=sub_d['Delta_Sec'], name=f"Δ {label}", line=dict(color=c, width=2), showlegend=False), row=2, col=1)

                if not corners_df.empty:
                    for _, c_row in corners_df.iterrows():
                        fig_tel.add_vline(
                            x=c_row['Distance'], line_width=1, line_dash="dot", line_color="rgba(220,220,220,0.22)",
                            annotation_text=c_row['CornerLabel'], annotation_position="bottom right", row=1, col=1
                        )

                fig_tel.update_layout(
                    height=950, margin=dict(t=85, b=40), template="plotly_dark", hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0)
                )
                fig_tel.update_xaxes(title_text="Distance Around Circuit (Meters)", row=5, col=1)
                st.plotly_chart(fig_tel, use_container_width=True)
                render_graph_summary(
                    f"Synchronized 20Hz telemetry across {int(tel_df['Distance'].max()):,}m of physical track distance.",
                    f"Steep slopes in Row 2 (Δt) pinpoint exact corners where time was gained or lost relative to {driver_lap_requests[0][0]}, while Row 1 & Row 4 reveal braking point differences."
                )

                # CHART 2: Corner Minimum Apex Speeds (Full Width — Stacked Below)
                if not corner_spd_df.empty:
                    fig_apex = px.bar(
                        corner_spd_df, x='Corner', y='ApexMinSpeed_kmh', color='Driver',
                        barmode='group', color_discrete_map=drv_trace_colors,
                        hover_data=['EntrySpeed_kmh', 'MinGear'],
                        title="Corner-by-Corner Minimum Apex Speed (km/h — Higher = Superior Mechanical & Aerodynamic Grip)",
                        template="plotly_dark", height=480
                    )
                    st.plotly_chart(fig_apex, use_container_width=True)
                    render_graph_summary(
                        "Compares the minimum speed carried through the slowest point (apex) of every official turn around the circuit.",
                        "Drivers with taller bars in medium and high-speed turns generate stronger aerodynamic downforce and mechanical grip."
                    )

                # CHART 3: 2D GPS Track Dominance Map (Full Width — Stacked Below)
                if len(comp_drivers) >= 2 and not delta_df.empty:
                    fastest_dist = delta_df.sort_values('Speed_Interp', ascending=False).groupby('Distance').first().reset_index()
                    fig_map = px.scatter(
                        fastest_dist, x='X', y='Y', color='Driver', color_discrete_map=drv_trace_colors,
                        title="2D GPS Circuit Mini-Sector Dominance Map (Colored by Fastest Driver at Each Track Coordinate)",
                        template="plotly_dark", height=520
                    )
                    fig_map.update_traces(marker=dict(size=7))
                    fig_map.update_yaxes(scaleanchor="x", scaleratio=1, visible=False)
                    fig_map.update_xaxes(visible=False)
                    st.plotly_chart(fig_map, use_container_width=True)
                    render_graph_summary(
                        "2D GPS circuit map colored by which driver achieved the highest speed at each 4-meter track segment.",
                        "Color dominance on long straights indicates a low-drag setup, while dominance in winding sections shows superior cornering grip."
                    )

                # TABLE 1: Full-Lap Telemetry DNA Table (Full Width + Table Summary)
                st.markdown("#### 📋 Full-Lap Telemetry DNA Metrics Table")
                tel_table = drv_sum_df[[
                    'Driver', 'LapTime_Sec', 'Compound', 'TopSpeed_kmh', 'AvgSpeed_kmh',
                    'FullThrottle_Pct', 'Braking_Pct', 'Coasting_Pct', 'GearShifts', 'DRS_Open_Pct', 'TyreLife', 'LapNumber'
                ]].rename(columns={
                    'LapTime_Sec': 'Lap Time (s)', 'TopSpeed_kmh': 'Top Speed (km/h)',
                    'AvgSpeed_kmh': 'Avg Speed (km/h)', 'FullThrottle_Pct': 'Full Throttle (%)',
                    'Braking_Pct': 'Braking (%)', 'Coasting_Pct': 'Coasting (%)',
                    'GearShifts': 'Gear Shifts', 'DRS_Open_Pct': 'DRS Active (%)',
                    'TyreLife': 'Tyre Age', 'LapNumber': 'Lap #'
                })
                st.dataframe(tel_table, use_container_width=True, hide_index=True)
                render_table_summary(
                    "Summarizes whole-lap sensor metrics for each selected driver, ordered from fastest lap time to slowest.",
                    "Compare Full Throttle % and Top Speed (km/h) against Braking % and Gear Shifts to understand each driver's input style and car setup."
                )

    # --------------------------------------------------------------------------
    # VIEW: PIT-WALL RACE TRACE, POSITION FLOW & OVERTAKES
    # --------------------------------------------------------------------------
    elif selected_module == "📈 Pit-Wall Race Trace, Position Flow & Overtakes":
        st.subheader("📈 Pit-Wall Race Trace ('Worm Chart'), Position Flow & Overtake Decomposition")
        trace_df, pos_flow_df, overtake_df, race_notes = cached_race_flow(selected_year, selected_round, selected_session)
        render_storytelling_notes(race_notes, "Virtual Reference Car Detrending & Position Flow Guide")

        if not trace_df.empty:
            trace_drvs = st.multiselect(
                "Select Drivers to Highlight on Race Trace & Position Flow (Default = Season Top 5)",
                all_drivers, default=default_top5_in_session, key="tr_drvs"
            )

            # CHART 1: Virtual Reference Car Worm Chart (Full Width)
            sub_trace = trace_df[trace_df['Driver'].isin(trace_drvs)]
            fig_worm = px.line(
                sub_trace, x='LapNumber', y='GapToVirtualCar_Sec', color='Driver',
                color_discrete_map=driver_color_map, hover_data=['Position', 'Compound', 'TyreLife', 'LapTime_Sec'],
                title="Pit-Wall Race Trace ('Worm Chart'): Cumulative Time Gap vs. Virtual Reference Winner Car",
                template="plotly_dark", height=520
            )
            fig_worm.update_yaxes(autorange="reversed", title_text="Gap to Virtual Reference Car (Sec) — Higher on Chart = Ahead on Track")
            st.plotly_chart(fig_worm, use_container_width=True)
            render_graph_summary(
                "Plots each driver's cumulative race gap against a constant-speed Virtual Winner Car (inverted Y-axis so the race leader sits at the top).",
                "Upward slopes indicate faster-than-average pace; vertical ~22s drops mark pit stops; sudden vertical compression across all lines indicates a Safety Car."
            )

            # CHART 2: Lap-by-Lap Position Bump Chart (Full Width)
            sub_pos = pos_flow_df[pos_flow_df['Driver'].isin(trace_drvs)]
            fig_bump = px.line(
                sub_pos, x='LapNumber', y='Position', color='Driver', markers=True,
                color_discrete_map=driver_color_map,
                title="Lap-by-Lap Track Position Flow (Lap 0 = Starting Grid Position -> Checkered Flag)",
                template="plotly_dark", height=500
            )
            fig_bump.update_yaxes(autorange="reversed", dtick=1, title_text="Track Position (P1 at Top)")
            st.plotly_chart(fig_bump, use_container_width=True)
            render_graph_summary(
                "Tracks the exact lap-by-lap running order from the Starting Grid (Lap 0) through every lap to the finish line.",
                "Line crossings represent on-track overtakes or pit-cycle undercuts/overcuts."
            )

            # CHART 3: Overtake Bar Chart (Full Width)
            fig_gain = px.bar(
                overtake_df.sort_values('Total_PositionsGained'),
                x=['Lap1_PositionsGained', 'PostLap1_PositionsGained'], y='Driver',
                orientation='h', barmode='relative',
                title="Net Positions Gained Breakdown: Lap 1 Launch vs. Post-Lap 1 Race Craft & Strategy",
                template="plotly_dark", height=500
            )
            st.plotly_chart(fig_gain, use_container_width=True)
            render_graph_summary(
                "Separates total positions gained into standing-start Lap 1 launches versus Lap 2+ overtakes and pit-wall strategy.",
                "Positive bars extending right show net climbers through the field; negative bars extending left show drivers who lost track position."
            )

            # TABLE 1: Complete Overtake Table (Full Width + Table Summary)
            st.markdown("#### 📋 Complete Grid-to-Flag Position Gain & Overtake Table")
            ov_table = overtake_df[[
                'FinishPosition', 'Driver', 'Team', 'Total_PositionsGained',
                'GridPosition', 'Lap1Position', 'Lap1_PositionsGained',
                'PostLap1_PositionsGained', 'PositionChangeEvents', 'Status'
            ]].rename(columns={
                'FinishPosition': 'Finish Pos', 'Total_PositionsGained': 'Total Net Pos Δ',
                'GridPosition': 'Start Grid', 'Lap1Position': 'Lap 1 Pos',
                'Lap1_PositionsGained': 'Lap 1 Gain', 'PostLap1_PositionsGained': 'Post-L1 Gain',
                'PositionChangeEvents': 'Pos Changes', 'Status': 'Classification Status'
            })
            st.dataframe(ov_table, use_container_width=True, hide_index=True, height=460)
            render_table_summary(
                "Lists all classified drivers ordered by Finishing Position, prioritizing Total Net Positions Gained on the left.",
                "Compare 'Lap 1 Gain' against 'Post-L1 Gain' to see whether a driver made up ground off the start line or through mid-race overtakes and pit strategy."
            )

    # --------------------------------------------------------------------------
    # VIEW: TYRE STINT STRATEGY, FUEL-CORRECTED DEGRADATION & PACE
    # --------------------------------------------------------------------------
    elif selected_module == "🛞 Tyre Stint Strategy, Fuel-Corrected Degradation & Pace":
        st.subheader("🛞 Tyre Stint Strategy Timeline, Fuel-Corrected OLS Degradation & Pace Distribution")
        stints_df, pace_dist_df, comp_sum_df, clean_laps_df, tyre_notes = cached_tyre_analytics(
            selected_year, selected_round, selected_session
        )
        render_storytelling_notes(tyre_notes, "Compound-Aware 107% Pace Filter & Fuel-Corrected OLS Degradation Guide")

        if not stints_df.empty:
            # CHART 1: Horizontal Stint Gantt Timeline (Full Width)
            fig_gantt = go.Figure()
            shown_compounds = set()
            for _, row in stints_df.iterrows():
                comp = str(row['Compound']).upper()
                show_leg = comp not in shown_compounds
                shown_compounds.add(comp)
                fig_gantt.add_trace(go.Bar(
                    y=[row['Driver']], x=[row['StintSpanLaps']], base=[row['StartLap'] - 1],
                    orientation='h', name=comp, legendgroup=comp, showlegend=show_leg,
                    marker=dict(color=TYRE_COLORS.get(comp, "#888888"), line=dict(color="#111111", width=1.2)),
                    hovertemplate=(
                        f"<b>{row['Driver']} ({row['Team']})</b> — Stint {row['Stint']}<br>"
                        f"Compound: <b>{comp}</b> | Laps {row['StartLap']}–{row['EndLap']} ({row['StintSpanLaps']} laps)<br>"
                        f"Median Pace: {row['MedianRawPace_Sec']}s | Deg Slope: {row['TyreDegSlope_SecPerLap']} s/lap<extra></extra>"
                    )
                ))
            fig_gantt.update_layout(
                barmode='overlay', template="plotly_dark", height=540,
                title="Driver Tyre Compound Stint Timeline (Exact StartLap -> EndLap Strategy History)",
                xaxis_title="Lap Number", yaxis_title="Driver", yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig_gantt, use_container_width=True)
            render_graph_summary(
                "Shows the exact lap window and Pirelli tyre compound run by each driver across every stint of the session.",
                "Vertical alignments of stint transitions reveal pit-stop undercut windows and Safety Car / weather crossover pit waves."
            )

            # CHART 2: Compound-Aware Boxplot (Full Width)
            if not clean_laps_df.empty:
                drv_med_order = pace_dist_df['Driver'].tolist() if not pace_dist_df.empty else all_drivers
                fig_box = px.box(
                    clean_laps_df, x='Driver', y='LapTime_Sec', color='Compound',
                    category_orders={'Driver': drv_med_order}, color_discrete_map=TYRE_COLORS, points="all",
                    hover_data=['LapNumber', 'TyreLife', 'FuelCorrectedLap_Sec'],
                    title="Compound-Aware Clean Pace Distribution (Sorted Left-to-Right by Median Pace Leader)",
                    template="plotly_dark", height=480
                )
                st.plotly_chart(fig_box, use_container_width=True)
                render_graph_summary(
                    "Non-parametric pace distribution (Median horizontal line & Interquartile Range box) sorted fastest-to-slowest.",
                    "Tighter vertical boxes indicate metronomic lap-to-lap consistency (low IQR), while lower boxes indicate faster race pace."
                )

            # CHART 3: OLS Fuel-Corrected Degradation Curves (Full Width)
            deg_drvs = st.multiselect(
                "Select Drivers for Fuel-Corrected OLS Tyre Degradation Slopes (Default = Season Top 5)",
                all_drivers, default=default_top5_in_session, key="ols_drvs"
            )
            deg_sub = clean_laps_df[clean_laps_df['Driver'].isin(deg_drvs)]
            if not deg_sub.empty:
                fig_ols = px.scatter(
                    deg_sub, x='TyreLife', y='FuelCorrectedLap_Sec', color='Driver', symbol='Compound',
                    trendline="ols", color_discrete_map=driver_color_map,
                    hover_data=['LapNumber', 'Stint', 'LapTime_Sec'],
                    title="Fuel-Corrected Lap Time (s) vs. Tyre Age (Laps) with Ordinary Least Squares (OLS) Trendlines",
                    template="plotly_dark", height=480
                )
                st.plotly_chart(fig_ols, use_container_width=True)
                render_graph_summary(
                    "Isolates pure rubber degradation by subtracting the ~0.055s/lap fuel burn-off weight bonus from raw lap times.",
                    "Steeper upward OLS trendlines indicate severe thermal tyre wear; flatter lines indicate superior tyre preservation."
                )

            # TABLE 1: Compound Benchmark Summary (Full Width + Table Summary)
            st.markdown("#### 📊 Pirelli Compound Benchmark Summary Table")
            st.dataframe(comp_sum_df, use_container_width=True, hide_index=True)
            render_table_summary(
                "Aggregates stint lengths, median clean lap times, and fuel-corrected OLS degradation slopes (s/lap) for each tyre compound used.",
                "Use this table to compare the speed-versus-durability trade-off between Soft, Medium, Hard, and Wet/Intermediate compounds."
            )

            # TABLE 2: Driver Parametric vs Non-Parametric Pace Table (Full Width + Table Summary)
            st.markdown("#### 📋 Driver Pace & Consistency Table: Non-Parametric (Median/IQR) vs. Parametric (Mean/StdDev)")
            pace_table = pace_dist_df[[
                'Driver', 'Team', 'MedianPace_Sec', 'GapToPaceLeader_Sec', 'IQR_Consistency_Sec',
                'BestLap_Sec', 'FuelCorrectedMedian_Sec', 'MeanPace_Sec', 'StdDev_Sec', 'CleanLaps', 'CompoundsUsed'
            ]].rename(columns={
                'MedianPace_Sec': 'Median Pace (s)', 'GapToPaceLeader_Sec': 'Gap to Leader (s)',
                'IQR_Consistency_Sec': 'Consistency IQR (±s)', 'BestLap_Sec': 'Best Lap (s)',
                'FuelCorrectedMedian_Sec': 'Fuel-Corr Median (s)', 'MeanPace_Sec': 'Mean Pace (s)',
                'StdDev_Sec': 'Std Dev (s)', 'CleanLaps': 'Clean Laps', 'CompoundsUsed': 'Compounds Raced'
            })
            st.dataframe(pace_table, use_container_width=True, hide_index=True, height=460)
            render_table_summary(
                "Ranks all drivers by Non-Parametric Median Clean Pace (s) and Interquartile Range Consistency (±s) on the left.",
                "Comparing Median/IQR against Mean/StdDev reveals how single-lap traffic outliers skew traditional parametric averages."
            )

    # --------------------------------------------------------------------------
    # VIEW: WEATHER CROSSOVERS, SAFETY CAR IMPACT & RELIABILITY
    # --------------------------------------------------------------------------
    elif selected_module == "🌧️ Weather Crossovers, Safety Car Impact & Reliability":
        st.subheader("🌧️ Weather Crossover Pit Decisions, VSC/Safety Car Impact & Session Reliability")
        dnf_df, attr_df, neut_df, pit_dec_df, weather_tl_df, inc_notes = cached_weather_incidents(
            selected_year, selected_round, selected_session
        )
        render_storytelling_notes(inc_notes, "Counterfactual Pit-Window Evaluation & Race Control Neutralizations Guide")

        # CHART 1: Live Session Weather Evolution (Full Width)
        if not weather_tl_df.empty:
            fig_w = make_subplots(specs=[[{"secondary_y": True}]])
            fig_w.add_trace(go.Scatter(x=weather_tl_df['SessionMinute'], y=weather_tl_df['TrackTemp'], name="Track Temp (°C)", line=dict(color="#FF595E", width=2.5)), secondary_y=False)
            fig_w.add_trace(go.Scatter(x=weather_tl_df['SessionMinute'], y=weather_tl_df['AirTemp'], name="Air Temp (°C)", line=dict(color="#FFCA3A", width=2)), secondary_y=False)
            fig_w.add_trace(go.Scatter(x=weather_tl_df['SessionMinute'], y=weather_tl_df['Humidity'], name="Humidity (%)", line=dict(color="#00BBF9", width=1.5, dash="dot")), secondary_y=True)
            fig_w.update_layout(title="Live Session Weather Evolution (Track Temp °C, Air Temp °C & Relative Humidity %)", template="plotly_dark", height=420)
            fig_w.update_yaxes(title_text="Temperature (°C)", secondary_y=False)
            fig_w.update_yaxes(title_text="Humidity (%)", secondary_y=True)
            st.plotly_chart(fig_w, use_container_width=True)
            render_graph_summary(
                "Tracks minute-by-minute asphalt Track Temperature (red), Air Temperature (yellow), and Humidity (blue dotted).",
                "Sudden drops in Track Temperature paired with rising Humidity signal approaching rain and trigger wet-tyre crossover windows."
            )

        # CHART 2: Session Reliability Breakdown (Full Width)
        if not attr_df.empty:
            fig_attr = px.pie(
                attr_df, names='ReliabilityCategory', values='DriverCount', hole=0.45,
                title="Session Classification & Attrition Breakdown (Lead Lap vs. Lapped vs. DNF / DNS)",
                template="plotly_dark", height=400
            )
            st.plotly_chart(fig_attr, use_container_width=True)
            render_graph_summary(
                "Shows the proportion of the grid finishing on the lead lap, finishing lapped, or retiring due to mechanical failure or collision.",
                "High-attrition races create major points-scoring outliers for midfield constructors."
            )

        # TABLE 1: Weather Crossover & Pit Stop Evaluator (Full Width + Table Summary)
        st.markdown("#### 🔄 Weather Crossover & Tyre Pit Stop Evaluator: *'Was the Pit Decision Worth It?'*")
        if not pit_dec_df.empty:
            pit_table = pit_dec_df[[
                'Driver', 'Team', 'PitLap', 'DecisionVerdict', 'NetPositions_5Laps',
                'PaceDelta_AfterPit_Sec', 'Transition', 'PittedUnder_SC_VSC', 'Pos_BeforePit', 'Pos_5LapsAfter'
            ]].rename(columns={
                'PitLap': 'Pit Lap', 'DecisionVerdict': 'Engineering Verdict',
                'NetPositions_5Laps': 'Net Pos Δ (5 Laps)', 'PaceDelta_AfterPit_Sec': 'Pace Δ (s/lap)',
                'Transition': 'Tyre Transition', 'PittedUnder_SC_VSC': 'SC / VSC Stop?',
                'Pos_BeforePit': 'Pos Before', 'Pos_5LapsAfter': 'Pos After (L+5)'
            })
            st.dataframe(pit_table, use_container_width=True, hide_index=True, height=450)
            render_table_summary(
                "Evaluates every mid-race tyre change by comparing driver position 2 laps before pitting against position 5 laps after the pit cycle settles.",
                "Look at 'Engineering Verdict', 'Net Pos Δ (5 Laps)', and 'Pace Δ (s/lap)' on the left to see which pit calls were Masterstroke undercuts versus costly losses."
            )
        else:
            st.info("No mid-session tyre stint changes recorded in this session.")

        # TABLE 2: Safety Car, VSC & Red Flag Episode Tracker (Full Width + Table Summary)
        st.markdown("#### 🚩 Safety Car (SC), Virtual Safety Car (VSC) & Red Flag Impact Tracker")
        if not neut_df.empty:
            neut_table = neut_df[[
                'Episode', 'IncidentType', 'StartLap', 'EndLap', 'Duration_Laps',
                'BiggestPositionGainers', 'BiggestPositionLosers', 'DriversWhoPitted_CheapStop'
            ]].rename(columns={
                'IncidentType': 'Neutralization Type', 'StartLap': 'Start Lap', 'EndLap': 'End Lap',
                'Duration_Laps': 'Laps', 'BiggestPositionGainers': 'Biggest Position Gainers',
                'BiggestPositionLosers': 'Biggest Position Losers', 'DriversWhoPitted_CheapStop': 'Cheap Pit Stop Takers'
            })
            st.dataframe(neut_table, use_container_width=True, hide_index=True)
            render_table_summary(
                "Logs every distinct Safety Car, Virtual Safety Car (VSC), and Red Flag neutralization episode during the session.",
                "Highlights which drivers capitalized on a 'Cheap Pit Stop' (~11s time loss saved) and who gained or lost the most track positions."
            )
        else:
            st.success("Green-flag session: 0 Safety Car, VSC, or Red Flag neutralizations occurred during this session.")

        # TABLE 3: Complete Reliability & Classification Status (Full Width + Table Summary)
        st.markdown("#### 🛠️ Complete Driver Classification, Laps Completed & DNF/DNS Attribution Table")
        dnf_table = dnf_df[[
            'FinishPosition', 'Driver', 'Team', 'ReliabilityCategory', 'OfficialStatus',
            'GridPosition', 'LapsCompleted', 'RaceCompletion_Pct', 'Points'
        ]].rename(columns={
            'FinishPosition': 'Finish Pos', 'ReliabilityCategory': 'Reliability Category',
            'OfficialStatus': 'FIA Status', 'GridPosition': 'Start Grid',
            'LapsCompleted': 'Laps Completed', 'RaceCompletion_Pct': 'Distance (%)'
        })
        st.dataframe(dnf_table, use_container_width=True, hide_index=True, height=450)
        render_table_summary(
            "Complete 20–22 car race classification distinguishing Lead-Lap Finishers, Lapped Finishers, Mechanical DNFs, Crash DNFs, and DNS.",
            "Inspect 'Distance (%)' and 'FIA Status' to identify exact retirement causes and lap counts."
        )

    # --------------------------------------------------------------------------
    # VIEW: CONSTRUCTOR AERO PHILOSOPHY & PIT-WALL STRATEGY AUDIT
    # --------------------------------------------------------------------------
    elif selected_module == "🏗️ Constructor Aero Philosophy & Pit-Wall Strategy Audit":
        st.subheader("🏗️ Constructor Aerodynamic Setup Quadrant, Pace Benchmarks & Pit-Wall Strategy Audit")
        ev_teams_df, strat_audit_df, season_wcc_df, con_notes = cached_constructors(
            selected_year, selected_round, selected_session
        )
        render_storytelling_notes(con_notes, "Constructor Aero Quadrant Partitioning & Pit-Wall Strategy Guide")

        if not ev_teams_df.empty:
            # CHART 1: Constructor Aero Quadrant (Full Width — Spacious & Zero Overlap!)
            fig_aero = px.scatter(
                ev_teams_df.dropna(subset=['Median_Pace_Sec', 'Peak_Speed_kmh']),
                x='Peak_Speed_kmh', y='Median_Pace_Sec', color='Constructor', text='Constructor',
                symbol='Aero_Setup_Profile', color_discrete_map=TEAM_COLORS,
                size='Clean_Laps', hover_data=['Drivers_Classification', 'Tyre_Deg_Slope'],
                title="Constructor Aero Quadrant: Peak Straight-Line Speed Trap (km/h) vs. Median Clean Pace (s)",
                template="plotly_dark", height=520
            )
            fig_aero.update_traces(textposition='top center', marker=dict(size=16))
            fig_aero.update_yaxes(autorange="reversed", title_text="Median Clean Pace (s) — Higher on Chart = Faster Lap Time")
            fig_aero.update_xaxes(title_text="Peak Speed Trap (km/h) — Further Right = Higher Straight-Line Top Speed")
            st.plotly_chart(fig_aero, use_container_width=True)
            render_graph_summary(
                "Maps each Constructor's aerodynamic setup philosophy across full screen width: Top-Right = Fast in corners AND fast on straights (High Efficiency).",
                "Top-Left = High-Downforce cornering priority; Bottom-Right = Low-Drag straight-line priority."
            )

            # CHART 2: Pit-Wall Strategy Execution Bar Chart (Full Width — Stacked Below)
            if not strat_audit_df.empty:
                fig_strat = px.bar(
                    strat_audit_df.sort_values('Strategy_Success_Rate_Pct'),
                    x='Strategy_Success_Rate_Pct', y='Team', color='Team', orientation='h',
                    color_discrete_map=TEAM_COLORS, text='Strategy_Success_Rate_Pct',
                    hover_data=['Masterstroke_Calls', 'Cheap_SC_VSC_Stops', 'Net_PitCycle_Positions'],
                    title="Constructor Pit-Wall Strategy Execution Success Rate (% of Pit Calls Gaining Pace or Position)",
                    template="plotly_dark", height=460
                )
                st.plotly_chart(fig_strat, use_container_width=True)
                render_graph_summary(
                    "Ranks each Constructor's Pit Wall by the percentage of pit stops that held/gained track position or unlocked faster pace.",
                    "Teams at the bottom either suffered zero-gain stops or got undercut during pit-stop crossover windows."
                )

            # TABLE 1: Event Constructor Performance & Aero Setup Table
            st.markdown("#### 🏎️ Event Constructor Pace, Tyre Degradation & Aero Setup Table")
            ev_table = ev_teams_df[[
                'Constructor', 'Drivers_Classification', 'Points_Scored', 'Median_Pace_Sec',
                'Gap_To_Fastest_Team_Sec', 'Aero_Setup_Profile', 'Peak_Speed_kmh', 'Tyre_Deg_Slope',
                'Best_Lap_Sec', 'Consistency_IQR_Sec', 'Net_Pos_Gained', 'Clean_Laps'
            ]].rename(columns={
                'Drivers_Classification': 'Drivers & Finish', 'Points_Scored': 'Points',
                'Median_Pace_Sec': 'Median Pace (s)', 'Gap_To_Fastest_Team_Sec': 'Gap to P1 Team (s)',
                'Aero_Setup_Profile': 'Aero Setup Profile', 'Peak_Speed_kmh': 'Peak Speed (km/h)',
                'Tyre_Deg_Slope': 'Tyre Deg (s/lap)', 'Best_Lap_Sec': 'Best Lap (s)',
                'Consistency_IQR_Sec': 'Pace IQR (±s)', 'Net_Pos_Gained': 'Net Pos Gained', 'Clean_Laps': 'Clean Laps'
            })
            st.dataframe(ev_table, use_container_width=True, hide_index=True, height=430)
            render_table_summary(
                "Ranks all 10–11 Constructors in the selected session by Median Clean Pace (s), showing both drivers' classifications and total team points on the left.",
                "Compare 'Gap to P1 Team (s)', 'Aero Setup Profile', and 'Tyre Deg (s/lap)' to evaluate chassis balance."
            )

            # TABLE 2: Constructor Pit-Wall Strategy Audit Table
            if not strat_audit_df.empty:
                st.markdown("#### 🧠 Constructor Pit-Wall Strategy Execution Audit Table")
                strat_table = strat_audit_df[[
                    'Team', 'Strategy_Success_Rate_Pct', 'Masterstroke_Calls', 'WorthIt_Calls',
                    'Costly_Undercut_Calls', 'Cheap_SC_VSC_Stops', 'Net_PitCycle_Positions',
                    'Avg_PostPit_PaceDelta_Sec', 'Total_Pit_Calls'
                ]].rename(columns={
                    'Team': 'Constructor', 'Strategy_Success_Rate_Pct': 'Strategy Success (%)',
                    'Masterstroke_Calls': 'Masterstroke Calls', 'WorthIt_Calls': 'Positive Stops',
                    'Costly_Undercut_Calls': 'Costly Undercuts', 'Cheap_SC_VSC_Stops': 'Cheap SC/VSC Stops',
                    'Net_PitCycle_Positions': 'Net Pit-Cycle Pos Δ', 'Avg_PostPit_PaceDelta_Sec': 'Avg Pace Δ After Pit (s)',
                    'Total_Pit_Calls': 'Total Stops'
                })
                st.dataframe(strat_table, use_container_width=True, hide_index=True, height=420)
                render_table_summary(
                    "Audits every Constructor's pit-wall decision-making during the session, ordered by Strategy Success Rate (%).",
                    "Shows how many stops were Masterstrokes vs. Costly Undercuts, along with net positions gained or lost across all pit cycles."
                )

            # TABLE 3: Season WCC Standings & Reliability Table
            st.markdown(f"#### 🏆 {selected_year} World Constructors' Championship (WCC), Mechanical Reliability % & Tyre Wear Table")
            wcc_table = season_wcc_df.rename(columns={
                'WCC_Rank': 'WCC Rank', 'Season_Points': 'Season Points', 'Race_Wins': 'Wins',
                'Points_Finishes': 'Top-10 Finishes', 'Avg_Grid_Pos': 'Avg Start Grid',
                'Avg_Finish_Pos': 'Avg Finish Pos', 'Season_Net_Overtakes': 'Season Net Overtakes',
                'Mechanical_Finish_Rate_Pct': 'Finish Reliability (%)', 'Season_TyreDeg_s_lap': 'Season Median Deg (s/lap)'
            })
            st.dataframe(wcc_table, use_container_width=True, hide_index=True, height=430)
            render_table_summary(
                f"Full {selected_year} World Constructors' Championship standings combining points, wins, and podiums with engineering reliability metrics.",
                "Compare 'Finish Reliability (%)' and 'Season Median Deg (s/lap)' to see how mechanical durability and tyre preservation impact championship position."
            )

    # --------------------------------------------------------------------------
    # VIEW: MULTI-AXIS DRIVER DNA RADAR, OPTIMAL SECTORS & TEAMMATES
    # --------------------------------------------------------------------------
    elif selected_module == "🧬 Multi-Axis Driver DNA Radar, Optimal Sectors & Teammates":
        st.subheader("🧬 Multi-Axis Driver DNA Radar, Theoretical Optimal Sectors & Teammate Head-to-Head")
        dna_df, sectors_df, h2h_df, dna_notes = cached_driver_dna(selected_year, selected_round, selected_session)
        render_storytelling_notes(dna_notes, "Multi-Dimensional Percentile Scaling & Theoretical Optimal Sectors Guide")

        # CHART 1: 6-Axis Driver DNA Radar (Full Width — Generous Margins so Labels Never Clip!)
        avail_dna_drvs = dna_df['Driver'].tolist() if not dna_df.empty else []
        def_rad = [d for d in default_top5_in_session if d in avail_dna_drvs][:5]
        if not def_rad:
            def_rad = avail_dna_drvs[:4]
        radar_drvs = st.multiselect("Select Drivers for 6-Axis DNA Radar Comparison (Default = Season Top 5)", avail_dna_drvs, default=def_rad, key="rad_d")

        if radar_drvs and not dna_df.empty:
            categories = ['Quali Pace', 'Race Pace', 'Tyre Mgmt', 'Overtake Craft', 'Top Speed', 'Consistency']
            fig_radar = go.Figure()
            for d in radar_drvs:
                row = dna_df[dna_df['Driver'] == d].iloc[0]
                vals = [
                    row['Quali_Pace_Score'], row['Race_Pace_Score'], row['Tyre_Mgmt_Score'],
                    row['Overtake_Craft_Score'], row['Top_Speed_Score'], row['Consistency_Score']
                ]
                vals.append(vals[0])
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals, theta=categories + [categories[0]], fill='toself', name=f"{d} (Overall: {row['Overall_DNA_Rating']})"
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[40, 100])),
                margin=dict(l=90, r=90, t=60, b=50),
                template="plotly_dark", height=520,
                title="Multi-Dimensional Driver DNA Radar Polygon (45–99 Percentile Scale Across 6 Orthogonal Traits)"
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            render_graph_summary(
                "Overlays 6 percentile-normalized performance dimensions on a uniform [45–99] scale across full screen width.",
                "A larger outer polygon area signifies a complete all-round weekend performance across qualifying speed, race pace, tyre wear, overtaking, and consistency."
            )

        # CHART 2: Sector Execution Gap Bar Chart (Full Width — Stacked Below)
        if not sectors_df.empty:
            fig_sec = px.bar(
                sectors_df.sort_values('Execution_Gap_Sec', ascending=True),
                x='Execution_Gap_Sec', y='Driver', color='Driver', orientation='h',
                color_discrete_map=driver_color_map, text='Execution_Gap_Sec',
                hover_data=['Actual_Best_Lap', 'Theoretical_Best_Lap', 'Peak_SpeedTrap_kmh'],
                title="Sector Execution Gap (s): Actual Best Lap minus Theoretical Optimal Lap (Best S1 + Best S2 + Best S3)",
                template="plotly_dark", height=500
            )
            st.plotly_chart(fig_sec, use_container_width=True)
            render_graph_summary(
                "Measures how close each driver came to combining their personal best Sector 1, Sector 2, and Sector 3 into a single lap.",
                "Near-zero bars at the top indicate flawless lap execution; wider bars at the bottom reveal untapped car speed lost to traffic or a corner mistake."
            )

        # TABLE 1: Complete 6-Axis Driver DNA Ratings Table
        st.markdown("#### 🕸️ Complete 6-Axis Driver DNA Ratings Table")
        if not dna_df.empty:
            dna_table = dna_df[[
                'Driver', 'Team', 'Overall_DNA_Rating', 'Race_Pace_Score', 'Quali_Pace_Score',
                'Tyre_Mgmt_Score', 'Overtake_Craft_Score', 'Consistency_Score', 'Top_Speed_Score', 'TotalPoints'
            ]].rename(columns={
                'Overall_DNA_Rating': 'Overall DNA (0-100)', 'Race_Pace_Score': 'Race Pace Score',
                'Quali_Pace_Score': 'Quali Score', 'Tyre_Mgmt_Score': 'Tyre Mgmt Score',
                'Overtake_Craft_Score': 'Overtake Score', 'Consistency_Score': 'Consistency Score',
                'Top_Speed_Score': 'Top Speed Score', 'TotalPoints': 'Points'
            })
            st.dataframe(dna_table, use_container_width=True, hide_index=True, height=450)
            render_table_summary(
                "Ranks all drivers by their composite Overall DNA Rating (0–100 average across all 6 normalized percentile dimensions).",
                "Allows instant identification of whether a driver's rating is driven by Saturday Qualifying speed or Sunday Tyre Management and Overtaking."
            )

        # TABLE 2: Theoretical Optimal Lap vs Actual Best Lap Table
        st.markdown("#### 🧩 Theoretical Optimal Lap (Best S1 + Best S2 + Best S3) vs. Actual Best Lap & Speed Trap Table")
        sec_table = sectors_df[[
            'Driver', 'Team', 'Actual_Best_Lap', 'Theoretical_Best_Lap', 'Execution_Gap_Sec',
            'Best_S1', 'Best_S2', 'Best_S3', 'Peak_SpeedTrap_kmh'
        ]].rename(columns={
            'Actual_Best_Lap': 'Actual Best Lap (s)', 'Theoretical_Best_Lap': 'Theoretical Optimal (s)',
            'Execution_Gap_Sec': 'Execution Gap (s)', 'Best_S1': 'Best S1 (s)',
            'Best_S2': 'Best S2 (s)', 'Best_S3': 'Best S3 (s)', 'Peak_SpeedTrap_kmh': 'Speed Trap (km/h)'
        })
        st.dataframe(sec_table, use_container_width=True, hide_index=True, height=450)
        render_table_summary(
            "Compares each driver's Actual Best Lap against their Theoretical Optimal Lap (sum of personal best Sector 1, Sector 2, and Sector 3).",
            "Ordered fastest-to-slowest by Actual Best Lap, with the untapped 'Execution Gap (s)' highlighted in column 5."
            )

        # TABLE 3: Season Teammate Head-to-Head Matrix
        st.markdown(f"#### 🥊 {selected_year} Season Teammate Head-to-Head Comparison Table (Identical Car Benchmark)")
        st.dataframe(h2h_df, use_container_width=True, hide_index=True, height=450)
        render_table_summary(
            f"Groups drivers by Constructor across the entire {selected_year} season to provide the ultimate control benchmark: performance in identical machinery.",
            "Compare Season Points, Average Race Finish, and Average Grid Position between teammates within each team."
        )


# ==============================================================================
# WORKSPACE 2: AI & MACHINE LEARNING SUITE (10 INDIVIDUAL DEDICATED MODULES!)
# ==============================================================================
else:

    # --------------------------------------------------------------------------
    # AI MODULE 1: PRACTICE-TO-QUALIFYING PACE PREDICTOR
    # --------------------------------------------------------------------------
    if selected_module == "⚡ Practice-to-Qualifying Pace Predictor (Regularized & Tree Regression)":
        st.subheader(f"⚡ Practice-to-Qualifying (FP → Q) Pace Predictor — {selected_year} {selected_event_name}")
        q_preds, q_bench, q_notes = cached_model1_quali(selected_year, selected_round)
        render_storytelling_notes(q_notes, "Regularized Linear (Ridge L2 / Lasso L1) vs. Tree Ensemble Regression Guide")

        if not q_bench.empty:
            fig_qb = px.bar(
                q_bench, x='Position_MAE', y='Algorithm', color='Algorithm', orientation='h',
                text='Position_MAE', title="Out-of-Sample Qualifying Prediction Error by Regression Family (MAE in Grid Positions — Lower = Better)",
                template="plotly_dark", height=420
            )
            st.plotly_chart(fig_qb, use_container_width=True)
            render_graph_summary(
                "Compares 4 distinct supervised regression algorithms predicting Saturday Qualifying results using strictly Friday Free Practice and rolling form data.",
                "Shorter bars indicate lower Mean Absolute Error (MAE) and better filtering of Friday Practice fuel-load sandbagging."
            )

            st.markdown("#### 📊 Regression Algorithm Evaluation Metrics Table (MAE, RMSE & R²)")
            st.dataframe(q_bench, use_container_width=True, hide_index=True)
            render_table_summary(
                "Benchmarks Ridge (L2 penalty), Lasso (L1 sparse selection), Random Forest (bagging), and XGBoost (gradient boosting).",
                "Compare Position MAE (average grid spot error), Position RMSE (penalizes large outlier errors), and Score R² (variance explained)."
            )

            st.markdown("#### 📋 Predicted Saturday Qualifying Classification vs. Actual Qualifying Result")
            q_table = q_preds.rename(columns={
                'Predicted_Quali_Pos': 'AI Pred Quali Pos', 'QualiPosition': 'Actual Quali Pos',
                'OfficialTeamName': 'Constructor', 'FP_BestDelta_Pct': 'Practice Pace Δ (%)',
                'DriverRollingQuali_5R': '5-Race Rolling Quali Avg', 'Predicted_Quali_Score': 'Raw Model Score'
            })
            st.dataframe(q_table, use_container_width=True, hide_index=True, height=460)
            render_table_summary(
                "Displays every driver's AI-predicted Qualifying position on the far left alongside their actual Saturday Qualifying classification.",
                "Shows how Friday Practice pace delta (%) and 5-race rolling qualifying average combine to forecast the starting grid."
            )

    # --------------------------------------------------------------------------
    # AI MODULE 2: PRE-RACE CLASSIFICATION RANKER (WALK-FORWARD BOOSTING)
    # --------------------------------------------------------------------------
    elif selected_module == "🏁 Pre-Race Classification Ranker (Walk-Forward Gradient Boosting)":
        st.subheader(f"🏁 Walk-Forward Pre-Race Classification Ranker (XGBoost) — {selected_year} {selected_event_name}")
        r_preds, _, _, r_notes = cached_models_2_3_5_race(selected_year, selected_round)
        render_storytelling_notes(r_notes, "Walk-Forward Time-Series Gradient Boosted Trees (XGBoost) Guide")

        if not r_preds.empty:
            fig_r2 = px.scatter(
                r_preds, x='PredictedPosition', y='FinishPosition', color='OfficialTeamName',
                text='Driver', color_discrete_map=TEAM_COLORS, size='Points_Prob_Pct',
                hover_data=['CleanGrid', 'ExpectedFinishScore'],
                title="Pre-Race AI Predicted Position vs. Actual Sunday Race Finish Position (Diagonal Line = Exact Match)",
                template="plotly_dark", height=500
            )
            fig_r2.add_shape(type="line", x0=1, y0=1, x1=20, y1=20, line=dict(color="rgba(255,255,255,0.3)", dash="dash"))
            fig_r2.update_traces(textposition='top center')
            fig_r2.update_xaxes(autorange="reversed", dtick=1, title_text="AI Predicted Finish Position (P1 on Right)")
            fig_r2.update_yaxes(autorange="reversed", dtick=1, title_text="Actual Race Finish Position (P1 at Top)")
            st.plotly_chart(fig_r2, use_container_width=True)
            render_graph_summary(
                "Plots pre-race AI predicted finishing positions against actual Sunday race classifications across full screen width.",
                "Drivers sitting on the dashed diagonal finished exactly where predicted; drivers above the line outperformed expectations."
            )

            st.markdown("#### 📋 Walk-Forward Pre-Race Classification Prediction Table")
            m2_table = r_preds[[
                'PredictedPosition', 'FinishPosition', 'Driver', 'OfficialTeamName',
                'CleanGrid', 'ExpectedFinishScore', 'SprintFinishPos', 'FP_LongRunDelta_Pct', 'DriverRollingFinish_5R'
            ]].rename(columns={
                'PredictedPosition': 'AI Pred Pos', 'FinishPosition': 'Actual Finish',
                'OfficialTeamName': 'Constructor', 'CleanGrid': 'Start Grid',
                'ExpectedFinishScore': 'Expected Score', 'SprintFinishPos': 'Sprint Pos',
                'FP_LongRunDelta_Pct': 'FP Long-Run Δ (%)', 'DriverRollingFinish_5R': '5-Race Rolling Finish'
            })
            st.dataframe(m2_table, use_container_width=True, hide_index=True, height=460)
            render_table_summary(
                "Full pre-race prediction table trained strictly on historical races prior to the selected Grand Prix (zero lookahead leakage).",
                "Prioritizes AI Predicted Position and Actual Finish on the left, followed by Starting Grid, Sprint finish, and Friday long-run pace."
            )

    # --------------------------------------------------------------------------
    # AI MODULE 3: PODIUM & POINTS PROBABILITY CLASSIFIER (PLATT CALIBRATION)
    # --------------------------------------------------------------------------
    elif selected_module == "🎯 Podium & Points Probability Classifier (Platt Sigmoid Calibration)":
        st.subheader(f"🎯 Calibrated Podium (Top 3) & Points (Top 10) Probability Classifier — {selected_year} {selected_event_name}")
        r_preds, clf_metrics, _, r_notes = cached_models_2_3_5_race(selected_year, selected_round)
        render_storytelling_notes(r_notes, "Supervised Classification & Platt Sigmoid Probability Calibration Guide")

        if not r_preds.empty:
            fig_prob = px.bar(
                r_preds.head(15).sort_values('Podium_Prob_Pct', ascending=True),
                x=['Podium_Prob_Pct', 'Points_Prob_Pct'], y='Driver', orientation='h', barmode='group',
                title="Calibrated Pre-Race Probabilities (%): Podium Finish (Top 3) vs. Points Finish (Top 10)",
                template="plotly_dark", height=500
            )
            st.plotly_chart(fig_prob, use_container_width=True)
            render_graph_summary(
                "Displays calibrated pre-race probabilities (%) generated via 3-fold Platt Sigmoid scaling on Gradient Boosted classifiers.",
                "Allows strategists to assess exact podium and points likelihoods rather than relying on uncalibrated binary predictions."
            )

            st.markdown("#### 📊 Classification Calibration Metrics Table (ROC-AUC & Brier Score Loss)")
            st.dataframe(clf_metrics, use_container_width=True, hide_index=True)
            render_table_summary(
                "Evaluates probability quality using ROC-AUC (ranking discrimination, where 1.0 is perfect) and Brier Score Loss (mean squared probability error, where 0.0 is perfect).",
                "Low Brier Score Loss confirms that a 70% predicted probability corresponds to a ~70% real-world hit rate."
            )

            st.markdown("#### 📋 Complete Driver Podium & Points Probability Table")
            m3_table = r_preds[[
                'Driver', 'OfficialTeamName', 'Podium_Prob_Pct', 'Points_Prob_Pct',
                'PredictedPosition', 'FinishPosition', 'CleanGrid', 'QualiBestDelta_Pct'
            ]].rename(columns={
                'OfficialTeamName': 'Constructor', 'Podium_Prob_Pct': 'Podium Prob (%)',
                'Points_Prob_Pct': 'Points Prob (%)', 'PredictedPosition': 'AI Pred Rank',
                'FinishPosition': 'Actual Finish', 'CleanGrid': 'Start Grid', 'QualiBestDelta_Pct': 'Quali Gap (%)'
            })
            st.dataframe(m3_table, use_container_width=True, hide_index=True, height=460)
            render_table_summary(
                "Lists all drivers with their calibrated Podium Probability (%) and Top-10 Points Probability (%) prioritized on the left.",
                "Compare pre-race probabilities against Actual Finish to spot high-value underdog podiums or unexpected top-team retirements."
            )

    # --------------------------------------------------------------------------
    # AI MODULE 4: THERMAL TYRE DEGRADATION SURFACE (DEEP LEARNING MLP)
    # --------------------------------------------------------------------------
    elif selected_module == "🧠 Thermal Tyre Degradation Surface (Deep Learning MLP Neural Network)":
        st.subheader("🧠 Deep Learning Multi-Layer Perceptron (MLP 64→32 ReLU) Thermal Tyre Degradation Model")
        nn_surface, nn_notes = cached_model4_nn(selected_year)
        render_storytelling_notes(nn_notes, "Feedforward Neural Network (MLPRegressor) with ReLU & Adam Optimization Guide")

        if not nn_surface.empty:
            fig_nn = px.line(
                nn_surface, x='TrackTemp_C', y='NN_Predicted_DegSlope_SecPerLap', color='Compound',
                markers=True, color_discrete_map=TYRE_COLORS,
                title="MLP Neural Network Learned Thermal Response Surface: Predicted Tyre Degradation (s/lap) vs. Track Temperature (°C)",
                template="plotly_dark", height=480
            )
            st.plotly_chart(fig_nn, use_container_width=True)
            render_graph_summary(
                "Visualizes the non-linear thermal wear curve learned by our 2-hidden-layer (64 -> 32 neuron) Feedforward Neural Network across 22°C–46°C.",
                "Soft tyres exhibit steep thermal degradation above 38°C, whereas negative Intermediate slopes capture drying-track acceleration."
            )

            st.markdown("#### 🌡️ Neural Network Predicted Thermal Degradation Lookup Matrix (s/lap)")
            piv_nn = nn_surface.pivot(index='TrackTemp_C', columns='Compound', values='NN_Predicted_DegSlope_SecPerLap').reset_index()
            piv_nn = piv_nn.rename(columns={'TrackTemp_C': 'Track Temperature (°C)'})
            st.dataframe(piv_nn, use_container_width=True, hide_index=True)
            render_table_summary(
                "Provides the exact neural-network-predicted tyre wear rate (seconds lost per lap of tyre age) for each Pirelli compound across 4 thermal regimes.",
                "Race engineers use this matrix to adjust pre-race pit stop windows when Sunday track temperatures spike or drop."
            )

    # --------------------------------------------------------------------------
    # AI MODULE 5: EXPLAINABLE AI & FEATURE ATTRIBUTION (XAI)
    # --------------------------------------------------------------------------
    elif selected_module == "🔎 Explainable AI & Feature Attribution (Permutation & Gain XAI)":
        st.subheader(f"🔎 Explainable AI (XAI): Global Feature Importance & Local Driver Attribution — {selected_year} {selected_event_name}")
        r_preds, _, xai_df, r_notes = cached_models_2_3_5_race(selected_year, selected_round)
        render_storytelling_notes(r_notes, "Explainable AI (XAI): Permutation Feature Importance & Local Attribution Guide")

        if not xai_df.empty:
            fig_xai = px.bar(
                xai_df.sort_values('XGB_Gain_Importance', ascending=True),
                x=['XGB_Gain_Importance', 'Permutation_Importance'], y='Feature',
                orientation='h', barmode='group',
                title="Explainable AI (XAI): XGBoost Internal Split Gain vs. Out-of-Sample Permutation Feature Importance",
                template="plotly_dark", height=480
            )
            st.plotly_chart(fig_xai, use_container_width=True)
            render_graph_summary(
                "Opens the AI 'black box' by comparing internal gradient-boosted tree split gain against out-of-sample Permutation Importance.",
                "Permutation Importance randomly shuffles one feature at a time on the test race to measure how much prediction error increases."
            )

            st.markdown("#### 📊 Global Feature Importance Rankings Table")
            st.dataframe(xai_df, use_container_width=True, hide_index=True)
            render_table_summary(
                "Ranks all 11 pre-race model inputs by their relative contribution to race outcome predictions.",
                "Confirms that Starting Grid Position, Qualifying Pace Delta, and Rolling Constructor Form are the primary drivers of Sunday finishing order."
            )

            st.markdown("#### 📋 Local Per-Driver Feature Attribution Table (*'Why Did the AI Rank Each Driver Here?'*)")
            xai_local_table = r_preds[[
                'PredictedPosition', 'FinishPosition', 'Driver', 'OfficialTeamName',
                'XAI_KeyDrivers', 'ExpectedFinishScore', 'CleanGrid', 'TeamRollingFinish_5R', 'DriverRollingFinish_5R'
            ]].rename(columns={
                'PredictedPosition': 'AI Rank', 'FinishPosition': 'Actual Finish',
                'OfficialTeamName': 'Constructor', 'XAI_KeyDrivers': 'Primary Local XAI Attribution (Position Impact)',
                'ExpectedFinishScore': 'Expected Finish Score', 'CleanGrid': 'Start Grid',
                'TeamRollingFinish_5R': 'Team 5R Avg Finish', 'DriverRollingFinish_5R': 'Driver 5R Avg Finish'
            })
            st.dataframe(xai_local_table, use_container_width=True, hide_index=True, height=460)
            render_table_summary(
                "Explains each individual driver's prediction by quantifying how many positions their Starting Grid and Team Form pushed them up (-) or down (+).",
                "Provides complete transparency for stakeholders asking why a specific driver was projected to finish in the top 5 or midfield."
            )

    # --------------------------------------------------------------------------
    # AI MODULE 6: DRIVER STYLE & PERFORMANCE ARCHETYPES (PCA + K-MEANS)
    # --------------------------------------------------------------------------
    elif selected_module == "🧩 Driver Style & Performance Archetypes (PCA + K-Means Clustering)":
        st.subheader(f"🧩 Unsupervised Learning: PCA (2D Projection) + K-Means Driver Archetypes ({selected_year} Season)")
        clusters_df, c_notes = cached_model6_pca(selected_year)
        render_storytelling_notes(c_notes, "Unsupervised Z-Score Standardization, PCA Dimensionality Reduction & K-Means Guide")

        if not clusters_df.empty:
            fig_pca = px.scatter(
                clusters_df, x='PCA_1', y='PCA_2', color='Driver_Archetype', text='Driver',
                size='Overall_DNA_Rating', hover_data=['Team', 'Quali_Pace_Score', 'Race_Pace_Score', 'Tyre_Mgmt_Score'],
                title=f"{selected_year} Driver Performance Archetypes in 2D Principal Component Space (PCA_1 vs. PCA_2)",
                template="plotly_dark", height=520
            )
            fig_pca.update_traces(textposition='top center', marker=dict(size=16))
            st.plotly_chart(fig_pca, use_container_width=True)
            render_graph_summary(
                "Projects 6-dimensional driver performance metrics onto 2 orthogonal Principal Components (PCA_1 & PCA_2) and colors by K-Means cluster.",
                "Drivers plotted close together share similar performance DNA (e.g., Saturday Quali specialists vs. Sunday Tyre Whisperers) without human labels."
            )

            st.markdown("#### 📋 Unsupervised Driver Archetype & Principal Component Coordinates Table")
            pca_table = clusters_df[[
                'Driver', 'Team', 'Driver_Archetype', 'Overall_DNA_Rating', 'PCA_1', 'PCA_2',
                'Quali_Pace_Score', 'Race_Pace_Score', 'Tyre_Mgmt_Score', 'Overtake_Craft_Score', 'Consistency_Score'
            ]].rename(columns={
                'Driver_Archetype': 'Discovered K-Means Archetype', 'Overall_DNA_Rating': 'Overall DNA (0-100)',
                'Quali_Pace_Score': 'Quali Score', 'Race_Pace_Score': 'Race Pace Score',
                'Tyre_Mgmt_Score': 'Tyre Mgmt Score', 'Overtake_Craft_Score': 'Overtake Score',
                'Consistency_Score': 'Consistency Score'
            })
            st.dataframe(pca_table, use_container_width=True, hide_index=True, height=460)
            render_table_summary(
                "Lists every driver's automatically assigned K-Means cluster archetype alongside their PCA_1 and PCA_2 coordinates.",
                "Silhouette Score optimization automatically selects the optimal number of clusters K based on intra-cluster cohesion and inter-cluster separation."
            )

    # --------------------------------------------------------------------------
    # AI MODULE 7: GREEN-FLAG LAP ANOMALY DETECTION (ISOLATION FOREST)
    # --------------------------------------------------------------------------
    elif selected_module == "🚨 Green-Flag Lap Anomaly Detection (Unsupervised Isolation Forest)":
        st.subheader(f"🚨 Unsupervised Isolation Forest: On-Track Green-Flag Lap Anomaly Detection — {selected_year} {selected_event_name}")
        anom_df, a_notes = cached_model7_anomalies(selected_year, selected_round, selected_session)
        render_storytelling_notes(a_notes, "Multivariate Tree-Based Outlier Detection (Isolation Forest) Guide")

        if not anom_df.empty:
            fig_anom = px.scatter(
                anom_df.head(30), x='LapNumber', y='LapTime_Sec_Dev', color='Worst_Sector',
                size='Anomaly_Score', text='Driver', hover_data=['Compound', 'LapTime_Sec', 'Sector1_Sec_Dev', 'Sector2_Sec_Dev', 'Sector3_Sec_Dev'],
                title="Isolated Green-Flag Lap Anomalies Across Session Laps (Colored by Primary Sector Where Time Was Lost)",
                template="plotly_dark", height=500
            )
            fig_anom.update_traces(textposition='top center')
            st.plotly_chart(fig_anom, use_container_width=True)
            render_graph_summary(
                "Pinpoints green-flag racing laps where a driver's sector times deviated abnormally from their own compound median baseline.",
                "Vertical clusters of anomalies on specific laps reveal damp track patches, traffic bottlenecks, brake lockups, or car damage."
            )

            st.markdown("#### 📋 Isolated Green-Flag Lap Anomalies Table (Top 25 Multivariate Outliers)")
            anom_table = anom_df[[
                'Driver', 'Team', 'LapNumber', 'Worst_Sector', 'LapTime_Sec_Dev', 'Anomaly_Score',
                'Compound', 'TyreLife', 'LapTime_Sec', 'Sector1_Sec_Dev', 'Sector2_Sec_Dev', 'Sector3_Sec_Dev'
            ]].rename(columns={
                'LapNumber': 'Lap #', 'Worst_Sector': 'Primary Loss Sector',
                'LapTime_Sec_Dev': 'Lap Time Loss (+s)', 'Anomaly_Score': 'Isolation Score',
                'TyreLife': 'Tyre Age', 'LapTime_Sec': 'Actual Lap (s)',
                'Sector1_Sec_Dev': 'Δ S1 (s)', 'Sector2_Sec_Dev': 'Δ S2 (s)', 'Sector3_Sec_Dev': 'Δ S3 (s)'
            })
            st.dataframe(anom_table.head(25), use_container_width=True, hide_index=True, height=460)
            render_table_summary(
                "Orders anomalous green-flag laps from largest time loss (+s) to smallest, showing the exact sector (S1, S2, or S3) where the incident occurred.",
                "Excludes Safety Car, VSC, and Pit In/Out laps so only genuine on-track incidents, lockups, or transition struggles are flagged."
            )

    # --------------------------------------------------------------------------
    # AI MODULE 8: WEEKEND TRACK EVOLUTION & GRIP FORECASTING (TIME-SERIES)
    # --------------------------------------------------------------------------
    elif selected_module == "📉 Weekend Track Evolution & Grip Forecasting (Time-Series Trend)":
        st.subheader(f"📉 Time-Series Weekend Track Evolution & Grip Gain — {selected_year} {selected_event_name}")
        evo_df, e_notes = cached_model8_evo(selected_year, selected_round)
        render_storytelling_notes(e_notes, "Chronological Session-over-Session Time-Series Trend Decomposition Guide")

        if not evo_df.empty:
            fig_evo = px.line(
                evo_df, x='SessionType', y=['BestLap_Sec', 'Top10PctLap_Sec'], markers=True,
                title="Progressive Weekend Track Evolution: Session Best Lap (s) vs. Top-10% Field Benchmark Across Sessions",
                template="plotly_dark", height=460
            )
            st.plotly_chart(fig_evo, use_container_width=True)
            render_graph_summary(
                "Tracks how lap times evolve chronologically from Friday Free Practice (FP1) through Qualifying (Q) and Sunday's Race (R).",
                "Downward slopes quantify track rubbering-in ('Cumulative Grip Gain'); upward jumps indicate wet sessions or heavy Sunday race fuel loads."
            )

            st.markdown("#### 📋 Chronological Session Track Evolution & Weather Regime Table")
            evo_table = evo_df.drop(columns=['ChronologicalOrder']).rename(columns={
                'SessionType': 'Session', 'BestLap_Sec': 'Fastest Lap (s)',
                'Top10PctLap_Sec': 'Top-10% Benchmark (s)', 'CumulativeGripGain_Sec': 'Cumulative Grip Gain vs FP1 (+s)',
                'TrackTemp_C': 'Avg Track Temp (°C)', 'IsWet': 'Wet Session?'
            })
            st.dataframe(evo_table, use_container_width=True, hide_index=True)
            render_table_summary(
                "Summarizes the fastest lap, top-10% field pace, track temperature, and cumulative grip improvement (in seconds) across every session of the weekend.",
                "Engineers use the dryFP1-to-Q3 progression slope to calculate what lap time will be required to make the Q3 shootout."
            )

    # --------------------------------------------------------------------------
    # AI MODULE 9: PURE DRIVER SKILL VS. CONSTRUCTOR CAR POWER (BAYESIAN ELO)
    # --------------------------------------------------------------------------
    elif selected_module == "⚖️ Pure Driver Skill vs. Constructor Car Power (Bayesian Elo Rating)":
        st.subheader(f"⚖️ Sequential Bayesian TrueSkill / Elo Rating: Separating Pure Driver Skill vs. Car Power (Up to {selected_year})")
        bayes_df, b_notes = cached_model9_bayes(selected_year)
        render_storytelling_notes(b_notes, "Sequential Bayesian Elo Updating & Confounder Separation Guide")

        if not bayes_df.empty:
            fig_bayes = px.scatter(
                bayes_df, x='ConstructorCar_Elo', y='PureDriverSkill_Elo', color='Team', text='Driver',
                size='Combined_WinPower_Index', color_discrete_map=TEAM_COLORS,
                hover_data=['DriverSkill_Rank', 'SkillUncertainty_Sigma'],
                title="Bayesian Decoupling Quadrant: Pure Driver Skill Elo (Y-Axis) vs. Constructor Car Power Elo (X-Axis)",
                template="plotly_dark", height=520
            )
            fig_bayes.update_traces(textposition='top center', marker=dict(size=16))
            st.plotly_chart(fig_bayes, use_container_width=True)
            render_graph_summary(
                "Decouples Pure Driver Skill (Y-axis, learned from Teammate Head-to-Head dominance and grid-to-flag craft) from Constructor Car Performance (X-axis).",
                "Drivers in the Top-Left quadrant are elite talents outperforming midfield machinery; Top-Right drivers pair elite skill with a front-running car."
            )

            st.markdown("#### 📋 Bayesian Pure Driver Skill vs. Constructor Car Elo Rankings Table")
            bayes_table = bayes_df[[
                'DriverSkill_Rank', 'Driver', 'Team', 'PureDriverSkill_Elo',
                'ConstructorCar_Elo', 'Combined_WinPower_Index', 'SkillUncertainty_Sigma'
            ]].rename(columns={
                'DriverSkill_Rank': 'Pure Skill Rank', 'PureDriverSkill_Elo': 'Pure Driver Skill (Elo)',
                'ConstructorCar_Elo': 'Constructor Car (Elo)', 'Combined_WinPower_Index': 'Combined Win Power Index',
                'SkillUncertainty_Sigma': 'Posterior Uncertainty (±σ)'
            })
            st.dataframe(bayes_table, use_container_width=True, hide_index=True, height=460)
            render_table_summary(
                "Ranks all active drivers by Pure Driver Skill Elo (column 4), independent of how fast or slow their team's car is.",
                "Compare 'Pure Driver Skill (Elo)' against 'Constructor Car (Elo)' to see which drivers extract the most performance above their car's baseline."
            )

    # --------------------------------------------------------------------------
    # AI MODULE 10: UPCOMING RACE & PIT STRATEGY SIMULATOR (MONTE CARLO ENGINE)
    # --------------------------------------------------------------------------
    elif selected_module == "🎲 Upcoming Race & Pit Strategy Simulator (Monte Carlo Engine)":
        st.subheader("🔮 Monte Carlo Stochastic Race Simulator (Upcoming Weekends & Counterfactual Scenarios)")
        mc_c1, mc_c2, mc_c3, mc_c4 = st.columns([2, 1, 1, 1])
        upcoming_target = mc_c1.selectbox(
            "Select Upcoming / Future Grand Prix to Forecast",
            [
                "2026 Las Vegas Grand Prix (Upcoming Race)",
                "2026 Qatar Grand Prix",
                "2026 Abu Dhabi Grand Prix (Season Finale)",
                f"{selected_year} {selected_event_name} (Selected Event Counterfactual)",
                "2027 Australian Grand Prix (2027 Season Opener Preview)"
            ]
        )
        rain_prob = mc_c2.slider("🌧️ Rain Probability (%)", 0, 100, 15, 5) / 100.0
        sc_prob = mc_c3.slider("🚩 Safety Car Probability (%)", 0, 100, 65, 5) / 100.0
        n_sims = mc_c4.selectbox("🎲 Simulation Runs", [500, 1000, 2500], index=1)

        mc_df, mc_notes = cached_model10_mc(selected_year, upcoming_target, n_sims, rain_prob, sc_prob)
        render_storytelling_notes(mc_notes, "Monte Carlo Stochastic Race Simulation & Pit Strategy Guide")

        # CHART 1: Monte Carlo Probabilities Bar Chart (Full Width)
        fig_mc = px.bar(
            mc_df.head(14).sort_values('Podium_Prob_Pct', ascending=True),
            x=['Win_Prob_Pct', 'Podium_Prob_Pct', 'Points_Prob_Pct'], y='Driver',
            orientation='h', barmode='group',
            title=f"Monte Carlo Forecast ({n_sims:,} Simulated Races) — Win %, Podium % & Top-10 Points % for {upcoming_target}",
            template="plotly_dark", height=500
        )
        st.plotly_chart(fig_mc, use_container_width=True)
        render_graph_summary(
            f"Displays win, podium, and points probabilities across {n_sims:,} stochastic race simulations incorporating {rain_prob*100:.0f}% rain and {sc_prob*100:.0f}% Safety Car probability.",
            "Higher Safety Car and Rain sliders compress field gaps and boost podium probabilities for skilled wet-weather and midfield drivers."
        )

        # TABLE 1: Monte Carlo Full Grid Table (Full Width + Table Summary)
        st.markdown(f"#### 📋 Monte Carlo Probability Distribution & Confidence Intervals Table — {upcoming_target}")
        mc_table = mc_df[[
            'Predicted_Rank', 'Driver', 'Team', 'Win_Prob_Pct', 'Podium_Prob_Pct',
            'Points_Prob_Pct', 'Expected_Finish_Mean', 'Median_Finish_P50', 'Ceiling_P10', 'Floor_P90', 'Simulated_DNF_Risk_Pct'
        ]].rename(columns={
            'Predicted_Rank': 'Sim Rank', 'Win_Prob_Pct': 'Win Prob (%)',
            'Podium_Prob_Pct': 'Podium Prob (%)', 'Points_Prob_Pct': 'Points Prob (%)',
            'Expected_Finish_Mean': 'Expected Finish (Mean)', 'Median_Finish_P50': 'Median Finish (P50)',
            'Ceiling_P10': 'Best-Case Ceiling (P10)', 'Floor_P90': 'Worst-Case Floor (P90)',
            'Simulated_DNF_Risk_Pct': 'DNF Risk (%)'
        })
        st.dataframe(mc_table, use_container_width=True, hide_index=True, height=460)
        render_table_summary(
            f"Complete {len(mc_table)}-driver probabilistic forecast ordered by Win % and Podium %, including 80% Confidence Intervals [P10 Ceiling to P90 Floor].",
            "A driver's 'Best-Case Ceiling (P10)' shows their upside in a chaotic race, while 'Worst-Case Floor (P90)' captures downside traffic/attrition risk."
        )

        st.divider()
        # CHART 2 & TABLE 2: Interactive Pit-Wall Strategy Simulator (1-Stop vs 2-Stop)
        st.subheader("🛠️ Interactive Pit-Wall Strategy Crossover Simulator (1-Stop vs. 2-Stop)")
        s_c1, s_c2, s_c3, s_c4 = st.columns(4)
        total_sim_laps = s_c1.slider("Race Distance (Laps)", 40, 78, int(laps_df['TotalSessionLaps'].max() or 60))
        pit_loss = s_c2.slider("Pit Lane Time Loss (Sec)", 16.0, 30.0, 22.0, 0.5)
        soft_deg = s_c3.number_input("SOFT Deg Rate (s/lap)", 0.01, 0.50, 0.12, 0.01)
        med_deg = s_c4.number_input("MEDIUM Deg Rate (s/lap)", 0.01, 0.40, 0.07, 0.01)

        sim_df = simulate_pit_strategies(
            total_laps=total_sim_laps,
            base_pace_sec=float(laps_df['SessionBestLap_Sec'].iloc[0] or 75.0),
            deg_rates={"SOFT": soft_deg, "MEDIUM": med_deg, "HARD": max(0.02, med_deg * 0.65)},
            compound_offsets={"SOFT": 0.0, "MEDIUM": 0.45, "HARD": 0.95},
            pit_loss_sec=pit_loss
        )
        final_totals = sim_df[sim_df['LapNumber'] == total_sim_laps].sort_values('CumulativeTime_Sec')
        best_total_time = final_totals.iloc[0]['CumulativeTime_Sec']
        ref_lap_pace = best_total_time / total_sim_laps
        sim_df['GapToVirtualWinner_Sec'] = np.round(sim_df['CumulativeTime_Sec'] - (sim_df['LapNumber'] * ref_lap_pace), 3)

        fig_sim = px.line(
            sim_df, x='LapNumber', y='GapToVirtualWinner_Sec', color='Strategy',
            title="Pit Strategy Simulation Trace vs. Virtual Constant-Pace Winner (Higher at Checkered Flag = Winning Strategy)",
            template="plotly_dark", height=460
        )
        fig_sim.update_yaxes(autorange="reversed", title_text="Gap to Virtual Reference Car (Sec) — Higher = Ahead on Track")
        st.plotly_chart(fig_sim, use_container_width=True)
        render_graph_summary(
            "Simulates 1-Stop vs. 2-Stop tyre strategies against a constant-speed Virtual Reference Car across the full race distance.",
            "Vertical drops represent pit-lane time losses; steep upward recovery slopes show fresh tyres catching and passing older degraded tyres."
        )

        strat_sum_table = final_totals[['Strategy', 'CumulativeTime_Sec', 'GapToFastestStrategy_Sec']].rename(columns={
            'Strategy': 'Simulated Pit Strategy Profile',
            'CumulativeTime_Sec': 'Total Simulated Race Time (s)',
            'GapToFastestStrategy_Sec': 'Time Deficit Behind Winning Strategy (+s)'
        })
        st.dataframe(strat_sum_table, use_container_width=True, hide_index=True)
        render_table_summary(
            "Ranks the 4 candidate pit strategies by Total Simulated Race Time (s) at the checkered flag.",
            "Adjust the Pit Lane Time Loss and Tyre Degradation sliders above to see the exact point where a 2-Stop overtakes a 1-Stop strategy."
        )