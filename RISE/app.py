import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats
import warnings
import io
import zipfile

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ForecastIQ · Sales Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  [data-testid="stSidebar"] { background: #0d0d0d; border-right: 1px solid #1f1f1f; }
  [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
  [data-testid="stSidebar"] .stSlider > label,
  [data-testid="stSidebar"] .stSelectbox > label,
  [data-testid="stSidebar"] .stMultiSelect > label,
  [data-testid="stSidebar"] .stCheckbox > label {
    color: #a0a0a0 !important; font-size: 0.78rem;
    letter-spacing: 0.08em; text-transform: uppercase;
  }

  .main .block-container { padding: 2rem 2.5rem 4rem; max-width: 1400px; }

  .hero-header {
    background: linear-gradient(135deg, #0d0d0d 0%, #141414 50%, #0d1117 100%);
    border: 1px solid #1f2937; border-radius: 16px;
    padding: 2.5rem 3rem; margin-bottom: 2rem;
    position: relative; overflow: hidden;
  }
  .hero-header::before {
    content: ''; position: absolute; top: -60px; right: -60px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
    border-radius: 50%;
  }
  .hero-title { font-family: 'Syne', sans-serif; font-size: 2.4rem; font-weight: 800; color: #f9fafb; letter-spacing: -0.02em; margin: 0; }
  .hero-subtitle { font-size: 0.95rem; color: #6b7280; margin-top: 0.4rem; font-weight: 300; }
  .hero-badge {
    display: inline-block; background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.3); color: #818cf8;
    font-family: 'DM Mono', monospace; font-size: 0.7rem;
    padding: 3px 10px; border-radius: 99px;
    letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 1rem;
  }

  .kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 2rem; }
  .kpi-card { background: #111111; border: 1px solid #1f1f1f; border-radius: 12px; padding: 1.25rem 1.5rem; position: relative; overflow: hidden; }
  .kpi-card::after { content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 2px; }
  .kpi-card.indigo::after { background: linear-gradient(90deg, #6366f1, #818cf8); }
  .kpi-card.emerald::after { background: linear-gradient(90deg, #10b981, #34d399); }
  .kpi-card.amber::after { background: linear-gradient(90deg, #f59e0b, #fcd34d); }
  .kpi-card.rose::after { background: linear-gradient(90deg, #f43f5e, #fb7185); }
  .kpi-label { font-size: 0.72rem; color: #6b7280; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.5rem; font-family: 'DM Mono', monospace; }
  .kpi-value { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 700; color: #f9fafb; }
  .kpi-delta { font-size: 0.78rem; margin-top: 0.3rem; }
  .kpi-delta.up { color: #10b981; }
  .kpi-delta.down { color: #f43f5e; }
  .kpi-delta.neutral { color: #6b7280; }

  .section-header {
    display: flex; align-items: center; gap: 10px;
    font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700; color: #f9fafb;
    margin: 2rem 0 1rem; padding-bottom: 0.6rem; border-bottom: 1px solid #1f1f1f;
  }
  .section-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }

  .metric-chip {
    display: inline-flex; align-items: center; gap: 6px;
    background: #111; border: 1px solid #1f1f1f; border-radius: 8px;
    padding: 6px 12px; font-family: 'DM Mono', monospace;
    font-size: 0.78rem; color: #a0a0a0; margin: 4px;
  }
  .metric-chip span { color: #f9fafb; font-weight: 500; }

  .stTabs [data-baseweb="tab-list"] { background: #111; border-radius: 10px; padding: 4px; gap: 4px; border: 1px solid #1f1f1f; }
  .stTabs [data-baseweb="tab"] { border-radius: 7px; font-size: 0.82rem; font-family: 'DM Mono', monospace; letter-spacing: 0.05em; color: #6b7280; background: transparent; }
  .stTabs [aria-selected="true"] { background: #1f1f1f !important; color: #f9fafb !important; }
  .stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem; }

  [data-testid="stFileUploader"] { border: 1.5px dashed #1f2937; border-radius: 12px; padding: 1rem; }
  hr { border-color: #1f1f1f; }

  .info-box { background: rgba(99,102,241,0.07); border: 1px solid rgba(99,102,241,0.2); border-radius: 10px; padding: 1rem 1.25rem; font-size: 0.85rem; color: #a5b4fc; margin-bottom: 1rem; }
  .warn-box { background: rgba(245,158,11,0.07); border: 1px solid rgba(245,158,11,0.2); border-radius: 10px; padding: 1rem 1.25rem; font-size: 0.85rem; color: #fcd34d; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <div class="hero-badge">v2.0 · Prophet-Powered</div>
  <div class="hero-title">ForecastIQ</div>
  <div class="hero-subtitle">Industrial-grade sales intelligence &amp; time-series forecasting platform</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model Configuration")
    st.divider()
    st.markdown("**Forecast Horizon**")
    periods = st.slider("Days to forecast", 7, 365, 30)
    st.markdown("**Seasonality**")
    yearly_seasonality = st.checkbox("Yearly seasonality", value=True)
    weekly_seasonality = st.checkbox("Weekly seasonality", value=True)
    daily_seasonality  = st.checkbox("Daily seasonality",  value=False)
    st.markdown("**Holidays**")
    country_code = st.selectbox("Holiday country", ["None","US","GB","IN","DE","FR","AU","CA","SG"], index=0)
    st.markdown("**Confidence Interval**")
    interval_width = st.slider("Interval width", 0.5, 0.99, 0.80, step=0.01)
    st.markdown("**Changepoint Detection**")
    changepoint_prior_scale = st.slider("Changepoint flexibility", 0.001, 0.5, 0.05, step=0.001, format="%.3f")
    seasonality_prior_scale = st.slider("Seasonality strength", 0.01, 20.0, 10.0, step=0.01)
    st.markdown("**Anomaly Detection**")
    anomaly_threshold = st.slider("Anomaly z-score threshold", 1.5, 4.0, 2.5, step=0.1)
    st.markdown("**Cross-Validation**")
    run_cv = st.checkbox("Run cross-validation", value=False)
    cv_horizon = "30 days"
    if run_cv:
        cv_horizon = st.text_input("CV horizon (e.g. 30 days)", "30 days")
    st.divider()
    st.caption("ForecastIQ uses Meta's Prophet library for robust decomposable time-series forecasting with Bayesian changepoint detection.")

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
PLOTLY_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,13,13,1)",
    font=dict(family="DM Sans, sans-serif", color="#9ca3af"),
    xaxis=dict(gridcolor="#1f2937", zeroline=False, showgrid=True),
    yaxis=dict(gridcolor="#1f2937", zeroline=False, showgrid=True),
    margin=dict(l=20, r=20, t=40, b=20),
)

def fmt_num(n):
    if abs(n) >= 1e6: return f"{n/1e6:.2f}M"
    if abs(n) >= 1e3: return f"{n/1e3:.1f}K"
    return f"{n:.1f}"

def detect_anomalies(df, threshold):
    df = df.copy()
    df["z_score"] = np.nan
    df["anomaly"] = False
    valid = df["y"].notna()
    if valid.sum() > 1:
        z = np.abs(stats.zscore(df.loc[valid, "y"]))
        df.loc[valid, "z_score"] = z
        df.loc[valid, "anomaly"] = z > threshold
    return df

# ─────────────────────────────────────────────
# DATA INGESTION
# ─────────────────────────────────────────────
st.markdown('<div class="section-header"><div class="section-dot" style="background:#6366f1"></div>Data Ingestion</div>', unsafe_allow_html=True)

upload_col, sample_col = st.columns([3, 1])
with upload_col:
    uploaded_file = st.file_uploader("Upload CSV with date + sales columns", type=["csv"], label_visibility="collapsed")
with sample_col:
    if st.button("📎 Load sample data", use_container_width=True):
        dates = pd.date_range("2021-01-01", periods=730, freq="D")
        np.random.seed(42)
        trend    = np.linspace(200, 500, 730)
        seasonal = 80 * np.sin(2 * np.pi * np.arange(730) / 365)
        weekly   = 30 * np.sin(2 * np.pi * np.arange(730) / 7)
        noise    = np.random.normal(0, 25, 730)
        y = trend + seasonal + weekly + noise
        y[100] *= 2.8
        sample_df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "sales": np.round(y, 2)})
        buf = io.StringIO()
        sample_df.to_csv(buf, index=False)
        st.download_button("⬇ Download sample CSV", data=buf.getvalue(), file_name="sample_sales.csv", mime="text/csv")

# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
if uploaded_file is not None:
    # Step 1: Read raw CSV
    try:
        raw_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    if raw_df.empty or len(raw_df.columns) < 2:
        st.markdown('<div class="warn-box">⚠ CSV must have at least 2 columns (date and sales).</div>', unsafe_allow_html=True)
        st.stop()

    # Step 2: Column selection
    col1, col2 = st.columns(2)
    with col1:
        date_col  = st.selectbox("Select date column",  raw_df.columns.tolist(), key="dc")
    with col2:
        sales_col = st.selectbox("Select sales column", raw_df.columns.tolist(), key="sc")

    # Step 3: Build df with ds/y — never rename, always construct fresh
    try:
        ds_series = pd.to_datetime(raw_df[date_col], errors="coerce")
        y_series  = pd.to_numeric(raw_df[sales_col], errors="coerce")
    except KeyError as e:
        st.error(f"Column not found: {e}. Please re-select columns above.")
        st.stop()

    df = pd.DataFrame({"ds": ds_series, "y": y_series})
    df.dropna(subset=["ds", "y"], inplace=True)
    df.sort_values("ds", inplace=True)
    df.reset_index(drop=True, inplace=True)

    if len(df) < 10:
        st.markdown('<div class="warn-box">⚠ Too few valid rows after parsing (need at least 10). Check date/sales column selection.</div>', unsafe_allow_html=True)
        st.stop()

    # Step 4: Anomaly detection
    df_anom = detect_anomalies(df, anomaly_threshold)
    n_anom  = int(df_anom["anomaly"].sum())

    # Step 5: KPIs
    half   = len(df) // 2
    prior  = df["y"].iloc[:half].sum()
    recent = df["y"].iloc[half:].sum()
    delta  = ((recent - prior) / prior * 100) if prior != 0 else 0
    delta_cls = "up" if delta >= 0 else "down"
    delta_sym = "▲" if delta >= 0 else "▼"

    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card indigo">
        <div class="kpi-label">Total Sales</div>
        <div class="kpi-value">{fmt_num(df["y"].sum())}</div>
        <div class="kpi-delta {delta_cls}">{delta_sym} {abs(delta):.1f}% vs prior period</div>
      </div>
      <div class="kpi-card emerald">
        <div class="kpi-label">Daily Average</div>
        <div class="kpi-value">{fmt_num(df["y"].mean())}</div>
        <div class="kpi-delta neutral">σ = {fmt_num(df["y"].std())}</div>
      </div>
      <div class="kpi-card amber">
        <div class="kpi-label">Data Points</div>
        <div class="kpi-value">{len(df):,}</div>
        <div class="kpi-delta neutral">observations loaded</div>
      </div>
      <div class="kpi-card rose">
        <div class="kpi-label">Anomalies</div>
        <div class="kpi-value">{n_anom}</div>
        <div class="kpi-delta {'down' if n_anom > 0 else 'up'}>{'⚠ detected' if n_anom > 0 else '✓ clean data'}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────
    # TRAIN MODEL (once, outside tabs)
    # ─────────────────────────────────────────
    with st.spinner("Training Prophet model…"):
        try:
            model = Prophet(
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=daily_seasonality,
                interval_width=interval_width,
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
            )
            if country_code != "None":
                model.add_country_holidays(country_name=country_code)
            model.fit(df[["ds", "y"]])
            future   = model.make_future_dataframe(periods=periods, freq="D")
            forecast = model.predict(future)
        except Exception as e:
            st.error(f"Model training failed: {e}")
            st.stop()

    # ─────────────────────────────────────────
    # METRICS (computed once, shared across tabs)
    # ─────────────────────────────────────────
    merged = pd.merge(df, forecast[["ds", "yhat"]], on="ds", how="inner")
    mae_v  = mean_absolute_error(merged["y"], merged["yhat"])
    rmse_v = np.sqrt(mean_squared_error(merged["y"], merged["yhat"]))
    mape_v = np.mean(np.abs((merged["y"] - merged["yhat"]) / merged["y"].replace(0, np.nan))) * 100
    ss_res = np.sum((merged["y"] - merged["yhat"]) ** 2)
    ss_tot = np.sum((merged["y"] - merged["y"].mean()) ** 2)
    r2_v   = max(0.0, 1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

    # ══════════════════════════════════════════
    # TABS
    # ══════════════════════════════════════════
    tabs = st.tabs(["📈 Historical", "🔮 Forecast", "📉 Components", "🚨 Anomalies", "📐 Metrics", "📦 Export"])

    # ── TAB 1: Historical ─────────────────────
    with tabs[0]:
        st.markdown('<div class="section-header"><div class="section-dot" style="background:#10b981"></div>Historical Sales</div>', unsafe_allow_html=True)

        roll_window = st.slider("Rolling average window (days)", 3, 90, 14, key="roll")
        df_plot = df.copy()
        df_plot["rolling"] = df_plot["y"].rolling(roll_window, center=True).mean()

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=df_plot["ds"], y=df_plot["y"], mode="lines", name="Actual",
            line=dict(color="#6366f1", width=1.2),
            fill="tozeroy", fillcolor="rgba(99,102,241,0.05)"
        ))
        fig_hist.add_trace(go.Scatter(
            x=df_plot["ds"], y=df_plot["rolling"], mode="lines",
            name=f"{roll_window}d MA", line=dict(color="#34d399", width=2, dash="dot")
        ))
        anom_pts = df_anom[df_anom["anomaly"]]
        if not anom_pts.empty:
            fig_hist.add_trace(go.Scatter(
                x=anom_pts["ds"], y=anom_pts["y"], mode="markers", name="Anomaly",
                marker=dict(color="#f43f5e", size=9, symbol="x", line=dict(width=2, color="#f43f5e"))
            ))
        fig_hist.update_layout(title="Sales Over Time", height=380, **PLOTLY_DARK,
                               legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.08))
        st.plotly_chart(fig_hist, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig_dist = px.histogram(df_plot, x="y", nbins=40, title="Sales Distribution",
                                    color_discrete_sequence=["#6366f1"])
            fig_dist.update_layout(height=300, **PLOTLY_DARK)
            st.plotly_chart(fig_dist, use_container_width=True)
        with c2:
            df_plot["month"] = df_plot["ds"].dt.month_name().str[:3]
            df_plot["year"]  = df_plot["ds"].dt.year
            pivot = df_plot.groupby(["year", "month"])["y"].mean().unstack()
            month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            pivot = pivot.reindex(columns=[m for m in month_order if m in pivot.columns])
            if not pivot.empty:
                fig_heat = px.imshow(pivot, color_continuous_scale="Viridis",
                                     title="Avg Sales Heatmap (Year × Month)", aspect="auto")
                fig_heat.update_layout(height=300, **PLOTLY_DARK)
                st.plotly_chart(fig_heat, use_container_width=True)

        with st.expander("🗃 Raw data preview"):
            st.dataframe(df.head(100), use_container_width=True)

    # ── TAB 2: Forecast ───────────────────────
    with tabs[1]:
        st.markdown('<div class="section-header"><div class="section-dot" style="background:#818cf8"></div>Prophet Forecast</div>', unsafe_allow_html=True)

        hist_fc = forecast[forecast["ds"] <= df["ds"].max()]
        pred_fc = forecast[forecast["ds"] >  df["ds"].max()]

        fig_fc = go.Figure()
        # CI band
        fig_fc.add_trace(go.Scatter(
            x=pd.concat([forecast["ds"], forecast["ds"][::-1]]),
            y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"][::-1]]),
            fill="toself", fillcolor="rgba(99,102,241,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            name=f"{int(interval_width*100)}% CI"
        ))
        # Historical fit
        fig_fc.add_trace(go.Scatter(
            x=hist_fc["ds"], y=hist_fc["yhat"], mode="lines", name="Fitted",
            line=dict(color="#6366f1", width=1.5, dash="dot")
        ))
        # Future forecast
        fig_fc.add_trace(go.Scatter(
            x=pred_fc["ds"], y=pred_fc["yhat"], mode="lines", name="Forecast",
            line=dict(color="#34d399", width=2.5)
        ))
        # Actuals
        fig_fc.add_trace(go.Scatter(
            x=df["ds"], y=df["y"], mode="lines", name="Actual",
            line=dict(color="#f9fafb", width=1), opacity=0.7
        ))
        # Changepoints
        for cp in model.changepoints:
            fig_fc.add_vline(x=cp, line_width=0.6, line_dash="dot", line_color="rgba(251,146,60,0.35)")
        fig_fc.update_layout(title="Sales Forecast", height=440, **PLOTLY_DARK,
                             legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.08))
        st.plotly_chart(fig_fc, use_container_width=True)

        # Summary chips
        last_30_avg  = df["y"].iloc[-30:].mean() if len(df) >= 30 else df["y"].mean()
        next_30      = pred_fc["yhat"].iloc[:30] if len(pred_fc) >= 30 else pred_fc["yhat"]
        next_30_avg  = next_30.mean() if len(next_30) > 0 else 0
        exp_chg      = ((next_30_avg - last_30_avg) / last_30_avg * 100) if last_30_avg != 0 else 0
        peak_date    = pred_fc.loc[pred_fc["yhat"].idxmax(), "ds"].strftime("%b %d, %Y") if len(pred_fc) > 0 else "N/A"
        chg_color    = "#10b981" if exp_chg >= 0 else "#f43f5e"
        chg_sym      = "▲" if exp_chg >= 0 else "▼"

        st.markdown(f"""
        <div style="display:flex;flex-wrap:wrap;gap:0;margin-top:1rem;">
          <div class="metric-chip">30d trailing avg <span>{fmt_num(last_30_avg)}</span></div>
          <div class="metric-chip">Next 30d avg <span>{fmt_num(next_30_avg)}</span></div>
          <div class="metric-chip">Expected change <span style="color:{chg_color}">{chg_sym}{abs(exp_chg):.1f}%</span></div>
          <div class="metric-chip">Peak date <span>{peak_date}</span></div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📋 Forecast table"):
            out = forecast[["ds","yhat","yhat_lower","yhat_upper"]].copy()
            out.columns = ["Date","Forecast","Lower Bound","Upper Bound"]
            out[["Forecast","Lower Bound","Upper Bound"]] = out[["Forecast","Lower Bound","Upper Bound"]].round(2)
            st.dataframe(out.tail(periods), use_container_width=True)

    # ── TAB 3: Components ─────────────────────
    with tabs[2]:
        st.markdown('<div class="section-header"><div class="section-dot" style="background:#f59e0b"></div>Decomposition</div>', unsafe_allow_html=True)

        comp_cols = [c for c in ["trend", "weekly", "yearly"] if c in forecast.columns]
        if comp_cols:
            fig_comp = make_subplots(rows=len(comp_cols), cols=1, shared_xaxes=True,
                                     vertical_spacing=0.05,
                                     subplot_titles=[c.title() for c in comp_cols])
            colors = ["#6366f1", "#34d399", "#f59e0b"]
            for i, comp in enumerate(comp_cols, 1):
                fig_comp.add_trace(go.Scatter(
                    x=forecast["ds"], y=forecast[comp], mode="lines",
                    name=comp.title(), line=dict(color=colors[i-1], width=2)
                ), row=i, col=1)
            fig_comp.update_layout(height=200 * len(comp_cols), showlegend=False, **PLOTLY_DARK)
            st.plotly_chart(fig_comp, use_container_width=True)

        # Changepoint magnitudes
        st.markdown('<div class="section-header"><div class="section-dot" style="background:#fb923c"></div>Changepoint Magnitudes</div>', unsafe_allow_html=True)
        try:
            cp_deltas = model.params["delta"].mean(axis=0)
            cp_df = pd.DataFrame({"date": model.changepoints.values, "magnitude": cp_deltas})
            fig_cp = px.bar(cp_df, x="date", y="magnitude",
                            color="magnitude", color_continuous_scale=["#f43f5e","#1f1f1f","#10b981"],
                            title="Trend Changepoint Magnitudes")
            fig_cp.update_layout(height=280, **PLOTLY_DARK, coloraxis_showscale=False)
            st.plotly_chart(fig_cp, use_container_width=True)
        except Exception:
            st.info("Changepoint magnitudes not available for this model configuration.")

    # ── TAB 4: Anomalies ─────────────────────
    with tabs[3]:
        st.markdown('<div class="section-header"><div class="section-dot" style="background:#f43f5e"></div>Anomaly Detection</div>', unsafe_allow_html=True)

        normal_pts = df_anom[~df_anom["anomaly"]]
        anom_pts2  = df_anom[ df_anom["anomaly"]]

        fig_az = go.Figure()
        fig_az.add_trace(go.Scatter(
            x=normal_pts["ds"], y=normal_pts["y"], mode="lines",
            name="Normal", line=dict(color="#6366f1", width=1.2)
        ))
        if not anom_pts2.empty:
            fig_az.add_trace(go.Scatter(
                x=anom_pts2["ds"], y=anom_pts2["y"], mode="markers", name="Anomaly",
                marker=dict(color="#f43f5e", size=11, symbol="diamond",
                            line=dict(color="#fff", width=1))
            ))
        fig_az.update_layout(title=f"Anomaly Scan (z-score > {anomaly_threshold})", height=360, **PLOTLY_DARK)
        st.plotly_chart(fig_az, use_container_width=True)

        if n_anom > 0:
            st.markdown(f'<div class="warn-box">⚠ {n_anom} anomalies detected. Consider reviewing these data points.</div>', unsafe_allow_html=True)
            anom_tbl = df_anom[df_anom["anomaly"]][["ds","y","z_score"]].copy()
            anom_tbl.columns = ["Date","Value","Z-Score"]
            anom_tbl["Z-Score"] = anom_tbl["Z-Score"].round(2)
            anom_tbl["Severity"] = anom_tbl["Z-Score"].apply(
                lambda z: "🔴 High" if z > anomaly_threshold * 1.5 else "🟡 Medium"
            )
            st.dataframe(anom_tbl.reset_index(drop=True), use_container_width=True)
        else:
            st.markdown('<div class="info-box">✓ No anomalies detected at the current threshold. Your data looks clean.</div>', unsafe_allow_html=True)

    # ── TAB 5: Metrics ────────────────────────
    with tabs[4]:
        st.markdown('<div class="section-header"><div class="section-dot" style="background:#34d399"></div>Model Performance Metrics</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="kpi-grid">
          <div class="kpi-card indigo">
            <div class="kpi-label">MAE</div>
            <div class="kpi-value">{fmt_num(mae_v)}</div>
            <div class="kpi-delta neutral">Mean Abs Error</div>
          </div>
          <div class="kpi-card emerald">
            <div class="kpi-label">RMSE</div>
            <div class="kpi-value">{fmt_num(rmse_v)}</div>
            <div class="kpi-delta neutral">Root Mean Sq Error</div>
          </div>
          <div class="kpi-card amber">
            <div class="kpi-label">MAPE</div>
            <div class="kpi-value">{mape_v:.1f}%</div>
            <div class="kpi-delta neutral">Mean Abs % Error</div>
          </div>
          <div class="kpi-card rose">
            <div class="kpi-label">R²</div>
            <div class="kpi-value">{r2_v:.3f}</div>
            <div class="kpi-delta neutral">Fit Quality</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        merged["residual"] = merged["y"] - merged["yhat"]
        fig_res = make_subplots(rows=1, cols=2,
                                subplot_titles=["Residuals Over Time", "Residual Distribution"])
        fig_res.add_trace(go.Scatter(
            x=merged["ds"], y=merged["residual"], mode="lines",
            line=dict(color="#818cf8", width=1.2), name="Residual"
        ), row=1, col=1)
        fig_res.add_hline(y=0, line_dash="dot", line_color="#6b7280", row=1, col=1)
        fig_res.add_trace(go.Histogram(
            x=merged["residual"], nbinsx=40, marker_color="#6366f1", name="Distribution"
        ), row=1, col=2)
        fig_res.update_layout(showlegend=False, height=320, **PLOTLY_DARK)
        st.plotly_chart(fig_res, use_container_width=True)

        if run_cv:
            st.markdown('<div class="section-header"><div class="section-dot" style="background:#f59e0b"></div>Cross-Validation</div>', unsafe_allow_html=True)
            with st.spinner("Running cross-validation (may take a minute)…"):
                try:
                    n_days  = len(df)
                    initial = f"{max(30, n_days // 2)} days"
                    period  = f"{max(7,  n_days // 10)} days"
                    df_cv   = cross_validation(model, initial=initial, period=period,
                                               horizon=cv_horizon, disable_tqdm=True)
                    df_pm   = performance_metrics(df_cv)
                    st.dataframe(df_pm.round(3), use_container_width=True)
                    fig_cv = px.line(df_pm, x="horizon", y=["mae","rmse","mape"],
                                     title="CV Metrics vs Horizon",
                                     color_discrete_map={"mae":"#6366f1","rmse":"#34d399","mape":"#f59e0b"})
                    fig_cv.update_layout(height=320, **PLOTLY_DARK)
                    st.plotly_chart(fig_cv, use_container_width=True)
                except Exception as e:
                    st.markdown(f'<div class="warn-box">⚠ Cross-validation failed: {e}</div>', unsafe_allow_html=True)

    # ── TAB 6: Export ─────────────────────────
    with tabs[5]:
        st.markdown('<div class="section-header"><div class="section-dot" style="background:#a78bfa"></div>Export & Download</div>', unsafe_allow_html=True)

        forecast_csv = forecast[["ds","yhat","yhat_lower","yhat_upper"]].to_csv(index=False)
        anomaly_csv  = df_anom.to_csv(index=False)
        metrics_csv  = pd.DataFrame([{"MAE": mae_v, "RMSE": rmse_v, "MAPE_pct": mape_v, "R2": r2_v}]).to_csv(index=False)

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr("forecast.csv",  forecast_csv)
            zf.writestr("anomalies.csv", anomaly_csv)
            zf.writestr("metrics.csv",   metrics_csv)
        zip_buf.seek(0)

        ec1, ec2, ec3, ec4 = st.columns(4)
        with ec1:
            st.download_button("📥 Forecast CSV",  data=forecast_csv, file_name="forecast.csv",  mime="text/csv", use_container_width=True)
        with ec2:
            st.download_button("🚨 Anomalies CSV", data=anomaly_csv,  file_name="anomalies.csv", mime="text/csv", use_container_width=True)
        with ec3:
            st.download_button("📐 Metrics CSV",   data=metrics_csv,  file_name="metrics.csv",   mime="text/csv", use_container_width=True)
        with ec4:
            st.download_button("📦 Download All (ZIP)", data=zip_buf,
                               file_name="forecastiq_export.zip", mime="application/zip", use_container_width=True)

        st.markdown('<div class="info-box">💡 Tip: Use the forecast CSV to feed downstream BI tools like Power BI, Tableau, or Looker.</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem;color:#4b5563;">
      <div style="font-size:3rem;margin-bottom:1rem;">📂</div>
      <div style="font-family:'Syne',sans-serif;font-size:1.2rem;color:#9ca3af;font-weight:600;">No data loaded yet</div>
      <div style="font-size:0.9rem;margin-top:0.5rem;">Upload a CSV above or use <strong style="color:#818cf8">Load sample data</strong> to get started.</div>
    </div>
    """, unsafe_allow_html=True)