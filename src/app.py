"""
app.py — Streamlit earnings surprise prediction dashboard.

Run:  streamlit run src/app.py

Features
────────
  • Ticker search bar — works for any S&P 500 company
  • Beat probability gauge (Plotly)
  • Top 5 drivers of the prediction (feature importance × feature value)
  • Company historical beat rate + beat/miss streak
  • Sector comparison (ticker vs sector average)
  • Last 8 quarters: actual EPS vs estimate + surprise %
  • Model transparency expandable section
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ── Allow running as `streamlit run src/app.py` from project root ──────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sentiment import get_sentiment_for_ticker  # noqa: E402

logging.basicConfig(level=logging.WARNING)

DATA_DIR   = ROOT / "data"
MODELS_DIR = ROOT / "models"
PLOTS_DIR  = ROOT / "plots"


# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Earnings Surprise Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    "<style>div[data-testid='metric-container']{background:#f7f9fc;"
    "border-radius:8px;padding:8px 12px;}</style>",
    unsafe_allow_html=True,
)


# ─── Cached resource loaders ──────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    path = MODELS_DIR / "earnings_model.pkl"
    if not path.exists():
        return None, None, {}
    art = joblib.load(path)
    medians = art.get("train_medians", {})
    return art["model"], art["feature_names"], medians


@st.cache_data(ttl=3600, show_spinner=False)
def load_features_data() -> pd.DataFrame:
    p = DATA_DIR / "features.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"], errors="coerce")
    return df


@st.cache_data(ttl=3600, show_spinner="Fetching market data…")
def fetch_ticker_data(ticker: str) -> dict:
    """Pull live yfinance data for *ticker*."""
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info

        raw = stock.earnings_dates
        earnings = None
        if raw is not None and not raw.empty:
            earnings = raw.copy().reset_index()
            # Normalise column names
            renames = {}
            for col in earnings.columns:
                lc = col.lower()
                if "eps estimate" in lc:
                    renames[col] = "eps_estimate"
                elif "reported eps" in lc:
                    renames[col] = "eps_actual"
                elif "surprise" in lc:
                    renames[col] = "surprise_pct"
                elif "date" in lc or "earnings" in lc:
                    renames[col] = "earnings_date"
            earnings = earnings.rename(columns=renames)
            if "earnings_date" not in earnings.columns:
                earnings.columns.values[0] = "earnings_date"
            earnings["earnings_date"] = pd.to_datetime(
                earnings["earnings_date"], utc=True, errors="coerce"
            )

        return {"info": info, "earnings": earnings}
    except Exception as exc:
        return {"error": str(exc), "info": {}, "earnings": None}


# ─── Feature computation ──────────────────────────────────────────────────────

def compute_live_features(ticker: str, hist: pd.DataFrame) -> dict:
    """Build feature dict from historical features CSV (pre-computed)."""
    features: dict = {}

    if not hist.empty and ticker in hist["ticker"].values:
        grp = hist[hist["ticker"] == ticker].sort_values("earnings_date")
        labeled = grp[grp["beat"].notna()]

        # Rolling beat rate (last 8 quarters)
        recent = labeled.tail(8)["beat"]
        features["historical_beat_rate"] = recent.mean() if len(recent) >= 2 else np.nan

        # Beat streak
        streak, direction = 0, None
        for val in labeled["beat"].values[::-1]:
            if direction is None:
                direction = val
            if val == direction:
                streak += (1 if direction == 1 else -1)
            else:
                break
        features["beat_streak"] = streak

        # Days since last beat
        beats = labeled[labeled["beat"] == 1]
        if not beats.empty:
            features["days_since_last_beat"] = (
                pd.Timestamp.now() - beats["earnings_date"].max()
            ).days
        else:
            features["days_since_last_beat"] = np.nan

        # Latest financial metrics
        for col in ["gross_margin", "operating_margin", "roe",
                    "debt_equity", "current_ratio", "revenue_growth_yoy", "margin_trend"]:
            ser = grp[col].dropna() if col in grp.columns else pd.Series(dtype=float)
            features[col] = ser.iloc[-1] if not ser.empty else np.nan

        # Estimate revision proxy
        if "eps_estimate" in grp.columns:
            est = grp["eps_estimate"].dropna().tail(2)
            features["estimate_revision_trend"] = (
                float(est.iloc[-1] - est.iloc[-2]) if len(est) >= 2 else np.nan
            )

        # Guidance sentiment
        if "guidance_sentiment" in grp.columns:
            s = grp["guidance_sentiment"].dropna()
            features["guidance_sentiment"] = int(s.iloc[-1]) if not s.empty else 0
        else:
            features["guidance_sentiment"] = 0

    # Sentiment from SEC EDGAR (cached)
    try:
        sent = get_sentiment_for_ticker(ticker)
        features["sentiment_score"] = sent.get("sentiment_score", 0.0)
    except Exception:
        features["sentiment_score"] = 0.0

    return features


def build_feature_vector(features: dict, feat_names: list, medians: dict) -> np.ndarray:
    row = []
    for name in feat_names:
        val = features.get(name)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = medians.get(name, 0.0)
        row.append(float(val))
    return np.array(row, dtype=float).reshape(1, -1)


# ─── UI components ────────────────────────────────────────────────────────────

def render_gauge(probability: float) -> go.Figure:
    pct = probability * 100
    fig = go.Figure(go.Indicator(
        mode  ="gauge+number+delta",
        value = pct,
        number={"suffix": "%", "font": {"size": 38}},
        delta = {"reference": 50, "increasing": {"color": "#2ecc71"},
                 "decreasing": {"color": "#e74c3c"}},
        title = {"text": "Beat Probability", "font": {"size": 18}},
        gauge = {
            "axis":    {"range": [0, 100], "ticksuffix": "%"},
            "bar":     {"color": "#2c3e50", "thickness": 0.25},
            "bgcolor": "white",
            "steps": [
                {"range": [0,  35], "color": "#fadbd8"},
                {"range": [35, 50], "color": "#fdebd0"},
                {"range": [50, 65], "color": "#d5f5e3"},
                {"range": [65, 100],"color": "#a9dfbf"},
            ],
            "threshold": {
                "line": {"color": "#7f8c8d", "width": 3},
                "thickness": 0.8,
                "value": 50,
            },
        },
    ))
    fig.update_layout(height=280, margin=dict(l=30, r=30, t=50, b=10))
    return fig


def render_prediction_signal(prob: float, threshold: float):
    if prob >= threshold:
        st.success(f"🎯 **High-confidence BEAT** signal  ({prob:.1%})")
    elif prob >= 0.55:
        st.info(f"📈 **Moderate BEAT** signal  ({prob:.1%})")
    elif prob <= (1 - threshold):
        st.error(f"⚠️ **High-confidence MISS** signal  ({prob:.1%})")
    elif prob <= 0.45:
        st.warning(f"📉 **Moderate MISS** signal  ({prob:.1%})")
    else:
        st.info(f"➖ **Uncertain** — close call  ({prob:.1%})")


def render_top_drivers(features: dict, feat_names: list, model, medians: dict):
    """Show top 5 features ranked by model importance × deviation from median."""
    if not hasattr(model, "feature_importances_"):
        st.info("Feature importance not available for this model type.")
        return

    importances = dict(zip(feat_names, model.feature_importances_))

    LABELS = {
        "historical_beat_rate":   ("Historical beat rate",      "% of last 8 qtrs with a beat"),
        "beat_streak":            ("Beat streak",               "consecutive beats (+) or misses (−)"),
        "days_since_last_beat":   ("Days since last beat",      "recency of last positive surprise"),
        "estimate_revision_trend":("Estimate revision",         "consensus EPS Δ vs prior quarter"),
        "guidance_sentiment":     ("Guidance sentiment",        "raised / maintained / lowered proxy"),
        "revenue_growth_yoy":     ("Revenue growth (YoY)",      "year-over-year revenue Δ"),
        "gross_margin":           ("Gross margin",              "gross profit ÷ revenue"),
        "operating_margin":       ("Operating margin",          "operating income ÷ revenue"),
        "roe":                    ("Return on equity",          "net income ÷ stockholders equity"),
        "debt_equity":            ("Debt / Equity",             "financial leverage"),
        "current_ratio":          ("Current ratio",             "short-term liquidity"),
        "margin_trend":           ("Margin trend",              "gross margin Δ vs prior quarter"),
        "sentiment_score":        ("SEC filing sentiment",      "MD&A tone from latest 10-Q"),
    }

    drivers = []
    for name in feat_names:
        if name.startswith("sector_"):
            continue
        val = features.get(name)
        med = medians.get(name, 0.0)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        label, desc = LABELS.get(name, (name, ""))
        imp          = importances.get(name, 0)
        positive     = float(val) > float(med or 0)
        drivers.append((imp, label, desc, val, med, positive))

    drivers.sort(reverse=True)

    for rank, (imp, label, desc, val, med, positive) in enumerate(drivers[:5], 1):
        icon = "🟢" if positive else "🔴"
        # Format the value sensibly
        if isinstance(val, float) and 0 < abs(val) < 10:
            val_str = f"{val:+.2f}"
        elif isinstance(val, float):
            val_str = f"{val:.1%}" if 0 <= abs(val) <= 1 else f"{val:+.1f}"
        else:
            val_str = str(val)

        st.markdown(f"**{rank}. {icon} {label}:** `{val_str}`")
        st.caption(f"↳ {desc}")


def render_earnings_history_chart(earnings: pd.DataFrame):
    if earnings is None or earnings.empty:
        st.info("No earnings history available.")
        return

    now  = pd.Timestamp.now(tz="UTC")
    past = earnings[earnings["earnings_date"] <= now].head(8).copy()

    if past.empty:
        st.info("No past earnings events found.")
        return

    # Build display table
    display = past[["earnings_date"]].copy()
    display["Date"] = display["earnings_date"].dt.strftime("%Y-%m-%d")

    for col in ["eps_estimate", "eps_actual", "surprise_pct"]:
        if col in past.columns:
            display[col] = past[col]

    if "surprise_pct" in display.columns:
        display["Beat / Miss"] = display["surprise_pct"].apply(
            lambda x: "✅ Beat" if (pd.notna(x) and x > 2)
            else ("❌ Miss" if (pd.notna(x) and x < -2) else "➖ In-line")
        )

    show_cols = ["Date"]
    rename    = {}
    for src, dst in [("eps_estimate", "EPS Estimate"), ("eps_actual", "EPS Actual"),
                     ("surprise_pct", "Surprise %"), ("Beat / Miss", "Beat / Miss")]:
        if src in display.columns:
            show_cols.append(src)
            rename[src] = dst
        elif src == "Beat / Miss" and "Beat / Miss" in display.columns:
            show_cols.append("Beat / Miss")

    st.dataframe(
        display[show_cols].rename(columns=rename),
        use_container_width=True, hide_index=True,
    )

    if "surprise_pct" in past.columns and past["surprise_pct"].notna().any():
        chart_df = past[past["surprise_pct"].notna()].copy()
        chart_df["Date"] = chart_df["earnings_date"].dt.strftime("%Y-%m-%d")
        chart_df["Signal"] = chart_df["surprise_pct"].apply(
            lambda x: "Beat" if x > 2 else ("Miss" if x < -2 else "In-line")
        )
        fig = px.bar(
            chart_df, x="Date", y="surprise_pct",
            color="Signal",
            color_discrete_map={"Beat": "#2ecc71", "Miss": "#e74c3c", "In-line": "#95a5a6"},
            labels={"surprise_pct": "Surprise %"},
            title="EPS Surprise % — Last 8 Quarters",
        )
        fig.add_hline(y= 2, line_dash="dash", line_color="#2ecc71",
                      annotation_text="+2% beat threshold")
        fig.add_hline(y=-2, line_dash="dash", line_color="#e74c3c",
                      annotation_text="−2% miss threshold")
        fig.update_layout(showlegend=True, height=320)
        st.plotly_chart(fig, use_container_width=True)


def render_sector_comparison(ticker: str, sector: str, hist: pd.DataFrame):
    if hist.empty or "sector" not in hist.columns:
        return
    sec_data = hist[hist["sector"] == sector]["beat"].dropna()
    tkr_data = hist[hist["ticker"] == ticker]["beat"].dropna()

    if sec_data.empty:
        return

    sec_rate = sec_data.mean()
    col1, col2 = st.columns(2)
    with col1:
        st.metric(f"{sector} sector avg", f"{sec_rate:.1%}")
    with col2:
        if not tkr_data.empty:
            tkr_rate = tkr_data.mean()
            st.metric(f"{ticker} beat rate", f"{tkr_rate:.1%}",
                      delta=f"{tkr_rate - sec_rate:+.1%} vs sector")


# ─── Main app ─────────────────────────────────────────────────────────────────

def main():
    st.title("📊 Earnings Surprise Predictor")
    st.caption(
        "Predict whether a company will beat or miss Wall Street consensus "
        "EPS estimates using historical patterns, financial metrics, and SEC filings."
    )

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        confidence_threshold = st.slider(
            "High-confidence threshold",
            min_value=0.50, max_value=0.90, value=0.62, step=0.01,
            help="Minimum probability to show a 'high-confidence' signal.",
        )
        st.divider()
        st.markdown("**Data pipeline**")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("`data_pull.py`")
            st.markdown("`features.py`")
            st.markdown("`sentiment.py`")
            st.markdown("`train.py`")
        with col_b:
            st.markdown("→ earnings_raw.csv")
            st.markdown("→ features.csv")
            st.markdown("→ sentiment_scores.csv")
            st.markdown("→ earnings_model.pkl")
        st.divider()
        st.caption(
            "⚠️ For educational / research use only. "
            "Not financial advice."
        )

    # ── Load model & data ────────────────────────────────────────────────────
    model, feat_names, medians = load_model()
    hist_features = load_features_data()

    if model is None:
        st.error(
            "**Model not found.**  "
            "Run the pipeline first:\n\n"
            "```bash\n"
            "python src/data_pull.py\n"
            "python src/features.py\n"
            "python src/sentiment.py\n"
            "python src/train.py\n"
            "```"
        )
        return

    # ── Ticker input ─────────────────────────────────────────────────────────
    col_in, col_btn = st.columns([4, 1])
    with col_in:
        raw_ticker = st.text_input(
            "Enter ticker symbol",
            placeholder="e.g.  AAPL  MSFT  NVDA  META",
            label_visibility="collapsed",
        )
    with col_btn:
        go_btn = st.button("Analyze ▶", type="primary", use_container_width=True)

    # Quick-pick examples
    examples = ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN", "JPM", "TSLA"]
    ex_cols  = st.columns(len(examples))
    selected = None
    for i, ex in enumerate(examples):
        if ex_cols[i].button(ex, use_container_width=True):
            selected = ex

    ticker = (selected or raw_ticker or "").upper().strip()

    if not ticker:
        st.info("👆 Type a ticker above or click an example to get started.")
        return

    if not (go_btn or selected or raw_ticker):
        return

    # ── Fetch & analyse ──────────────────────────────────────────────────────
    with st.spinner(f"Analysing {ticker}…"):
        data = fetch_ticker_data(ticker)
        if "error" in data or not data.get("info"):
            st.error(f"Could not fetch data for **{ticker}**.  "
                     "Check the symbol and try again.")
            return

        info    = data["info"]
        name    = info.get("longName", ticker)
        sector  = info.get("sector", "Unknown")
        industry= info.get("industry", "N/A")
        mktcap  = info.get("marketCap", 0)

        features = compute_live_features(ticker, hist_features)
        X        = build_feature_vector(features, feat_names, medians)
        beat_prob = float(model.predict_proba(X)[0][1])

    # ── Company header ───────────────────────────────────────────────────────
    st.markdown(f"## {name}  `{ticker}`")
    h1, h2, h3 = st.columns(3)
    h1.metric("Sector",   sector)
    h2.metric("Industry", industry)
    if mktcap:
        h3.metric("Market Cap", f"${mktcap / 1e9:.1f} B")
    st.divider()

    # ── Main two-column layout ───────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("Beat Probability")
        st.plotly_chart(render_gauge(beat_prob), use_container_width=True)
        render_prediction_signal(beat_prob, confidence_threshold)

    with col_right:
        st.subheader("Top 5 Prediction Drivers")
        render_top_drivers(features, feat_names, model, medians)

    st.divider()

    # ── Beat-rate & sector comparison ────────────────────────────────────────
    r1, r2 = st.columns(2, gap="large")

    with r1:
        st.subheader("📈 Historical Beat Performance")
        if not hist_features.empty and ticker in hist_features["ticker"].values:
            tkr_hist = hist_features[hist_features["ticker"] == ticker]
            labeled  = tkr_hist["beat"].dropna()
            if not labeled.empty:
                m1, m2 = st.columns(2)
                m1.metric("Last 8 quarters", f"{labeled.tail(8).mean():.1%}")
                m2.metric("All-time",        f"{labeled.mean():.1%}")

                streak_val = features.get("beat_streak", 0) or 0
                if streak_val > 0:
                    st.success(f"🔥 **{abs(int(streak_val))}-quarter beat streak**")
                elif streak_val < 0:
                    st.error(f"❄️ **{abs(int(streak_val))}-quarter miss streak**")
                else:
                    st.info("No active streak")
            else:
                st.info("Insufficient labeled history.")
        else:
            st.info("Ticker not in training dataset; beat rate unavailable.")

    with r2:
        st.subheader("🏢 Sector Comparison")
        render_sector_comparison(ticker, sector, hist_features)

    st.divider()

    # ── Earnings history ─────────────────────────────────────────────────────
    st.subheader("📋 Last 8 Quarters — Actual vs Estimate")
    render_earnings_history_chart(data.get("earnings"))

    st.divider()

    # ── Model transparency ────────────────────────────────────────────────────
    with st.expander("🔍 How does this model work?"):
        st.markdown("""
### Model Architecture
This dashboard uses a **LightGBM** gradient-boosted tree classifier trained on
S&P 500 earnings events.

### Features
| Feature | Description |
|---------|-------------|
| `historical_beat_rate` | Fraction of last 8 quarters where company beat estimates |
| `beat_streak` | Consecutive beats (+) or misses (−) entering the current quarter |
| `days_since_last_beat` | Calendar days since the most recent positive surprise |
| `estimate_revision_trend` | Quarter-over-quarter change in consensus EPS estimate |
| `guidance_sentiment` | Raised (+1) / maintained (0) / lowered (−1) proxy from estimate Δ |
| `revenue_growth_yoy` | Year-over-year revenue growth from quarterly income statement |
| `gross_margin` | Gross profit ÷ revenue (most recent quarter) |
| `operating_margin` | Operating income ÷ revenue (most recent quarter) |
| `roe` | Net income ÷ stockholders equity |
| `debt_equity` | Total debt ÷ equity (leverage) |
| `current_ratio` | Current assets ÷ current liabilities (liquidity) |
| `margin_trend` | Gross margin change vs prior quarter |
| `sentiment_score` | Loughran-McDonald keyword sentiment from latest 10-Q MD&A |
| `sector_*` | One-hot encoded GICS sector dummies |

### Training Methodology
- **Training set**: S&P 500 earnings events before 2022 (temporal split)
- **Test set**: 2022–2024 (out-of-sample)
- **Target**: `beat = 1` if surprise > +2%, `miss = 0` if surprise < −2% (near-misses excluded)
- **Class balancing**: `scale_pos_weight` compensates for the historical ~65% beat rate
- **Early stopping**: 15% of training data held out as a chronological validation set

### Limitations
> ⚠️ Analyst estimate revision data (30/60/90-day revision counts) is **approximated** —
> production-grade signals would use Bloomberg or Refinitiv.
> Macro environment, management quality, and one-off items are **not modelled**.
> **This is not financial advice.**

### Performance
Check `models/evaluation_metrics.csv` for AUC-ROC, precision, recall, F1, and
betting accuracy at multiple confidence thresholds.
        """)


if __name__ == "__main__":
    main()
