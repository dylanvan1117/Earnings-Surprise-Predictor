"""
features.py — Feature engineering for earnings surprise prediction.

Reads  data/earnings_raw.csv  +  data/financials_raw.csv
Writes data/features.csv

Features engineered per earnings event
───────────────────────────────────────
  historical_beat_rate   — company beat % over last 8 quarters (rolling)
  beat_streak            — consecutive beats (+) or misses (−) entering this qtr
  days_since_last_beat   — calendar days since most recent positive surprise
  estimate_revision_trend — change in consensus EPS estimate vs prior quarter
  estimate_revision_count — placeholder (requires premium data)
  guidance_sentiment     — raised (+1) / maintained (0) / lowered (−1) proxy
  revenue_growth_yoy     — YoY revenue growth (from financials)
  gross_margin           — most recent quarter gross margin
  operating_margin       — most recent quarter operating margin
  roe                    — return on equity
  debt_equity            — total debt / equity
  current_ratio          — current assets / current liabilities
  margin_trend           — gross margin Δ vs previous quarter
  sector_*               — one-hot encoded GICS sectors
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

# Columns kept in the final output
BASE_COLS = [
    "ticker", "earnings_date", "sector",
    "eps_estimate", "eps_actual", "surprise_pct", "beat",
    "historical_beat_rate", "beat_streak", "days_since_last_beat",
    "estimate_revision_trend", "estimate_revision_count",
    "guidance_sentiment",
    "revenue_growth_yoy", "gross_margin", "operating_margin",
    "roe", "debt_equity", "current_ratio", "margin_trend",
]


# ─── Beat-history features ─────────────────────────────────────────────────────

def compute_beat_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each earnings event compute rolling look-back features that use only
    information available *before* the current announcement.
    """
    df = df.sort_values(["ticker", "earnings_date"]).copy()

    beat_rates, streaks, days_since = [], [], []

    for _ticker, grp in df.groupby("ticker", sort=False):
        grp = grp.sort_values("earnings_date").reset_index(drop=True)
        n   = len(grp)

        for i in range(n):
            past = grp.iloc[:i]  # strictly prior events

            # ── historical_beat_rate ────────────────────────────────────────
            labeled = past["beat"].dropna().tail(8)
            beat_rates.append(labeled.mean() if len(labeled) >= 2 else np.nan)

            # ── beat_streak ─────────────────────────────────────────────────
            streak    = 0
            direction = None
            for k in range(i - 1, -1, -1):
                val = grp.at[k, "beat"]
                if pd.isna(val):
                    break
                if direction is None:
                    direction = val
                if val == direction:
                    streak += (1 if direction == 1 else -1)
                else:
                    break
            streaks.append(streak)

            # ── days_since_last_beat ────────────────────────────────────────
            past_beats = past[past["beat"] == 1]
            if not past_beats.empty:
                delta = grp.at[i, "earnings_date"] - past_beats["earnings_date"].max()
                days_since.append(delta.days)
            else:
                days_since.append(np.nan)

    df["historical_beat_rate"]  = beat_rates
    df["beat_streak"]           = streaks
    df["days_since_last_beat"]  = days_since
    return df


# ─── Estimate-revision features ────────────────────────────────────────────────

def add_estimate_revision_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Approximate estimate revision from quarter-over-quarter EPS estimate change.
    True 30/60/90-day revision data requires Bloomberg/Refinitiv; this proxy
    captures the direction (raised / cut) well enough for modelling.
    """
    df = df.sort_values(["ticker", "earnings_date"])
    df["estimate_revision_trend"] = df.groupby("ticker")["eps_estimate"].diff()
    df["estimate_revision_count"] = np.nan   # premium data required
    return df


def add_guidance_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Proxy for guidance: if consensus estimate was raised >5c QoQ → +1 (raised),
    cut >5c → -1 (lowered), else 0 (maintained).
    """
    df = df.sort_values(["ticker", "earnings_date"])
    delta = df.groupby("ticker")["eps_estimate"].diff()
    df["guidance_sentiment"] = np.where(
        delta >  0.05,  1,
        np.where(delta < -0.05, -1, 0),
    )
    return df


# ─── Financial-metric features ─────────────────────────────────────────────────

def merge_financial_features(earnings: pd.DataFrame, financials: pd.DataFrame) -> pd.DataFrame:
    """
    For each earnings event, look up the most-recent quarter of financial data
    that precedes the announcement date (as-of join).
    """
    financials = financials.copy()
    financials["date"] = pd.to_datetime(financials["date"], errors="coerce")

    fin_metrics = [
        "gross_margin", "operating_margin", "roe",
        "debt_equity", "current_ratio", "revenue_growth_yoy",
    ]

    rows = []
    for _, row in earnings.iterrows():
        ticker = row["ticker"]
        edate  = row["earnings_date"]

        subset = financials[
            (financials["ticker"] == ticker) & (financials["date"] < edate)
        ].sort_values("date", ascending=False)

        record: dict = {}
        if not subset.empty:
            latest = subset.iloc[0]
            for m in fin_metrics:
                record[m] = latest.get(m, np.nan)

            # margin_trend = most-recent gross_margin − previous quarter gross_margin
            if len(subset) >= 2 and "gross_margin" in subset.columns:
                record["margin_trend"] = (
                    subset.iloc[0]["gross_margin"] - subset.iloc[1]["gross_margin"]
                )
            else:
                record["margin_trend"] = np.nan
        else:
            for m in fin_metrics:
                record[m] = np.nan
            record["margin_trend"] = np.nan

        rows.append(record)

    fin_df = pd.DataFrame(rows, index=earnings.index)
    return pd.concat([earnings, fin_df], axis=1)


# ─── Sector one-hot encoding ──────────────────────────────────────────────────

def add_sector_dummies(df: pd.DataFrame) -> pd.DataFrame:
    if "sector" not in df.columns:
        logger.warning("No sector column; skipping one-hot encoding")
        return df
    df["sector"] = df["sector"].fillna("Unknown")
    dummies = pd.get_dummies(df["sector"], prefix="sector")
    return pd.concat([df, dummies], axis=1)


# ─── Main pipeline ─────────────────────────────────────────────────────────────

def build_features(
    earnings_path: Path | None  = None,
    financials_path: Path | None = None,
) -> pd.DataFrame:

    earnings_path   = earnings_path   or DATA_DIR / "earnings_raw.csv"
    financials_path = financials_path or DATA_DIR / "financials_raw.csv"

    logger.info("Loading earnings data from %s", earnings_path)
    earnings = pd.read_csv(earnings_path)
    earnings["earnings_date"] = pd.to_datetime(earnings["earnings_date"], errors="coerce")

    total = len(earnings)
    labeled = earnings["beat"].notna().sum()
    logger.info("Loaded %d events (%d labeled, %d unlabeled / near-miss)", total, labeled, total - labeled)

    logger.info("Computing beat-history features…")
    earnings = compute_beat_features(earnings)

    logger.info("Adding estimate-revision features…")
    earnings = add_estimate_revision_features(earnings)

    logger.info("Adding guidance sentiment…")
    earnings = add_guidance_sentiment(earnings)

    if financials_path.exists():
        logger.info("Merging quarterly financial data…")
        financials = pd.read_csv(financials_path)
        earnings = merge_financial_features(earnings, financials)
    else:
        logger.warning("financials_raw.csv not found — financial features will be NaN")
        for col in ["gross_margin", "operating_margin", "roe",
                    "debt_equity", "current_ratio", "revenue_growth_yoy", "margin_trend"]:
            earnings[col] = np.nan

    logger.info("Adding sector dummies…")
    earnings = add_sector_dummies(earnings)

    # Keep only labeled rows
    features = earnings[earnings["beat"].notna()].copy()

    # Assemble final column list
    sector_cols = [c for c in features.columns if c.startswith("sector_")]
    final_cols  = [c for c in BASE_COLS + sector_cols if c in features.columns]
    features    = features[final_cols]

    out = DATA_DIR / "features.csv"
    features.to_csv(out, index=False)
    logger.info("Saved %d feature rows → %s", len(features), out)
    return features


if __name__ == "__main__":
    build_features()
