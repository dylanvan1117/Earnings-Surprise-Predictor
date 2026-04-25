"""
demo_data.py — Generate realistic synthetic data so the full training pipeline
can be tested without live network access (or when Yahoo Finance is unavailable).

Usage:
    python src/demo_data.py           # generates data/ files and runs the pipeline

The synthetic data mirrors real distributional properties:
  - ~65% historical beat rate (companies beat consensus most of the time)
  - Realistic EPS surprise distribution (right-skewed, mean ≈ +3%)
  - Seasonal quarterly pattern
  - Correlated features (good beat_rate companies tend to have better margins)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

RNG = np.random.default_rng(42)

# A representative cross-sector S&P 500 subset
TICKERS_BY_SECTOR = {
    "Information Technology": [
        "AAPL","MSFT","NVDA","AVGO","ORCL","CSCO","ACN","AMD","INTC","TXN",
        "QCOM","IBM","AMAT","NOW","INTU",
    ],
    "Financials": [
        "JPM","BAC","WFC","GS","MS","BLK","AXP","CB","MMC","PGR",
        "TFC","USB","SCHW","C","COF",
    ],
    "Health Care": [
        "UNH","JNJ","LLY","ABBV","MRK","TMO","ABT","DHR","BMY","AMGN",
        "ISRG","MDT","SYK","ELV","CVS",
    ],
    "Consumer Discretionary": [
        "AMZN","TSLA","HD","MCD","NKE","LOW","SBUX","TJX","BKNG","CMG",
        "ORLY","AZO","MAR","HLT","YUM",
    ],
    "Communication Services": [
        "META","GOOGL","GOOG","NFLX","DIS","CMCSA","T","VZ","TMUS","ATVI",
        "EA","TTWO","WBD","FOXA","NWS",
    ],
    "Industrials": [
        "GE","HON","UPS","CAT","RTX","LMT","BA","DE","MMM","FDX",
        "ETN","EMR","NOC","GD","PH",
    ],
    "Consumer Staples": [
        "PG","KO","PEP","COST","WMT","PM","MO","CL","KHC","GIS",
        "SYY","ADM","HSY","K","CAG",
    ],
    "Energy": [
        "XOM","CVX","SLB","EOG","MPC","PXD","VLO","PSX","COP","OXY",
        "HAL","DVN","BKR","FANG","HES",
    ],
    "Utilities": [
        "NEE","DUK","SO","D","AEP","EXC","SRE","PCG","WEC","ES",
        "ED","XEL","ETR","PPL","CMS",
    ],
    "Real Estate": [
        "PLD","AMT","EQIX","CCI","PSA","WELL","DLR","AVB","EQR","O",
        "VICI","WY","ARE","BXP","VTR",
    ],
    "Materials": [
        "LIN","APD","SHW","FCX","NEM","NUE","DOW","DD","PPG","ALB",
        "CF","MOS","ECL","EMN","RPM",
    ],
}

SECTORS = list(TICKERS_BY_SECTOR.keys())
TICKER_SECTOR = {t: s for s, ticks in TICKERS_BY_SECTOR.items() for t in ticks}

# Approximate sub-industries (simplified)
TICKER_SUB = {t: s + " - General" for s, ticks in TICKERS_BY_SECTOR.items() for t in ticks}


def _quarter_dates(start: str = "2014-01-01", end: str = "2024-12-31") -> list[pd.Timestamp]:
    """Return quarter-end earnings dates over the period."""
    dates = pd.date_range(start, end, freq="QE")
    return dates.tolist()


def generate_earnings_raw(n_quarters: int = 40) -> pd.DataFrame:
    """
    Generate synthetic earnings_raw.csv with realistic distributions.

    Parameters:
        n_quarters: number of quarters of history per ticker (≈10 years = 40 qtrs)
    """
    logger.info("Generating synthetic earnings data for %d tickers × %d quarters…",
                len(TICKER_SECTOR), n_quarters)

    rows = []
    all_dates = _quarter_dates()[-n_quarters:]

    for ticker, sector in TICKER_SECTOR.items():
        # Company-level quality parameter drives beat tendency
        quality     = RNG.beta(3, 2)          # 0–1, mostly above 0.5
        base_margin = RNG.uniform(0.15, 0.55) # gross margin baseline

        prev_beat = 1  # track streak

        for i, q_date in enumerate(all_dates):
            # EPS estimate (slowly grows over time, mean-reverts with noise)
            eps_est = max(0.05, 1.0 + i * 0.02 + RNG.normal(0, 0.3) * quality)

            # Surprise distribution: right-skewed, quality companies beat more
            surprise_mean = (quality - 0.45) * 8   # -3.6% to +4.4% range
            surprise_pct  = surprise_mean + RNG.normal(0, 5)
            # Add momentum: recent beats make future beats slightly more likely
            surprise_pct += prev_beat * 0.5

            eps_actual = eps_est * (1 + surprise_pct / 100)

            # Target label
            if surprise_pct > 2:
                beat = 1.0
            elif surprise_pct < -2:
                beat = 0.0
            else:
                beat = np.nan   # near-miss excluded

            prev_beat = 1 if surprise_pct > 0 else -1

            # Financial metrics (correlated with quality)
            revenue      = max(1e8, RNG.lognormal(22, 1.5))
            gross_margin = np.clip(base_margin + RNG.normal(0, 0.03), 0.05, 0.90)
            op_margin    = gross_margin - RNG.uniform(0.05, 0.20)
            roe          = np.clip(quality * 0.3 + RNG.normal(0, 0.05), -0.2, 0.6)
            d_e          = max(0, RNG.lognormal(0, 0.8))
            cur_ratio    = max(0.3, RNG.lognormal(0.5, 0.4))
            rev_growth   = RNG.normal(0.07, 0.12)

            rows.append({
                "ticker":         ticker,
                "earnings_date":  q_date,
                "eps_estimate":   round(eps_est, 2),
                "eps_actual":     round(eps_actual, 2),
                "surprise_pct":   round(surprise_pct, 2),
                "beat":           beat,
                "sector":         sector,
                "sub_industry":   TICKER_SUB[ticker],
                # Financial metrics (would come from financials_raw.csv merge)
                "gross_margin":        round(gross_margin, 4),
                "operating_margin":    round(op_margin, 4),
                "roe":                 round(roe, 4),
                "debt_equity":         round(d_e, 4),
                "current_ratio":       round(cur_ratio, 4),
                "revenue_growth_yoy":  round(rev_growth, 4),
                "revenue":             round(revenue, 0),
            })

    df = pd.DataFrame(rows)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    return df


def generate_financials_raw(earnings: pd.DataFrame) -> pd.DataFrame:
    """Build a financials_raw.csv from the synthetic earnings table."""
    # We already have financial fields embedded; just reshape to the expected format
    fin_cols = ["ticker", "earnings_date", "gross_margin", "operating_margin",
                "roe", "debt_equity", "current_ratio", "revenue_growth_yoy", "revenue"]
    available = [c for c in fin_cols if c in earnings.columns]
    df = earnings[available].copy()
    df = df.rename(columns={"earnings_date": "date"})
    return df


def generate_features_csv(earnings: pd.DataFrame) -> pd.DataFrame:
    """Run the features pipeline on synthetic data (import features module)."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from features import (
        compute_beat_features,
        add_estimate_revision_features,
        add_guidance_sentiment,
        add_sector_dummies,
    )

    df = compute_beat_features(earnings)
    df = add_estimate_revision_features(df)
    df = add_guidance_sentiment(df)
    df = add_sector_dummies(df)
    df["sentiment_score"] = RNG.uniform(-0.3, 0.6, size=len(df))  # placeholder
    df["margin_trend"] = df.groupby("ticker")["gross_margin"].diff()
    df["estimate_revision_count"] = np.nan

    labeled = df[df["beat"].notna()].copy()
    return labeled


def generate_sentiment_csv(tickers: list) -> pd.DataFrame:
    """Create plausible sentiment scores without hitting EDGAR."""
    rows = []
    for t in tickers:
        rows.append({
            "ticker":         t,
            "sentiment_score": float(RNG.uniform(-0.4, 0.7)),
            "filing_date":    "2024-11-01",
            "word_count":     int(RNG.integers(3000, 8000)),
            "positive_words": int(RNG.integers(40, 120)),
            "negative_words": int(RNG.integers(20, 80)),
            "source":         "synthetic_demo",
        })
    df = pd.DataFrame(rows)
    out = DATA_DIR / "sentiment_scores.csv"
    df.to_csv(out, index=False)
    logger.info("Saved synthetic sentiment scores → %s", out)
    return df


def run_demo_pipeline():
    """Generate all data files and run the full train pipeline."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))

    # 1. earnings_raw.csv
    logger.info("=== Step 1: Generate earnings_raw.csv ===")
    earnings = generate_earnings_raw()
    raw_path = DATA_DIR / "earnings_raw.csv"
    earnings.to_csv(raw_path, index=False)
    logger.info("Saved %d rows → %s", len(earnings), raw_path)

    # 2. financials_raw.csv
    logger.info("=== Step 2: Generate financials_raw.csv ===")
    fin = generate_financials_raw(earnings)
    fin_path = DATA_DIR / "financials_raw.csv"
    fin.to_csv(fin_path, index=False)
    logger.info("Saved %d rows → %s", len(fin), fin_path)

    # 3. features.csv
    logger.info("=== Step 3: Build features.csv ===")
    features = generate_features_csv(earnings)
    feat_path = DATA_DIR / "features.csv"
    features.to_csv(feat_path, index=False)
    logger.info("Saved %d rows → %s", len(features), feat_path)

    # 4. sentiment_scores.csv
    logger.info("=== Step 4: Generate sentiment_scores.csv ===")
    tickers = list(TICKER_SECTOR.keys())
    generate_sentiment_csv(tickers)

    # 5. Train model
    logger.info("=== Step 5: Train model ===")
    from train import run_training
    run_training()

    logger.info("\n✅  Demo pipeline complete.  Run:  streamlit run src/app.py")


if __name__ == "__main__":
    run_demo_pipeline()
