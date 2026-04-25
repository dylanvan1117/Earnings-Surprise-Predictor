"""
data_pull.py — Pull S&P 500 earnings history and quarterly financial data via yfinance.

Produces data/earnings_raw.csv with one row per earnings event including:
  - actual EPS, estimated EPS, surprise %
  - key financial ratios from the surrounding quarter
  - target variable: beat=1 (surprise > 2%), miss=0 (surprise < -2%)
"""

import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


# ─── S&P 500 ticker list ───────────────────────────────────────────────────────

def get_sp500_tickers() -> pd.DataFrame:
    """Return DataFrame[ticker, sector, sub_industry] from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    df = pd.read_html(url)[0][["Symbol", "GICS Sector", "GICS Sub-Industry"]]
    df.columns = ["ticker", "sector", "sub_industry"]
    # yfinance wants dashes, not dots (BRK.B → BRK-B)
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    return df


# ─── Per-ticker helpers ────────────────────────────────────────────────────────

def _normalise_earnings_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename yfinance earnings_dates columns to canonical names."""
    renames = {}
    for col in df.columns:
        lc = col.lower()
        if "eps estimate" in lc or "eps_estimate" in lc:
            renames[col] = "eps_estimate"
        elif "reported eps" in lc or "reported_eps" in lc:
            renames[col] = "eps_actual"
        elif "surprise" in lc:
            renames[col] = "surprise_pct"
    df = df.rename(columns=renames)
    # The index is the earnings date — bring it into a column
    if df.index.name and "date" in df.index.name.lower():
        df = df.reset_index().rename(columns={df.index.name: "earnings_date"})
    elif "earnings_date" not in df.columns:
        df = df.reset_index(drop=False)
        df.columns.values[0] = "earnings_date"
    return df


def get_earnings_data(ticker: str) -> pd.DataFrame:
    """Pull earnings history (EPS actual, estimate, surprise%) for one ticker."""
    try:
        stock = yf.Ticker(ticker)
        raw = stock.earnings_dates
        if raw is None or raw.empty:
            return pd.DataFrame()

        df = _normalise_earnings_columns(raw.copy())
        df["ticker"] = ticker

        # Parse dates — yfinance returns tz-aware; strip tz for uniform handling
        df["earnings_date"] = pd.to_datetime(df["earnings_date"], utc=True, errors="coerce")
        df["earnings_date"] = df["earnings_date"].dt.tz_localize(None)

        # Keep only required columns that actually exist
        keep = ["ticker", "earnings_date"] + [
            c for c in ["eps_estimate", "eps_actual", "surprise_pct"] if c in df.columns
        ]
        return df[keep]

    except Exception as exc:
        logger.warning("Earnings fetch failed for %s: %s", ticker, exc)
        return pd.DataFrame()


def get_financial_data(ticker: str) -> pd.DataFrame:
    """Pull quarterly income-statement and balance-sheet metrics for one ticker."""
    try:
        stock = yf.Ticker(ticker)
        inc = stock.quarterly_income_stmt
        bal = stock.quarterly_balance_sheet

        if inc is None or inc.empty:
            return pd.DataFrame()

        inc_t = inc.T
        bal_t = bal.T if bal is not None and not bal.empty else pd.DataFrame()

        def first_match(df_row, candidates):
            for name in candidates:
                if name in df_row.index and pd.notna(df_row[name]):
                    return df_row[name]
            return np.nan

        records = []
        for date in inc_t.index:
            row_inc = inc_t.loc[date]
            row_bal = bal_t.loc[date] if (not bal_t.empty and date in bal_t.index) else pd.Series(dtype=float)

            revenue  = first_match(row_inc, ["Total Revenue", "Revenue"])
            gprofit  = first_match(row_inc, ["Gross Profit"])
            op_inc   = first_match(row_inc, ["Operating Income", "EBIT"])
            net_inc  = first_match(row_inc, ["Net Income", "Net Income Common Stockholders"])
            assets   = first_match(row_bal, ["Total Assets"])
            debt     = first_match(row_bal, ["Total Debt", "Long Term Debt"])
            equity   = first_match(row_bal, ["Stockholders Equity", "Total Stockholders Equity",
                                              "Common Stock Equity"])
            cur_ast  = first_match(row_bal, ["Current Assets", "Total Current Assets"])
            cur_lib  = first_match(row_bal, ["Current Liabilities", "Total Current Liabilities"])

            records.append({
                "ticker": ticker,
                "date":   date,
                "revenue": revenue,
                "gross_profit": gprofit,
                "operating_income": op_inc,
                "net_income": net_inc,
                "total_assets": assets,
                "total_debt": debt,
                "stockholders_equity": equity,
                "current_assets": cur_ast,
                "current_liabilities": cur_lib,
            })

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_localize(None)
        df = df.sort_values("date").reset_index(drop=True)

        # Derived ratios
        rev = df["revenue"].replace(0, np.nan)
        eq  = df["stockholders_equity"].replace(0, np.nan)
        cl  = df["current_liabilities"].replace(0, np.nan)

        df["gross_margin"]     = df["gross_profit"] / rev
        df["operating_margin"] = df["operating_income"] / rev
        df["roe"]              = df["net_income"] / eq
        df["debt_equity"]      = df["total_debt"] / eq
        df["current_ratio"]    = df["current_assets"] / cl

        # YoY revenue growth (compare to same quarter one year earlier)
        if len(df) >= 4:
            df["revenue_growth_yoy"] = df["revenue"].pct_change(4)
        else:
            df["revenue_growth_yoy"] = np.nan

        return df

    except Exception as exc:
        logger.warning("Financial fetch failed for %s: %s", ticker, exc)
        return pd.DataFrame()


# ─── Main pipeline ─────────────────────────────────────────────────────────────

def pull_all_data(max_tickers: int | None = None, delay: float = 0.5):
    """
    Download earnings + financial data for every S&P 500 company.

    Args:
        max_tickers: cap for quick testing (None = all ~503)
        delay: seconds between yfinance calls to avoid rate-limit bans
    """
    logger.info("Fetching S&P 500 constituent list from Wikipedia…")
    sp500 = get_sp500_tickers()
    if max_tickers:
        sp500 = sp500.head(max_tickers)

    tickers = sp500["ticker"].tolist()
    logger.info("Processing %d tickers", len(tickers))

    all_earnings:   list[pd.DataFrame] = []
    all_financials: list[pd.DataFrame] = []

    for ticker in tqdm(tickers, desc="Pulling data"):
        e = get_earnings_data(ticker)
        if not e.empty:
            all_earnings.append(e)

        f = get_financial_data(ticker)
        if not f.empty:
            all_financials.append(f)

        time.sleep(delay)

    if not all_earnings:
        logger.error(
            "No earnings data collected — check network access to Yahoo Finance.\n"
            "  If you are in an offline/restricted environment, run:\n"
            "    python src/demo_data.py\n"
            "  to generate realistic synthetic data and train the full pipeline."
        )
        return

    # ── Combine earnings ──────────────────────────────────────────────────────
    earnings = pd.concat(all_earnings, ignore_index=True)
    earnings["earnings_date"] = pd.to_datetime(earnings["earnings_date"], errors="coerce")

    # Derive surprise_pct if yfinance didn't supply it
    if "surprise_pct" not in earnings.columns:
        if {"eps_actual", "eps_estimate"}.issubset(earnings.columns):
            earnings["surprise_pct"] = (
                (earnings["eps_actual"] - earnings["eps_estimate"])
                / earnings["eps_estimate"].abs().replace(0, np.nan)
                * 100
            )
        else:
            earnings["surprise_pct"] = np.nan

    # Target: beat=1 if surprise>2%, miss=0 if surprise<-2%, NaN otherwise
    earnings["beat"] = np.where(
        earnings["surprise_pct"] > 2,  1,
        np.where(earnings["surprise_pct"] < -2, 0, np.nan),
    )

    # Keep only past events within the last 10 years
    now          = pd.Timestamp.now()
    ten_years_ago = now - pd.DateOffset(years=10)
    earnings = earnings[
        (earnings["earnings_date"] <= now) &
        (earnings["earnings_date"] >= ten_years_ago)
    ]

    # Attach sector / sub-industry
    earnings = earnings.merge(
        sp500[["ticker", "sector", "sub_industry"]], on="ticker", how="left"
    )

    # ── Combine financials ────────────────────────────────────────────────────
    if all_financials:
        financials = pd.concat(all_financials, ignore_index=True)
        fin_path = DATA_DIR / "financials_raw.csv"
        financials.to_csv(fin_path, index=False)
        logger.info("Saved %d quarterly financial rows → %s", len(financials), fin_path)

    # ── Save earnings ─────────────────────────────────────────────────────────
    out = DATA_DIR / "earnings_raw.csv"
    earnings.to_csv(out, index=False)
    labeled = earnings["beat"].notna()
    logger.info(
        "Saved %d earnings events (%d labeled) → %s",
        len(earnings), labeled.sum(), out,
    )
    beat_rate = earnings.loc[labeled, "beat"].mean()
    logger.info("Historical beat rate (|surprise|>2%%): %.1f%%", beat_rate * 100)
    return earnings


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pull S&P 500 earnings data")
    parser.add_argument("--max-tickers", type=int, default=None,
                        help="Limit number of tickers (for testing)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds between API calls (default 0.5)")
    args = parser.parse_args()
    pull_all_data(max_tickers=args.max_tickers, delay=args.delay)
