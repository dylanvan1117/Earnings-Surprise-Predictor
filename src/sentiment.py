"""
sentiment.py — SEC EDGAR 10-Q MD&A sentiment scoring.

For each ticker:
  1. Resolves the CIK number via EDGAR company-tickers endpoint.
  2. Finds the most recent 10-Q filing via the submissions API.
  3. Downloads the primary HTML document and extracts the MD&A section.
  4. Scores sentiment with the Loughran-McDonald financial word list (subset).
  5. Caches results in data/sentiment_cache/ as JSON to avoid re-fetches.

Returns a sentiment_score in [-1, 1]:
  +1 = strongly positive language (growth, record, exceeded …)
  -1 = strongly negative language (challenging, headwinds, declined …)
   0 = neutral or unavailable

Usage
-----
  python src/sentiment.py                    # scores a demo set of tickers
  python src/sentiment.py AAPL MSFT GOOGL   # score specific tickers
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
CACHE_DIR = DATA_DIR / "sentiment_cache"
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# SEC requires a descriptive User-Agent; using a placeholder project identifier.
EDGAR_HEADERS = {
    "User-Agent": "EarningsSurprisePredictor research@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json, text/html, */*",
}

# ─── Loughran-McDonald financial sentiment word lists (curated subset) ─────────

POSITIVE_WORDS: set[str] = {
    "growth", "grew", "strong", "record", "exceeded", "outperformed",
    "improved", "improvement", "expanding", "expansion", "accelerating",
    "favorable", "positive", "robust", "solid", "momentum", "gained",
    "increase", "increased", "increasing", "higher", "beat", "ahead",
    "raised", "raising", "surpassed", "exceptional", "excellent",
    "significant", "substantial", "successful", "success", "profitable",
    "profitability", "gains", "gain", "upside", "optimistic", "confidence",
    "confident", "leading", "leadership", "strength", "strengths",
    "opportunity", "opportunities", "breakthrough", "innovation",
    "outperform", "exceeded", "above", "growing", "achieved", "achieve",
    "delivered", "deliver", "strong", "accelerated", "accelerate",
}

NEGATIVE_WORDS: set[str] = {
    "challenging", "challenges", "challenge", "headwinds", "headwind",
    "declined", "decline", "declining", "decreased", "decrease", "decreasing",
    "uncertainty", "uncertain", "risk", "risks", "adverse", "adversely",
    "weakness", "weak", "weakening", "deteriorated", "deterioration",
    "impaired", "impairment", "loss", "losses", "lower", "reduced",
    "reduction", "shortfall", "below", "missed", "miss", "disappointing",
    "disappointed", "difficult", "difficulties", "pressure", "pressures",
    "slowdown", "slowing", "volatile", "volatility", "concern", "concerns",
    "negative", "unfavorable", "downside", "cautious", "caution",
    "restructuring", "writedown", "writeoff", "litigation", "dispute",
    "regulatory", "investigation", "defaulted", "default", "delinquent",
    "bankruptcy", "impairments", "goodwill", "unfavourable", "setback",
}


# ─── CIK resolution ───────────────────────────────────────────────────────────

def _load_cik_cache() -> dict:
    cache_file = CACHE_DIR / "cik_map.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)
    return {}


def _save_cik_cache(cik_map: dict) -> None:
    with open(CACHE_DIR / "cik_map.json", "w") as f:
        json.dump(cik_map, f)


def get_cik_for_ticker(ticker: str) -> Optional[str]:
    """Return zero-padded 10-digit CIK for *ticker*, or None if not found."""
    cik_map = _load_cik_cache()
    if ticker.upper() in cik_map:
        return cik_map[ticker.upper()]

    try:
        resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=EDGAR_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        for entry in resp.json().values():
            t = entry.get("ticker", "").upper()
            cik = str(entry["cik_str"]).zfill(10)
            cik_map[t] = cik
        _save_cik_cache(cik_map)
        return cik_map.get(ticker.upper())
    except Exception as exc:
        logger.warning("CIK lookup failed for %s: %s", ticker, exc)
        return None


# ─── Filing retrieval ─────────────────────────────────────────────────────────

def get_latest_10q_metadata(cik: str) -> Optional[dict]:
    """Return dict with accession number and filing date for the most recent 10-Q."""
    try:
        url  = f"https://data.sec.gov/submissions/CIK{cik}.json"
        resp = requests.get(url, headers=EDGAR_HEADERS, timeout=15)
        resp.raise_for_status()

        recent = resp.json().get("filings", {}).get("recent", {})
        forms       = recent.get("form", [])
        dates       = recent.get("filingDate", [])
        accessions  = recent.get("accessionNumber", [])

        for i, form in enumerate(forms):
            if form == "10-Q":
                acc_dashed  = accessions[i]                          # "0000320193-24-000001"
                acc_nodash  = acc_dashed.replace("-", "")            # "000032019324000001"
                return {"cik": cik, "date": dates[i],
                        "acc_dashed": acc_dashed, "acc_nodash": acc_nodash}
    except Exception as exc:
        logger.warning("Filing lookup failed for CIK %s: %s", cik, exc)
    return None


def _fetch_filing_document(cik: str, acc_nodash: str, acc_dashed: str) -> Optional[str]:
    """
    Download the primary HTML/text document from an EDGAR filing.
    First checks the filing index to find the document URL.
    """
    cik_int    = int(cik)
    index_url  = (
        f"https://www.sec.gov/Archives/edgar/data/{cik_int}"
        f"/{acc_nodash}/{acc_dashed}-index.htm"
    )
    try:
        idx_resp = requests.get(index_url, headers=EDGAR_HEADERS, timeout=15)
        if idx_resp.status_code == 200:
            # Find the first .htm link in the index that is not the index itself
            links = re.findall(r'href="(/Archives/edgar/data/[^"]+\.htm)"',
                               idx_resp.text, re.IGNORECASE)
            for link in links:
                if "index" not in link.lower():
                    doc_url = f"https://www.sec.gov{link}"
                    doc_resp = requests.get(doc_url, headers=EDGAR_HEADERS, timeout=30)
                    if doc_resp.status_code == 200:
                        return doc_resp.text
    except Exception as exc:
        logger.debug("Document fetch error: %s", exc)

    # Fallback: try listing the filing directory
    try:
        dir_url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/"
        )
        dir_resp = requests.get(dir_url, headers=EDGAR_HEADERS, timeout=15)
        if dir_resp.status_code == 200:
            links = re.findall(r'href="([^"]+\.htm)"', dir_resp.text, re.IGNORECASE)
            for link in links:
                if "index" not in link.lower():
                    full = link if link.startswith("http") else f"https://www.sec.gov{link}"
                    r = requests.get(full, headers=EDGAR_HEADERS, timeout=30)
                    if r.status_code == 200:
                        return r.text
    except Exception as exc:
        logger.debug("Directory fallback error: %s", exc)

    return None


# ─── MD&A extraction and scoring ─────────────────────────────────────────────

# Regex patterns for locating the MD&A section (Item 2) inside a 10-Q
_MDA_PATTERNS = [
    r"(?i)item\s+2[\.\s]*[–—-]?\s*management.s\s+discussion\s+and\s+analysis(.*?)"
    r"(?=item\s+3[\.\s]|item\s+4[\.\s]|$)",
    r"(?i)management.s\s+discussion\s+and\s+analysis(.*?)"
    r"(?=quantitative\s+and\s+qualitative|controls\s+and\s+procedures|$)",
]


def extract_mda_text(html: str) -> str:
    """Extract MD&A section from raw HTML/text of a 10-Q filing."""
    # Strip HTML tags
    clean = re.sub(r"<[^>]+>", " ", html)
    clean = re.sub(r"\s+", " ", clean)

    for pattern in _MDA_PATTERNS:
        m = re.search(pattern, clean, re.DOTALL)
        if m:
            return m.group(1)[:60_000]  # cap at 60k chars

    # Fallback: middle section of the document (skip boilerplate header/footer)
    return clean[10_000:70_000]


def score_sentiment(text: str) -> float:
    """
    Compute a normalised sentiment score in [-1, +1].

    Net sentiment = (positive_count − negative_count) / total_words.
    Scaled by 50× so that a typical filing (≈1% sentiment word density)
    maps to ≈ ±0.5.
    """
    if not text:
        return 0.0
    words = re.findall(r"\b[a-z]+\b", text.lower())
    n     = len(words)
    if n == 0:
        return 0.0
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    raw_score = (pos - neg) / n
    return float(np.clip(raw_score * 50, -1.0, 1.0))


# ─── Public API ───────────────────────────────────────────────────────────────

def get_sentiment_for_ticker(ticker: str) -> dict:
    """
    Return sentiment metadata for *ticker*'s most recent 10-Q.
    Result is cached in data/sentiment_cache/<ticker>_sentiment.json.
    """
    cache_file = CACHE_DIR / f"{ticker.upper()}_sentiment.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    result = {
        "ticker": ticker.upper(),
        "sentiment_score": 0.0,
        "filing_date": None,
        "word_count": 0,
        "positive_words": 0,
        "negative_words": 0,
        "source": "not_found",
    }

    try:
        cik = get_cik_for_ticker(ticker)
        if not cik:
            logger.warning("No CIK found for %s", ticker)
            _write_cache(cache_file, result)
            return result

        time.sleep(0.12)   # stay well under EDGAR's 10 req/s limit

        filing = get_latest_10q_metadata(cik)
        if not filing:
            logger.warning("No 10-Q found for %s (CIK %s)", ticker, cik)
            _write_cache(cache_file, result)
            return result

        time.sleep(0.12)

        html = _fetch_filing_document(cik, filing["acc_nodash"], filing["acc_dashed"])
        if not html:
            logger.warning("Could not download filing document for %s", ticker)
            _write_cache(cache_file, result)
            return result

        text  = extract_mda_text(html)
        words = re.findall(r"\b[a-z]+\b", text.lower())
        pos   = sum(1 for w in words if w in POSITIVE_WORDS)
        neg   = sum(1 for w in words if w in NEGATIVE_WORDS)

        result.update({
            "sentiment_score": score_sentiment(text),
            "filing_date":     filing["date"],
            "word_count":      len(words),
            "positive_words":  pos,
            "negative_words":  neg,
            "source":          "edgar_10q",
        })

    except Exception as exc:
        logger.error("Sentiment failed for %s: %s", ticker, exc)

    _write_cache(cache_file, result)
    return result


def _write_cache(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def build_sentiment_dataset(tickers: list[str], delay: float = 0.2) -> pd.DataFrame:
    """
    Score a list of tickers and save results to data/sentiment_scores.csv.

    Args:
        tickers: list of ticker symbols
        delay:   extra sleep between tickers (on top of per-request sleep)
    """
    from tqdm import tqdm

    results = []
    for ticker in tqdm(tickers, desc="SEC filing sentiment"):
        res = get_sentiment_for_ticker(ticker)
        results.append(res)
        time.sleep(delay)

    df = pd.DataFrame(results)
    out = DATA_DIR / "sentiment_scores.csv"
    df.to_csv(out, index=False)
    logger.info("Saved sentiment scores for %d tickers → %s", len(df), out)
    return df


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        tickers = [t.upper() for t in sys.argv[1:]]
    else:
        # Default: attempt to score tickers present in features.csv; else demo set
        feat_file = DATA_DIR / "earnings_raw.csv"
        if feat_file.exists():
            tickers = pd.read_csv(feat_file)["ticker"].unique().tolist()
        else:
            tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
        logger.info("Scoring %d tickers", len(tickers))

    df = build_sentiment_dataset(tickers)
    print(df[["ticker", "sentiment_score", "filing_date", "word_count"]].to_string(index=False))
