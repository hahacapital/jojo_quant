"""Cross-section backtest — rank stocks per SPX trend × vol regime.

For every stock in the long-history universe, runs jojo Strategy 1 and 2,
tags each trade by its entry-date market regime (3 SPX trend states ×
3 volatility buckets = 9 regimes), and ranks per regime by
`score = profit_factor × sqrt(trades)`.

Usage:
    python3 cross_section.py                     # default: all strategies, top 30, push
    python3 cross_section.py --strategy 1
    python3 cross_section.py --top 50 --no-push
    python3 cross_section.py --limit 50 --no-push  # smoke test
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

import data_loader as dl
from backtest import run_backtest
from screener import EXTRA_TICKERS

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
SPX_CACHE_PATH = REPO_ROOT / "data" / "spx.parquet"
INDEX_MEMBERS_PATH = REPO_ROOT / "data" / "index_members.json"
REPORTS_DIR = REPO_ROOT / "reports"

SPX_SYMBOL = "^GSPC"
SPX_START = "2008-01-01"
MIN_HISTORY_BARS = 2520  # ~10 years of trading days
MIN_TRADES_DEFAULT = 5
TOP_N_DEFAULT = 30

TREND_SMA_FAST = 50
TREND_SMA_SLOW = 200
TREND_SMA_REGIME = 225

VOL_WINDOW = 30          # 30-day realized vol
VOL_RANK_WINDOW = 252 * 5  # 5-year rolling rank
VOL_LOW_Q = 0.33
VOL_HIGH_Q = 0.67


# ---------------------------------------------------------------------------
# SPX cache
# ---------------------------------------------------------------------------

def _save_spx(df: pd.DataFrame) -> None:
    SPX_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = df[["open", "high", "low", "close"]].copy()
    out.index = pd.DatetimeIndex(out.index, name="date")
    out.to_parquet(SPX_CACHE_PATH, compression="snappy")


def _load_cached_spx() -> pd.DataFrame | None:
    if not SPX_CACHE_PATH.exists():
        return None
    df = pd.read_parquet(SPX_CACHE_PATH)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.DatetimeIndex(df.index)
    return df


def load_or_fetch_spx(max_staleness_days: int = 7) -> pd.DataFrame:
    """Return SPX OHLC from cache if fresh, else fetch and update cache."""
    cached = _load_cached_spx()
    if cached is not None and not cached.empty:
        last = cached.index[-1]
        if (pd.Timestamp.utcnow().tz_localize(None) - last).days <= max_staleness_days:
            return cached

    print(f"Fetching SPX ({SPX_SYMBOL}) from {SPX_START}...")
    raw = yf.download(SPX_SYMBOL, start=SPX_START, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.droplevel("Ticker", axis=1)
    raw.columns = [c.lower() for c in raw.columns]
    df = raw[["open", "high", "low", "close"]].dropna(how="all").copy()
    _save_spx(df)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="jojo cross-section backtest")
    parser.add_argument("--strategy", type=str, default="all",
                        choices=["1", "2", "all"])
    parser.add_argument("--top", type=int, default=TOP_N_DEFAULT)
    parser.add_argument("--min-trades", type=int, default=MIN_TRADES_DEFAULT)
    parser.add_argument("--limit", type=int, default=0,
                        help="Restrict universe to first N tickers (smoke test)")
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()

    print("cross_section.py — placeholder; functions added in subsequent tasks")
    print(f"  args = {vars(args)}")


if __name__ == "__main__":
    main()
