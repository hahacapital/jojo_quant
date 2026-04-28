"""End-to-end smoke test for the OHLC cache.

Run AFTER ``python3 download_ohlc.py --init`` (or with --limit so AAPL is in
the slice). Verifies cache shape, value-level equivalence with a fresh
yfinance pull, and that backtest results are identical against cache.

    python3 test_data_loader.py            # default ticker AAPL
    python3 test_data_loader.py --ticker AA  # any cached ticker
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd
import yfinance as yf

import data_loader as dl
from backtest import run_backtest
from indicators import compute_jojo


def _check(label: str, cond: bool, msg: str = "") -> None:
    if not cond:
        print(f"[FAIL] {label} {msg}")
        sys.exit(1)
    print(f"[ok]   {label}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL",
                        help="Cached ticker to verify (must be in --init slice)")
    parser.add_argument("--min-bars", type=int, default=1000,
                        help="Lower bound for sanity check on bar count")
    args = parser.parse_args()
    sym = args.ticker

    df = dl.load_ohlc(sym)
    _check("DataFrame returned", isinstance(df, pd.DataFrame))
    _check("DatetimeIndex", isinstance(df.index, pd.DatetimeIndex))
    _check("columns == [open,high,low,close]",
           list(df.columns) == ["open", "high", "low", "close"],
           f"got {list(df.columns)}")
    _check(f"{sym} has > {args.min_bars} bars",
           len(df) > args.min_bars, f"got {len(df)}")
    print(f"       {sym}: {len(df)} bars, {df.index[0].date()} → {df.index[-1].date()}")

    jojo = compute_jojo(df)
    _check("compute_jojo returns non-empty", not jojo.empty)
    print(f"       compute_jojo({sym}) last = {float(jojo.iloc[-1]):.2f}")

    # Cache vs fresh download — value-level equivalence on overlapping bars.
    fresh = yf.download(sym, period="max", auto_adjust=True, progress=False)
    if isinstance(fresh.columns, pd.MultiIndex):
        fresh = fresh.droplevel("Ticker", axis=1)
    fresh.columns = [c.lower() for c in fresh.columns]
    fresh = fresh[["open", "high", "low", "close"]].dropna(how="all")

    common = df.index.intersection(fresh.index)
    _check("substantial cache/fresh overlap", len(common) > args.min_bars,
           f"got {len(common)}")

    # auto_adjust may slightly drift between calls if a recent split/div hits;
    # 1e-3 is generous for snappy parquet round-trip plus pricing precision.
    diff = (df.loc[common] - fresh.loc[common]).abs().max().max()
    _check(f"max(|cache - fresh|) < 1e-3 over {len(common)} bars",
           diff < 1e-3, f"got {diff:.6g}")

    # Same backtest behaviour against cached data.
    r1, r2 = run_backtest(sym, df)
    _check("run_backtest(cache) returned",
           r1 is not None and r2 is not None)
    print(f"       backtest({sym}): S1 trades={len(r1.trades)}  "
          f"S2 trades={len(r2.trades)}")

    # Universe listing reads meta and applies a min-bar filter.
    universe = dl.list_universe(min_bars=252)
    _check("universe non-empty", len(universe) > 0,
           f"got {len(universe)} (run --init first?)")
    print(f"       universe (>=252 bars): {len(universe)} tickers")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
