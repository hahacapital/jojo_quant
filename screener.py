"""
Stock Screener — scan NASDAQ + NYSE for 温度计 signals.

Strategy 1 (超买动量): thermometer crosses UP through 76 today
Strategy 2 (超卖反转): thermometer was below 28 and turns upward today

Usage:
    python screener.py                # scan both strategies
    python screener.py --top 30       # show top 30 per strategy
"""

import argparse
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from indicators import compute_thermometer


# ---------------------------------------------------------------------------
# 1. Get all NASDAQ + NYSE tickers
# ---------------------------------------------------------------------------

def get_exchange_tickers() -> list[str]:
    """Download NASDAQ + NYSE ticker lists from NASDAQ FTP."""
    tickers = set()
    urls = {
        "NASDAQ": "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "NYSE": "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    }
    for exchange, url in urls.items():
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            lines = resp.text.strip().split("\n")
            for line in lines[1:]:
                parts = line.split("|")
                if not parts:
                    continue
                sym = parts[0].strip()
                if not sym or " " in sym or len(sym) > 6:
                    continue
                if exchange == "NASDAQ" and len(parts) > 6 and parts[6].strip() == "Y":
                    continue
                if exchange == "NYSE" and len(parts) > 6 and parts[6].strip() == "Y":
                    continue
                if any(c in sym for c in ["$", ".", "-"]):
                    continue
                tickers.add(sym)
        except Exception as e:
            print(f"[WARN] Failed to fetch {exchange} list: {e}")

    if not tickers:
        print("[WARN] Could not fetch ticker lists, using fallback S&P500 list")
        tickers = _fallback_sp500()

    return sorted(tickers)


def _fallback_sp500() -> set[str]:
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        return set(tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist())
    except Exception:
        return {"AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"}


# ---------------------------------------------------------------------------
# 2. Download OHLC data in batches
# ---------------------------------------------------------------------------

def download_ohlc(tickers: list[str], days: int = 120, batch_size: int = 200) -> dict[str, pd.DataFrame]:
    end = datetime.now()
    start = end - timedelta(days=days)

    all_data = {}
    total = len(tickers)

    for i in range(0, total, batch_size):
        batch = tickers[i : i + batch_size]
        pct = min(100, (i + len(batch)) / total * 100)
        print(f"\r  Downloading batch {i // batch_size + 1} "
              f"({len(batch)} tickers, {pct:.0f}% total)...", end="", flush=True)

        try:
            df = yf.download(
                batch,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
        except Exception as e:
            print(f"\n[WARN] Batch download failed: {e}")
            continue

        if df.empty:
            continue

        if isinstance(df.columns, pd.MultiIndex):
            for sym in batch:
                try:
                    sub = df[sym].dropna(how="all")
                    if len(sub) < 60:
                        continue
                    sub.columns = [c.lower() for c in sub.columns]
                    if all(c in sub.columns for c in ["open", "high", "low", "close"]):
                        all_data[sym] = sub[["open", "high", "low", "close"]].copy()
                except (KeyError, Exception):
                    pass
        else:
            df.columns = [c.lower() for c in df.columns]
            if len(df) >= 60 and all(c in df.columns for c in ["open", "high", "low", "close"]):
                all_data[batch[0]] = df[["open", "high", "low", "close"]].dropna(how="all").copy()

        time.sleep(0.5)

    print(f"\r  Downloaded data for {len(all_data)} tickers.                    ")
    return all_data


# ---------------------------------------------------------------------------
# 3. Detect signals
# ---------------------------------------------------------------------------

def scan_signals(all_data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scan all stocks for two types of signals.

    Strategy 1 (超买动量): today > 76 AND yesterday <= 76
    Strategy 2 (超卖反转): recently was below 28, today turns up (today > yesterday)
                           while still in the "below 28 zone" (not yet crossed 51)

    Returns (strategy1_df, strategy2_df)
    """
    s1_results = []
    s2_results = []
    errors = 0
    total = len(all_data)

    for idx, (sym, df) in enumerate(all_data.items()):
        if (idx + 1) % 500 == 0:
            print(f"\r  Scanning... {idx + 1}/{total}", end="", flush=True)

        try:
            therm = compute_thermometer(df)
            vals = therm.values
            if len(vals) < 3:
                continue

            today = vals[-1]
            yesterday = vals[-2]
            if np.isnan(today) or np.isnan(yesterday):
                continue

            last_date = (df.index[-1].strftime("%Y-%m-%d")
                         if hasattr(df.index[-1], "strftime") else str(df.index[-1]))
            last_close = float(df["close"].iloc[-1])

            # --- Strategy 1: cross above 76 ---
            if today > 76 and yesterday <= 76:
                s1_results.append({
                    "ticker": sym,
                    "date": last_date,
                    "close": round(last_close, 2),
                    "thermometer": round(today, 2),
                    "prev": round(yesterday, 2),
                })

            # --- Strategy 2: below 28 zone, turning up ---
            # Look back to find if thermometer was recently below 28
            # and has NOT yet crossed above 51 since then
            recent = vals[-20:]  # look back 20 bars
            recent_valid = recent[~np.isnan(recent)]
            if len(recent_valid) >= 3:
                was_below_28 = np.any(recent_valid[:-1] < 28)
                # Check it hasn't already recovered above 51
                if was_below_28:
                    # Find the last time it was below 28
                    below_idx = None
                    for j in range(len(recent) - 2, -1, -1):
                        if not np.isnan(recent[j]) and recent[j] < 28:
                            below_idx = j
                            break
                    if below_idx is not None:
                        # Check no value between below_idx and now crossed 51
                        crossed_51 = False
                        for j in range(below_idx + 1, len(recent)):
                            if not np.isnan(recent[j]) and recent[j] > 51:
                                crossed_51 = True
                                break
                        # Signal: turning up (today > yesterday) and hasn't crossed 51 yet
                        if not crossed_51 and today > yesterday and today < 51:
                            s2_results.append({
                                "ticker": sym,
                                "date": last_date,
                                "close": round(last_close, 2),
                                "thermometer": round(today, 2),
                                "prev": round(yesterday, 2),
                                "recent_low": round(float(np.nanmin(recent)), 2),
                            })

        except Exception:
            errors += 1

    print(f"\r  Scanned {total} stocks ({errors} errors).                    ")

    cols1 = ["ticker", "date", "close", "thermometer", "prev"]
    cols2 = ["ticker", "date", "close", "thermometer", "prev", "recent_low"]

    df1 = pd.DataFrame(s1_results, columns=cols1)
    df2 = pd.DataFrame(s2_results, columns=cols2)

    if not df1.empty:
        df1 = df1.sort_values("thermometer", ascending=False).reset_index(drop=True)
    if not df2.empty:
        df2 = df2.sort_values("thermometer", ascending=True).reset_index(drop=True)

    return df1, df2


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="温度计 stock screener")
    parser.add_argument("--top", type=int, default=0,
                        help="Show only top N results per strategy (default: all)")
    parser.add_argument("--days", type=int, default=120,
                        help="Calendar days of history to download (default: 120)")
    parser.add_argument("--batch", type=int, default=200,
                        help="Tickers per download batch (default: 200)")
    args = parser.parse_args()

    print("=== 温度计 Stock Screener ===")
    print("Strategy 1: cross above 76 (超买动量)")
    print("Strategy 2: below 28 turn up (超卖反转)")
    print()

    print("[1/3] Fetching NASDAQ + NYSE ticker list...")
    tickers = get_exchange_tickers()
    print(f"  Found {len(tickers)} tickers.")
    print()

    print("[2/3] Downloading OHLC data...")
    all_data = download_ohlc(tickers, days=args.days, batch_size=args.batch)
    print()

    print("[3/3] Scanning for signals...")
    s1, s2 = scan_signals(all_data)

    if args.top > 0:
        s1 = s1.head(args.top)
        s2 = s2.head(args.top)

    print()
    print("=" * 60)
    print("Strategy 1 — 超买动量 (cross above 76)")
    print("=" * 60)
    if s1.empty:
        print("  No signals today.")
    else:
        print(f"  Found {len(s1)} signal(s):\n")
        print(s1.to_string(index=False))

    print()
    print("=" * 60)
    print("Strategy 2 — 超卖反转 (below 28, turning up)")
    print("=" * 60)
    if s2.empty:
        print("  No signals today.")
    else:
        print(f"  Found {len(s2)} signal(s):\n")
        print(s2.to_string(index=False))

    print()
    print("Done.")
    return s1, s2


if __name__ == "__main__":
    main()
