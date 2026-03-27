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

def get_exchange_tickers() -> tuple[list[str], dict[str, str]]:
    """Download NASDAQ + NYSE ticker lists from NASDAQ FTP.

    Returns (sorted_tickers, name_map) where name_map maps symbol -> security name.
    """
    tickers = set()
    name_map = {}
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
                if len(parts) > 1:
                    name_map[sym] = parts[1].strip()
        except Exception as e:
            print(f"[WARN] Failed to fetch {exchange} list: {e}")

    if not tickers:
        print("[WARN] Could not fetch ticker lists, using fallback S&P500 list")
        tickers = _fallback_sp500()

    return sorted(tickers), name_map


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

FMP_API_KEY = "F5UGdTRIUvC0ioFTuUcj3zmuUpBzVsEo"
FMP_PROFILE_URL = "https://financialmodelingprep.com/stable/profile"


def _fetch_fmp_profiles(symbols: list[str]) -> dict[str, dict]:
    """Fetch company profiles from FMP (name, sector, industry, marketCap).

    Free tier: 250 requests/day. Each symbol = 1 request.
    """
    result = {}
    for i, sym in enumerate(symbols):
        try:
            resp = requests.get(
                FMP_PROFILE_URL,
                params={"symbol": sym, "apikey": FMP_API_KEY},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    item = data[0]
                    result[sym] = {
                        "companyName": item.get("companyName", ""),
                        "sector": item.get("sector", ""),
                        "industry": item.get("industry", ""),
                        "marketCap": item.get("marketCap", 0) or 0,
                        "isEtf": item.get("isEtf", False),
                    }
                    continue
            elif resp.status_code == 429:
                print(f"\n  [WARN] FMP rate limit at {i}/{len(symbols)}, stopping.", end="")
                break
        except Exception:
            pass
        result[sym] = {"companyName": "", "sector": "", "industry": "", "marketCap": 0, "isEtf": False}
    return result


def enrich_signals(df: pd.DataFrame, name_map: dict[str, str]) -> pd.DataFrame:
    """Add company name, sector, industry, and market cap via FMP API.

    Falls back to NASDAQ FTP name_map for tickers not covered by FMP.
    """
    if df.empty:
        return df

    df = df.copy()
    tickers = df["ticker"].tolist()

    # FMP free tier: 250 req/day. Fetch top tickers, rest use NASDAQ FTP name only.
    fmp_limit = 200
    fetch_tickers = tickers[:fmp_limit] if len(tickers) > fmp_limit else tickers
    if len(tickers) > fmp_limit:
        print(f"  Fetching profiles for top {fmp_limit}/{len(tickers)} tickers (FMP)...", end="", flush=True)
    else:
        print(f"  Fetching profiles for {len(tickers)} tickers (FMP)...", end="", flush=True)
    profiles = _fetch_fmp_profiles(fetch_tickers)
    print(" done.")

    df["name"] = df["ticker"].apply(
        lambda s: profiles.get(s, {}).get("companyName", "") or name_map.get(s, "")
    )
    df["name"] = df["name"].str.slice(0, 40)
    df["sector"] = df["ticker"].apply(lambda s: profiles.get(s, {}).get("sector", ""))
    df["industry"] = df["ticker"].apply(lambda s: profiles.get(s, {}).get("industry", ""))
    df["market_cap"] = df["ticker"].apply(lambda s: profiles.get(s, {}).get("marketCap", 0))

    def fmt_cap(val):
        if not val or val == 0:
            return ""
        if val >= 1e12:
            return f"{val / 1e12:.1f}T"
        if val >= 1e9:
            return f"{val / 1e9:.1f}B"
        if val >= 1e6:
            return f"{val / 1e6:.0f}M"
        return f"{val:.0f}"

    df["mkt_cap_fmt"] = df["market_cap"].apply(fmt_cap)
    df = df.sort_values("market_cap", ascending=False).reset_index(drop=True)

    return df


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
            # Must simulate position state: only signal if NOT already in a trade
            # (i.e. previous cross above 76 must have been closed by crossing below 68)
            in_pos = False
            signal_today = False
            for k in range(1, len(vals)):
                vk = vals[k]
                vk1 = vals[k - 1]
                if np.isnan(vk) or np.isnan(vk1):
                    continue
                if not in_pos:
                    if vk > 76 and vk1 <= 76:
                        in_pos = True
                        if k == len(vals) - 1:
                            signal_today = True
                else:
                    if vk < 68 and vk1 >= 68:
                        in_pos = False

            if signal_today:
                s1_results.append({
                    "ticker": sym,
                    "date": last_date,
                    "close": round(last_close, 2),
                    "thermometer": round(today, 2),
                    "prev": round(yesterday, 2),
                })

            # --- Strategy 2: below 28 zone, turning up ---
            # Simulate position state: buy when below 28 and turning up,
            # sell when crossing above 51 or dropping below 28 again.
            # Only report signal if a NEW buy triggers on the last bar.
            in_pos2 = False
            signal_today2 = False
            recent_low2 = np.nan
            for k in range(1, len(vals)):
                vk = vals[k]
                vk1 = vals[k - 1]
                if np.isnan(vk) or np.isnan(vk1):
                    continue
                if not in_pos2:
                    if vk1 < 28 and vk > vk1:
                        in_pos2 = True
                        recent_low2 = float(np.nanmin(vals[max(0, k - 20):k + 1]))
                        if k == len(vals) - 1:
                            signal_today2 = True
                else:
                    if (vk > 51 and vk1 <= 51) or (vk < 28 and vk1 >= 28):
                        in_pos2 = False

            if signal_today2:
                s2_results.append({
                    "ticker": sym,
                    "date": last_date,
                    "close": round(last_close, 2),
                    "thermometer": round(today, 2),
                    "prev": round(yesterday, 2),
                    "recent_low": round(recent_low2, 2),
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

    print("[1/4] Fetching NASDAQ + NYSE ticker list...")
    tickers, name_map = get_exchange_tickers()
    print(f"  Found {len(tickers)} tickers.")
    print()

    print("[2/4] Downloading OHLC data...")
    all_data = download_ohlc(tickers, days=args.days, batch_size=args.batch)
    print()

    print("[3/4] Scanning for signals...")
    s1, s2 = scan_signals(all_data)

    # Enrich with company info
    print("\n[4/4] Fetching company info...")
    s1 = enrich_signals(s1, name_map)
    s2 = enrich_signals(s2, name_map)

    if args.top > 0:
        s1 = s1.head(args.top)
        s2 = s2.head(args.top)

    display_cols1 = ["ticker", "name", "sector", "industry", "mkt_cap_fmt", "close", "thermometer"]
    display_cols2 = ["ticker", "name", "sector", "industry", "mkt_cap_fmt", "close", "thermometer", "recent_low"]

    print()
    print("=" * 100)
    print("Strategy 1 — 超买动量 (cross above 76)")
    print("=" * 100)
    if s1.empty:
        print("  No signals today.")
    else:
        print(f"  Found {len(s1)} signal(s), sorted by market cap:\n")
        cols = [c for c in display_cols1 if c in s1.columns]
        print(s1[cols].to_string(index=False))

    print()
    print("=" * 100)
    print("Strategy 2 — 超卖反转 (below 28, turning up)")
    print("=" * 100)
    if s2.empty:
        print("  No signals today.")
    else:
        print(f"  Found {len(s2)} signal(s), sorted by market cap:\n")
        cols = [c for c in display_cols2 if c in s2.columns]
        print(s2[cols].to_string(index=False))

    print()
    print("Done.")
    return s1, s2


if __name__ == "__main__":
    main()
