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
import io
import json
import math
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
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


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------

def build_trend_state(spx: pd.DataFrame) -> pd.Series:
    """Return per-date trend_state in {'bull', 'bear', 'neutral'}.

    bull    : close >= SMA225 AND SMA50 >= SMA200
    bear    : close <  SMA225 AND SMA50 <  SMA200
    neutral : otherwise (mixed signals)

    All inputs use only data on or before each date. NaN rows
    (warm-up) marked 'warmup'.
    """
    close = spx["close"].astype(float)
    sma50 = close.rolling(TREND_SMA_FAST).mean()
    sma200 = close.rolling(TREND_SMA_SLOW).mean()
    sma225 = close.rolling(TREND_SMA_REGIME).mean()

    state = pd.Series("warmup", index=close.index, dtype=object)
    valid = sma200.notna() & sma225.notna() & sma50.notna()

    bull_mask = valid & (close >= sma225) & (sma50 >= sma200)
    bear_mask = valid & (close < sma225) & (sma50 < sma200)
    neutral_mask = valid & ~bull_mask & ~bear_mask

    state[bull_mask] = "bull"
    state[bear_mask] = "bear"
    state[neutral_mask] = "neutral"
    return state


def build_vol_bucket(spx: pd.DataFrame) -> pd.Series:
    """Return per-date vol_bucket in {'low_vol', 'mid_vol', 'high_vol'}.

    Step 1: 30-day realized log-return vol (annualised).
    Step 2: 5-year rolling percentile rank of that vol series.
    Step 3: bucket by VOL_LOW_Q / VOL_HIGH_Q thresholds.

    Uses only past data; 'warmup' for any row before rolling rank
    has min_periods.
    """
    close = spx["close"].astype(float)
    log_ret = np.log(close).diff()
    vol = log_ret.rolling(VOL_WINDOW).std() * math.sqrt(252)

    rank = vol.rolling(VOL_RANK_WINDOW, min_periods=252).rank(pct=True)

    bucket = pd.Series("warmup", index=close.index, dtype=object)
    valid = rank.notna()
    bucket[valid & (rank <= VOL_LOW_Q)] = "low_vol"
    bucket[valid & (rank > VOL_LOW_Q) & (rank <= VOL_HIGH_Q)] = "mid_vol"
    bucket[valid & (rank > VOL_HIGH_Q)] = "high_vol"
    return bucket


def build_regimes(spx: pd.DataFrame) -> pd.DataFrame:
    """Combine trend_state and vol_bucket into a single labelled DataFrame.

    Returns DataFrame indexed by date with columns:
      trend_state, vol_bucket, regime  (regime = trend_state + '_' + vol_bucket)

    Rows where either dimension is in warm-up are labelled 'warmup'.
    """
    trend = build_trend_state(spx)
    vol = build_vol_bucket(spx)
    df = pd.DataFrame({"trend_state": trend, "vol_bucket": vol})

    is_warmup = (df["trend_state"] == "warmup") | (df["vol_bucket"] == "warmup")
    df["regime"] = df["trend_state"] + "_" + df["vol_bucket"]
    df.loc[is_warmup, "regime"] = "warmup"
    return df


def lookup_regime(entry_date_str: str, regimes: pd.DataFrame) -> str:
    """Return regime label for the given date, ffill to the most recent trading day.

    'warmup' if the date predates all regime data.
    """
    try:
        dt = pd.Timestamp(str(entry_date_str)[:10])
    except Exception:
        return "warmup"
    idx = regimes.index.get_indexer([dt], method="ffill")
    if idx[0] < 0:
        return "warmup"
    return str(regimes.iloc[idx[0]]["regime"])


# ---------------------------------------------------------------------------
# Universe construction
# ---------------------------------------------------------------------------

WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKI_RUSSELL1000 = "https://en.wikipedia.org/wiki/Russell_1000_Index"


WIKI_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def _scrape_index_members() -> set[str]:
    """Scrape Russell 1000 + S&P 500 ticker symbols from Wikipedia.

    Uses requests with a real User-Agent (Wikipedia returns 403 to default
    urllib agent), then hands the HTML to pd.read_html for table parsing.
    """
    members: set[str] = set()
    headers = {"User-Agent": WIKI_UA}
    for label, url in (("S&P 500", WIKI_SP500), ("Russell 1000", WIKI_RUSSELL1000)):
        try:
            resp = requests.get(url, headers=headers, timeout=20)
            resp.raise_for_status()
            tables = pd.read_html(io.StringIO(resp.text))
        except Exception as e:
            print(f"  [WARN] {label} scrape failed: {e}")
            continue
        # Heuristic: pick the first table containing a 'Symbol' or 'Ticker' column.
        for tbl in tables:
            cols = {c.lower(): c for c in tbl.columns.astype(str)}
            sym_col = cols.get("symbol") or cols.get("ticker")
            if sym_col is None:
                continue
            for s in tbl[sym_col].dropna().astype(str):
                s = s.strip().replace(".", "-")  # BRK.B -> BRK-B
                if s and " " not in s and len(s) <= 6:
                    members.add(s)
            break
    return members


def _load_or_scrape_members() -> set[str]:
    if INDEX_MEMBERS_PATH.exists():
        with open(INDEX_MEMBERS_PATH) as f:
            data = json.load(f)
        return set(data.get("members", []))

    print("Scraping Russell 1000 + S&P 500 from Wikipedia...")
    members = _scrape_index_members()
    if not members:
        raise RuntimeError(
            "Index membership scrape returned empty and no cache exists; "
            "fix the scrape or hand-create data/index_members.json with "
            '{"members": [...]}'
        )
    INDEX_MEMBERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_MEMBERS_PATH, "w") as f:
        json.dump({"members": sorted(members),
                   "fetched": datetime.utcnow().isoformat(timespec="seconds")},
                  f, indent=2)
    print(f"  Cached {len(members)} members to {INDEX_MEMBERS_PATH}")
    return members


def _filter_universe(meta: pd.DataFrame, members: set[str], *,
                     min_bars: int, commodity_set: set[str]) -> list[str]:
    """Pure helper for testing: apply filters to a meta-like DataFrame."""
    eligible = meta[meta["num_bars"] >= min_bars]
    keep: list[str] = []
    for ticker in eligible.index:
        if ticker in commodity_set:
            keep.append(ticker)
        elif ticker in members:
            keep.append(ticker)
    return sorted(keep)


def build_universe() -> list[str]:
    meta = dl.read_meta()
    if meta.empty:
        raise RuntimeError(
            "OHLC cache is empty; run `python3 download_ohlc.py --init` first."
        )
    members = _load_or_scrape_members()
    universe = _filter_universe(
        meta, members,
        min_bars=MIN_HISTORY_BARS,
        commodity_set=set(EXTRA_TICKERS),
    )
    n_stocks = sum(1 for t in universe if t not in EXTRA_TICKERS)
    n_commod = sum(1 for t in universe if t in EXTRA_TICKERS)
    print(f"Universe: {n_stocks} stocks + {n_commod} commodities = {len(universe)}")
    return universe


# ---------------------------------------------------------------------------
# Trade classification + aggregation
# ---------------------------------------------------------------------------

def classify_trades(ticker: str, strategy: str, trades: list,
                    regimes: pd.DataFrame) -> list[dict]:
    """Return a record per trade with entry-date regime attached."""
    rows: list[dict] = []
    for t in trades:
        regime = lookup_regime(t.entry_date, regimes)
        rows.append({
            "ticker": ticker,
            "strategy": strategy,
            "regime": regime,
            "entry_date": str(t.entry_date)[:10],
            "pnl_pct": float(t.pnl_pct),
            "holding_days": int(t.holding_days),
        })
    return rows


def _group_metrics(group: pd.DataFrame) -> dict:
    pnls = group["pnl_pct"].astype(float).values
    trades = len(pnls)
    wins = int((pnls > 0).sum())
    win_rate = wins / trades * 100 if trades else 0.0
    gross_profit = float(pnls[pnls > 0].sum()) if trades else 0.0
    gross_loss = float(-pnls[pnls < 0].sum()) if trades else 0.0
    if gross_loss > 0:
        pf = gross_profit / gross_loss
    elif gross_profit > 0:
        pf = float("inf")
    else:
        pf = 0.0

    # Max drawdown on chronologically-ordered equity curve
    chrono = group.sort_values("entry_date")["pnl_pct"].astype(float).values
    equity = 1.0
    peak = equity
    max_dd = 0.0
    for p in chrono:
        equity *= 1 + p / 100.0
        peak = max(peak, equity)
        dd = (peak - equity) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)

    score = pf * math.sqrt(trades) if math.isfinite(pf) else float("inf")

    return {
        "trades": trades,
        "win_rate": round(win_rate, 2),
        "total_pnl": round(float(pnls.sum()), 2),
        "avg_pnl": round(float(pnls.mean()) if trades else 0.0, 3),
        "pf": pf if math.isinf(pf) else round(pf, 3),
        "max_dd": round(max_dd, 2),
        "avg_holding": round(float(group["holding_days"].mean()) if trades else 0.0, 1),
        "score": score if math.isinf(score) else round(score, 3),
    }


def aggregate(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=[
            "ticker", "strategy", "regime",
            "trades", "win_rate", "total_pnl", "avg_pnl",
            "pf", "max_dd", "avg_holding", "score",
        ])
    df = pd.DataFrame(records)
    grouped = df.groupby(["ticker", "strategy", "regime"], sort=False)
    rows = []
    for (ticker, strategy, regime), grp in grouped:
        m = _group_metrics(grp)
        rows.append({"ticker": ticker, "strategy": strategy, "regime": regime, **m})
    out = pd.DataFrame(rows)
    return out


def rank(agg: pd.DataFrame, *, min_trades: int = MIN_TRADES_DEFAULT
         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter & sort the aggregate. Returns (main_rank, perfect_record).

    main_rank: trades >= min_trades AND pf finite, sorted by score desc,
               total_pnl desc, win_rate desc.
    perfect_record: trades >= min_trades AND pf == inf (all wins).
    """
    if agg.empty:
        empty = agg.iloc[0:0].copy()
        return empty, empty.copy()

    eligible = agg[agg["trades"] >= min_trades].copy()
    if eligible.empty:
        return eligible, eligible.copy()

    pf_series = eligible["pf"]
    is_inf = pf_series.apply(lambda v: isinstance(v, float) and math.isinf(v))
    perfect = eligible[is_inf].copy()
    main = eligible[~is_inf].copy()

    main = main.sort_values(
        by=["regime", "score", "total_pnl", "win_rate"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    perfect = perfect.sort_values(
        by=["regime", "total_pnl", "trades"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    return main, perfect


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------

REGIME_ORDER = [
    "bull_low_vol", "bull_mid_vol", "bull_high_vol",
    "neutral_low_vol", "neutral_mid_vol", "neutral_high_vol",
    "bear_low_vol", "bear_mid_vol", "bear_high_vol",
]


def _format_row(rank_idx: int, row: pd.Series) -> str:
    pf = row["pf"]
    pf_str = "inf" if isinstance(pf, float) and math.isinf(pf) else f"{pf:.2f}"
    score = row["score"]
    score_str = "inf" if isinstance(score, float) and math.isinf(score) else f"{score:.2f}"
    return (f"| {rank_idx} | {row['ticker']} | {int(row['trades'])} | "
            f"{row['win_rate']:.1f} | {row['total_pnl']:+.1f} | "
            f"{pf_str} | {row['max_dd']:.1f} | {score_str} |")


def render_markdown(main_rank: pd.DataFrame, perfect: pd.DataFrame,
                    regimes: pd.DataFrame, *, top_n: int,
                    universe_size: tuple[int, int],
                    period: tuple[str, str],
                    strategies: list[str]) -> str:
    n_stocks, n_commod = universe_size
    start_str, end_str = period
    parts: list[str] = []
    parts.append(f"# Cross-Section Report — {datetime.utcnow().date()}\n")
    parts.append("## Universe\n")
    parts.append(f"Stocks: {n_stocks}  |  Commodities: {n_commod}  |  "
                 f"Min history: {MIN_HISTORY_BARS // 252}y\n")
    parts.append(f"Effective period: {start_str} → {end_str}\n")
    parts.append("\n*Note*: Universe drawn from current Russell 1000 + S&P 500 "
                 "membership snapshot, intersected with the local OHLC cache. "
                 "Survivorship bias is acknowledged; delisted names are not in this "
                 "table.\n")

    parts.append("\n## Regime time distribution (trading days)\n")
    counts = regimes["regime"].value_counts()
    parts.append("| regime | days |\n|--------|------|")
    for r in REGIME_ORDER + ["warmup"]:
        if r in counts.index:
            parts.append(f"| {r} | {int(counts[r])} |")

    for strategy in strategies:
        nice = "超买动量" if strategy == "S1" else "超卖反转"
        parts.append(f"\n## Strategy {strategy[1]} — {nice}\n")
        s_main = main_rank[main_rank["strategy"] == strategy]
        s_perf = perfect[perfect["strategy"] == strategy]
        for regime in REGIME_ORDER:
            sub = s_main[s_main["regime"] == regime].head(top_n)
            if sub.empty:
                continue
            parts.append(f"\n### {regime}  (top {len(sub)} by score, "
                         f"min trades={MIN_TRADES_DEFAULT})\n")
            parts.append("| rank | ticker | trades | win% | total_pnl% | "
                         "pf | max_dd% | score |")
            parts.append("|------|--------|--------|------|------------|"
                         "----|---------|-------|")
            for i, (_, row) in enumerate(sub.iterrows(), start=1):
                parts.append(_format_row(i, row))

        if not s_perf.empty:
            parts.append(f"\n### Perfect-record entries (pf = inf)\n")
            parts.append("| ticker | regime | trades | total_pnl% |")
            parts.append("|--------|--------|--------|------------|")
            for _, row in s_perf.iterrows():
                parts.append(f"| {row['ticker']} | {row['regime']} | "
                             f"{int(row['trades'])} | {row['total_pnl']:+.1f} |")
    return "\n".join(parts) + "\n"


def write_csv(agg: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["ticker", "strategy", "regime", "trades", "win_rate",
            "total_pnl", "avg_pnl", "pf", "max_dd", "avg_holding", "score"]
    agg.to_csv(path, index=False, columns=cols)


def git_push_reports(paths: list[Path]) -> None:
    for p in paths:
        subprocess.run(["git", "add", str(p)], check=True, cwd=REPO_ROOT)
    subprocess.run(
        ["git", "commit", "-m", f"cross_section report {datetime.utcnow().date()}"],
        check=True, cwd=REPO_ROOT,
    )
    subprocess.run(["git", "push"], check=True, cwd=REPO_ROOT)


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

    strategies = (["S1", "S2"] if args.strategy == "all"
                  else [f"S{args.strategy}"])

    print("[1/6] Building universe...")
    universe = build_universe()
    if args.limit > 0:
        universe = universe[: args.limit]
        print(f"  --limit {args.limit} → using {len(universe)} tickers")

    print("\n[2/6] Loading SPX + building regimes...")
    spx = load_or_fetch_spx()
    regimes = build_regimes(spx)
    print(f"  SPX bars: {len(spx)}  |  regime labels: "
          f"{regimes['regime'].nunique()} distinct")

    print(f"\n[3/6] Running backtests for {len(universe)} tickers...")
    records: list[dict] = []
    skipped = 0
    for i, ticker in enumerate(universe, start=1):
        if i % 100 == 0:
            print(f"  {i}/{len(universe)}  records so far: {len(records)}")
        try:
            df = dl.load_ohlc(ticker)
        except Exception as e:
            print(f"  [WARN] load {ticker}: {e}")
            skipped += 1
            continue
        try:
            r1, r2 = run_backtest(ticker, df)
        except Exception as e:
            print(f"  [WARN] backtest {ticker}: {e}")
            skipped += 1
            continue
        if "S1" in strategies:
            records.extend(classify_trades(ticker, "S1", r1.trades, regimes))
        if "S2" in strategies:
            records.extend(classify_trades(ticker, "S2", r2.trades, regimes))
    print(f"  Done. Records: {len(records)}, skipped: {skipped}")

    # Drop warmup-tagged trades from the analysis universe
    records = [r for r in records if r["regime"] != "warmup"]
    print(f"  Records after dropping warmup: {len(records)}")

    print("\n[4/6] Aggregating per (ticker, strategy, regime)...")
    agg = aggregate(records)
    print(f"  Groups: {len(agg)}")

    print("\n[5/6] Ranking...")
    main_rank, perfect = rank(agg, min_trades=args.min_trades)
    print(f"  Main-rank rows: {len(main_rank)}  |  perfect-record rows: {len(perfect)}")

    print("\n[6/6] Writing report...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.utcnow().date().isoformat()
    md_path = REPORTS_DIR / f"cross_section_{today}.md"
    csv_path = REPORTS_DIR / f"cross_section_{today}.csv"

    n_stocks = sum(1 for t in universe if t not in EXTRA_TICKERS)
    n_commod = sum(1 for t in universe if t in EXTRA_TICKERS)
    period = (str(spx.index[0].date()), str(spx.index[-1].date()))

    md = render_markdown(main_rank, perfect, regimes, top_n=args.top,
                         universe_size=(n_stocks, n_commod),
                         period=period, strategies=strategies)
    md_path.write_text(md)
    write_csv(agg, csv_path)
    print(f"  Wrote {md_path}\n  Wrote {csv_path}")

    if args.no_push:
        print("\nSkipped git push (--no-push).")
    else:
        print("\nPushing to GitHub...")
        try:
            git_push_reports([md_path, csv_path])
            print("  Push done.")
        except subprocess.CalledProcessError as e:
            print(f"  [WARN] push failed: {e}")


if __name__ == "__main__":
    main()
