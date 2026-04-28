# Cross-Section Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `cross_section.py` tool that classifies every Strategy 1 / Strategy 2 trade across ~1000 stocks into 9 SPX-based market regimes (trend × volatility) and ranks per-regime which stocks fit best by `score = pf × √trades`.

**Architecture:** New orchestration module reusing `data_loader`, `backtest`, `indicators`. Regime classification done once on cached SPX series, then each trade tagged by entry-date regime. Output is markdown report (top-N per regime) plus full CSV.

**Tech Stack:** pandas, numpy, yfinance (SPX cache), `pd.read_html` (index membership scrape), existing `data_loader` parquet cache, existing `backtest.run_backtest`.

**Spec:** `docs/superpowers/specs/2026-04-28-cross-section-backtest-design.md`

---

## File Structure

| File | Status | Responsibility |
|------|--------|----------------|
| `cross_section.py` | Create | Main module: universe build, regimes, classification, aggregation, ranking, output, CLI |
| `data/spx.parquet` | Create at runtime | Cached SPX OHLC (one-time download, refreshed on update) |
| `data/index_members.json` | Create at runtime | Cached Russell 1000 + S&P 500 membership snapshot |
| `reports/cross_section_<date>.md` | Create at runtime | Top-N per-regime ranking report |
| `reports/cross_section_<date>.csv` | Create at runtime | Full per-(ticker,strategy,regime) aggregate |
| `test_logic.py` | Modify | Add 5 tests for regime / aggregation / ranking |
| `README.md` | Modify | Add cross-section usage section |
| `CLAUDE.md` | Modify | Add command snippet + project-file row |

`cross_section.py` is sized for one-file orchestration (~400 lines projected). It will import from `data_loader`, `backtest`, `screener`, `indicators`. No new shared infrastructure needed.

---

## Task 1: Bootstrap `cross_section.py` skeleton + smoke test

**Files:**
- Create: `cross_section.py`

This task establishes the module file with all top-level constants and the `main()` argparse skeleton, so subsequent tasks can fill in functions one at a time.

- [ ] **Step 1: Create `cross_section.py` with module docstring and constants**

```python
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
```

- [ ] **Step 2: Run smoke test**

Run: `python3 cross_section.py --limit 5 --no-push`
Expected output:
```
cross_section.py — placeholder; functions added in subsequent tasks
  args = {'strategy': 'all', 'top': 30, 'min_trades': 5, 'limit': 5, 'no_push': True}
```

- [ ] **Step 3: Commit**

```bash
git add cross_section.py
git commit -m "Scaffold cross_section.py module skeleton"
```

---

## Task 2: SPX cache loader (`load_or_fetch_spx`)

**Files:**
- Modify: `cross_section.py`
- Modify: `test_logic.py`

This function caches SPX OHLC to `data/spx.parquet`. On subsequent runs it loads from disk; if the cached `last_date` is older than today minus 7 days, it re-fetches. Other modules (`generate_report.py`) re-download SPX every run; we cache so successive cross-section runs are fast.

- [ ] **Step 1: Add the failing test to `test_logic.py`**

Add this test function near the other tests, before the `if __name__ == "__main__":` block:

```python
# ============================================================
# Test 9: SPX cache round-trip
# ============================================================
def test_spx_cache_roundtrip(tmp_dir=None):
    """Saving SPX to parquet and re-reading must preserve OHLC."""
    import tempfile
    from pathlib import Path
    import pandas as pd
    import cross_section as cs

    with tempfile.TemporaryDirectory() as td:
        # Force the cache path into a temp dir
        original = cs.SPX_CACHE_PATH
        cs.SPX_CACHE_PATH = Path(td) / "spx.parquet"
        try:
            # Build a fake SPX frame
            dates = pd.bdate_range("2020-01-01", periods=300)
            df = pd.DataFrame({
                "open": np.linspace(3000, 3500, 300),
                "high": np.linspace(3010, 3510, 300),
                "low":  np.linspace(2990, 3490, 300),
                "close": np.linspace(3005, 3505, 300),
            }, index=dates)
            cs._save_spx(df)
            loaded = cs._load_cached_spx()
            assert loaded is not None, "cached SPX should load"
            assert len(loaded) == 300
            assert (loaded["close"] == df["close"]).all()
        finally:
            cs.SPX_CACHE_PATH = original

    print("  PASS: SPX cache roundtrip preserves OHLC")
```

Append to the `tests` list in `__main__`:

```python
        ("SPX cache roundtrip", test_spx_cache_roundtrip),
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -c "from test_logic import test_spx_cache_roundtrip; test_spx_cache_roundtrip()"`
Expected: AttributeError or ImportError mentioning `_save_spx` / `_load_cached_spx` not defined.

- [ ] **Step 3: Implement the SPX cache helpers in `cross_section.py`**

Add after the constants block, before `main()`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -c "from test_logic import test_spx_cache_roundtrip; test_spx_cache_roundtrip()"`
Expected: `PASS: SPX cache roundtrip preserves OHLC`

- [ ] **Step 5: Commit**

```bash
git add cross_section.py test_logic.py
git commit -m "Add SPX cache loader to cross_section"
```

---

## Task 3: Build trend-state series (no look-ahead)

**Files:**
- Modify: `cross_section.py`
- Modify: `test_logic.py`

Computes the SPX trend state per day from SMA50, SMA200, SMA225. The truncation test guards against any future hand-edit that introduces a centered window or full-sample feature.

- [ ] **Step 1: Add the failing test**

Append to `test_logic.py` before `__main__`:

```python
# ============================================================
# Test 10: trend_state has no look-ahead
# ============================================================
def test_trend_state_no_lookahead():
    """trend_state[t] computed on full series == trend_state[t] on truncated."""
    import cross_section as cs
    np.random.seed(0)
    n = 600
    dates = pd.bdate_range("2018-01-01", periods=n)
    close = 3000 + np.cumsum(np.random.randn(n) * 5)
    spx_full = pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
    }, index=dates)

    cutoff_idx = 500
    cutoff = dates[cutoff_idx]
    spx_trunc = spx_full.iloc[: cutoff_idx + 1]

    trend_full = cs.build_trend_state(spx_full)
    trend_trunc = cs.build_trend_state(spx_trunc)
    assert trend_full.loc[cutoff] == trend_trunc.loc[cutoff], (
        f"trend_state[{cutoff.date()}] differs: full={trend_full.loc[cutoff]} "
        f"vs trunc={trend_trunc.loc[cutoff]}"
    )
    print("  PASS: trend_state has no look-ahead")
```

Append to `tests` list:

```python
        ("trend_state no look-ahead", test_trend_state_no_lookahead),
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -c "from test_logic import test_trend_state_no_lookahead; test_trend_state_no_lookahead()"`
Expected: AttributeError: module 'cross_section' has no attribute 'build_trend_state'.

- [ ] **Step 3: Implement `build_trend_state`**

Add to `cross_section.py` after the SPX cache helpers:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -c "from test_logic import test_trend_state_no_lookahead; test_trend_state_no_lookahead()"`
Expected: `PASS: trend_state has no look-ahead`.

- [ ] **Step 5: Commit**

```bash
git add cross_section.py test_logic.py
git commit -m "Add SPX trend_state classifier"
```

---

## Task 4: Build vol-bucket series (no look-ahead, rolling rank)

**Files:**
- Modify: `cross_section.py`
- Modify: `test_logic.py`

Volatility regime uses 30-day realized log-return vol, then a 5-year **rolling** percentile rank — never full-sample, which would be look-ahead.

- [ ] **Step 1: Add the failing test**

Append to `test_logic.py`:

```python
# ============================================================
# Test 11: vol_bucket has no look-ahead
# ============================================================
def test_vol_bucket_no_lookahead():
    """vol_bucket[t] from full series == from truncated series at the same t."""
    import cross_section as cs
    np.random.seed(1)
    n = 1800
    dates = pd.bdate_range("2015-01-01", periods=n)
    rets = np.random.randn(n) * 0.01
    close = 3000 * np.exp(np.cumsum(rets))
    spx_full = pd.DataFrame({
        "open": close, "high": close, "low": close, "close": close,
    }, index=dates)

    cutoff_idx = 1500
    cutoff = dates[cutoff_idx]
    spx_trunc = spx_full.iloc[: cutoff_idx + 1]

    vb_full = cs.build_vol_bucket(spx_full)
    vb_trunc = cs.build_vol_bucket(spx_trunc)
    assert vb_full.loc[cutoff] == vb_trunc.loc[cutoff], (
        f"vol_bucket[{cutoff.date()}] differs: full={vb_full.loc[cutoff]} "
        f"vs trunc={vb_trunc.loc[cutoff]}"
    )
    print("  PASS: vol_bucket has no look-ahead")
```

Append to `tests` list:

```python
        ("vol_bucket no look-ahead", test_vol_bucket_no_lookahead),
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -c "from test_logic import test_vol_bucket_no_lookahead; test_vol_bucket_no_lookahead()"`
Expected: AttributeError: module 'cross_section' has no attribute 'build_vol_bucket'.

- [ ] **Step 3: Implement `build_vol_bucket`**

Add to `cross_section.py` directly below `build_trend_state`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -c "from test_logic import test_vol_bucket_no_lookahead; test_vol_bucket_no_lookahead()"`
Expected: `PASS: vol_bucket has no look-ahead`.

- [ ] **Step 5: Commit**

```bash
git add cross_section.py test_logic.py
git commit -m "Add SPX vol_bucket classifier with rolling rank"
```

---

## Task 5: Combine into `build_regimes` + `lookup_regime`

**Files:**
- Modify: `cross_section.py`
- Modify: `test_logic.py`

Joins trend × vol into the 9 combined labels, plus a lookup helper that uses `method="ffill"` so any non-trading date returns the most-recent prior regime.

- [ ] **Step 1: Add the failing test**

Append to `test_logic.py`:

```python
# ============================================================
# Test 12: combined regime labels + ffill lookup
# ============================================================
def test_build_regimes_and_lookup():
    """build_regimes gives 9 buckets + warmup; lookup ffills to prior trading day."""
    import cross_section as cs
    np.random.seed(2)
    n = 1800
    dates = pd.bdate_range("2015-01-01", periods=n)
    close = 3000 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
    spx = pd.DataFrame({
        "open": close, "high": close, "low": close, "close": close,
    }, index=dates)

    regimes = cs.build_regimes(spx)
    assert {"trend_state", "vol_bucket", "regime"} <= set(regimes.columns)
    valid_labels = {
        f"{t}_{v}"
        for t in ("bull", "bear", "neutral")
        for v in ("low_vol", "mid_vol", "high_vol")
    } | {"warmup"}
    assert set(regimes["regime"].unique()) <= valid_labels, (
        f"unexpected regime labels: {set(regimes['regime'].unique()) - valid_labels}"
    )

    # Lookup on a Saturday (non-trading day) should ffill to Friday.
    fri = dates[1500]
    sat = fri + pd.Timedelta(days=1)
    assert cs.lookup_regime(str(sat.date()), regimes) == regimes.loc[fri, "regime"]

    # Lookup before any data → 'warmup'
    assert cs.lookup_regime("1990-01-01", regimes) == "warmup"

    print("  PASS: build_regimes labels + lookup_regime ffill")
```

Append to `tests` list:

```python
        ("build_regimes + lookup", test_build_regimes_and_lookup),
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -c "from test_logic import test_build_regimes_and_lookup; test_build_regimes_and_lookup()"`
Expected: AttributeError on `cs.build_regimes` or `cs.lookup_regime`.

- [ ] **Step 3: Implement `build_regimes` and `lookup_regime`**

Add below `build_vol_bucket` in `cross_section.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -c "from test_logic import test_build_regimes_and_lookup; test_build_regimes_and_lookup()"`
Expected: `PASS: build_regimes labels + lookup_regime ffill`.

- [ ] **Step 5: Commit**

```bash
git add cross_section.py test_logic.py
git commit -m "Combine trend_state + vol_bucket into 9-label regimes + lookup"
```

---

## Task 6: Universe builder (cache ∩ membership ∩ ≥10y + commodities)

**Files:**
- Modify: `cross_section.py`
- Modify: `test_logic.py`

Builds the analysis universe: tickers that have ≥10 years of cached daily bars **and** are members of the current Russell 1000 or S&P 500 from Wikipedia, plus the commodity futures (no membership requirement; only the 10y filter applies). Membership is cached to `data/index_members.json` after first scrape; the cache is reused indefinitely until manually deleted.

- [ ] **Step 1: Add the failing test**

Append to `test_logic.py`:

```python
# ============================================================
# Test 13: universe builder respects cache + membership + min history
# ============================================================
def test_build_universe_filters():
    """build_universe should keep tickers in (cache ∩ membership) with ≥min_bars,
    drop everything else, and always include commodity futures with ≥min_bars."""
    import cross_section as cs

    cache_meta = pd.DataFrame({
        "num_bars": [3000, 3000, 1000, 3000, 3000],
        "status": ["active", "active", "active", "active", "active"],
    }, index=["AAPL", "MSFT", "OLDCO", "GC=F", "RANDM"])

    members = {"AAPL", "MSFT"}  # OLDCO too short, RANDM not a member, GC=F is commodity
    universe = cs._filter_universe(cache_meta, members, min_bars=2520,
                                   commodity_set={"GC=F"})

    assert set(universe) == {"AAPL", "MSFT", "GC=F"}, f"got {universe}"
    print("  PASS: build_universe filters cache ∩ membership ∩ ≥min_bars + commodities")
```

Append to `tests` list:

```python
        ("Universe filters", test_build_universe_filters),
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -c "from test_logic import test_build_universe_filters; test_build_universe_filters()"`
Expected: AttributeError on `cs._filter_universe`.

- [ ] **Step 3: Implement universe helpers**

Add to `cross_section.py`, below `lookup_regime`:

```python
# ---------------------------------------------------------------------------
# Universe construction
# ---------------------------------------------------------------------------

WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKI_RUSSELL1000 = "https://en.wikipedia.org/wiki/Russell_1000_Index"


def _scrape_index_members() -> set[str]:
    """Scrape Russell 1000 + S&P 500 ticker symbols from Wikipedia."""
    members: set[str] = set()
    for label, url in (("S&P 500", WIKI_SP500), ("Russell 1000", WIKI_RUSSELL1000)):
        try:
            tables = pd.read_html(url)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -c "from test_logic import test_build_universe_filters; test_build_universe_filters()"`
Expected: `PASS: build_universe filters cache ∩ membership ∩ ≥min_bars + commodities`.

- [ ] **Step 5: Commit**

```bash
git add cross_section.py test_logic.py
git commit -m "Add universe builder with Wikipedia membership cache"
```

---

## Task 7: Trade classification (`classify_trades`)

**Files:**
- Modify: `cross_section.py`
- Modify: `test_logic.py`

Maps each `Trade` from a `StrategyResult` to a record dict tagged with the entry-date regime.

- [ ] **Step 1: Add the failing test**

Append to `test_logic.py`:

```python
# ============================================================
# Test 14: classify_trades attaches the right regime per trade
# ============================================================
def test_classify_trades_tags_entry_regime():
    """classify_trades produces (ticker, strategy, regime, pnl_pct, holding_days)
    rows where regime corresponds to entry_date."""
    import cross_section as cs
    from backtest import Trade

    dates = pd.bdate_range("2020-01-01", periods=10)
    regimes = pd.DataFrame({
        "trend_state": ["bull"] * 10,
        "vol_bucket": ["low_vol"] * 5 + ["high_vol"] * 5,
        "regime": ["bull_low_vol"] * 5 + ["bull_high_vol"] * 5,
    }, index=dates)

    trades = [
        Trade(entry_date=str(dates[2]), entry_price=100, exit_date=str(dates[4]),
              exit_price=110, holding_days=2, pnl_pct=10.0, exit_reason="x"),
        Trade(entry_date=str(dates[7]), entry_price=200, exit_date=str(dates[9]),
              exit_price=180, holding_days=2, pnl_pct=-10.0, exit_reason="y"),
    ]
    rows = cs.classify_trades("AAPL", "S1", trades, regimes)
    assert len(rows) == 2
    assert rows[0]["regime"] == "bull_low_vol"
    assert rows[1]["regime"] == "bull_high_vol"
    assert rows[0]["ticker"] == "AAPL" and rows[0]["strategy"] == "S1"
    assert rows[0]["pnl_pct"] == 10.0
    print("  PASS: classify_trades tags entry-date regime per trade")
```

Append to `tests` list:

```python
        ("classify_trades", test_classify_trades_tags_entry_regime),
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -c "from test_logic import test_classify_trades_tags_entry_regime; test_classify_trades_tags_entry_regime()"`
Expected: AttributeError on `cs.classify_trades`.

- [ ] **Step 3: Implement `classify_trades`**

Add to `cross_section.py`, below `build_universe`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -c "from test_logic import test_classify_trades_tags_entry_regime; test_classify_trades_tags_entry_regime()"`
Expected: `PASS: classify_trades tags entry-date regime per trade`.

- [ ] **Step 5: Commit**

```bash
git add cross_section.py test_logic.py
git commit -m "Tag trades with entry-date regime"
```

---

## Task 8: Aggregation (`aggregate`)

**Files:**
- Modify: `cross_section.py`
- Modify: `test_logic.py`

Group trade records by `(ticker, strategy, regime)` and compute the metric block listed in the spec. Max-drawdown is computed by compounding `pnl_pct` chronologically per group.

- [ ] **Step 1: Add the failing test**

Append to `test_logic.py`:

```python
# ============================================================
# Test 15: aggregate computes pf, win_rate, max_dd
# ============================================================
def test_aggregate_metrics():
    """aggregate produces correct trades / win_rate / pf / max_dd."""
    import cross_section as cs
    rows = [
        {"ticker": "A", "strategy": "S1", "regime": "bull_low_vol",
         "entry_date": "2020-01-02", "pnl_pct": 10.0, "holding_days": 5},
        {"ticker": "A", "strategy": "S1", "regime": "bull_low_vol",
         "entry_date": "2020-02-02", "pnl_pct": -5.0, "holding_days": 3},
        {"ticker": "A", "strategy": "S1", "regime": "bull_low_vol",
         "entry_date": "2020-03-02", "pnl_pct": 20.0, "holding_days": 8},
    ]
    agg = cs.aggregate(rows)
    r = agg.iloc[0]
    assert r["trades"] == 3
    assert abs(r["win_rate"] - (2 / 3 * 100)) < 1e-6
    assert abs(r["total_pnl"] - 25.0) < 1e-6
    # gross profit 30, gross loss 5 → pf = 6
    assert abs(r["pf"] - 6.0) < 1e-6
    # equity 1.0 → 1.10 → 1.045 → 1.254. peak=1.10 then dropping to 1.045 = 5% drawdown
    assert abs(r["max_dd"] - 5.0) < 1e-3
    print("  PASS: aggregate metrics match expected values")
```

Append to `tests` list:

```python
        ("Aggregate metrics", test_aggregate_metrics),
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -c "from test_logic import test_aggregate_metrics; test_aggregate_metrics()"`
Expected: AttributeError on `cs.aggregate`.

- [ ] **Step 3: Implement `aggregate`**

Add to `cross_section.py`, below `classify_trades`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -c "from test_logic import test_aggregate_metrics; test_aggregate_metrics()"`
Expected: `PASS: aggregate metrics match expected values`.

- [ ] **Step 5: Commit**

```bash
git add cross_section.py test_logic.py
git commit -m "Aggregate trades to per-(ticker,strategy,regime) metrics"
```

---

## Task 9: Ranking + perfect-record split

**Files:**
- Modify: `cross_section.py`
- Modify: `test_logic.py`

Apply the `trades ≥ N` filter and split off `pf == inf` rows. Sort by score desc, total_pnl desc, win_rate desc.

- [ ] **Step 1: Add the failing test**

Append to `test_logic.py`:

```python
# ============================================================
# Test 16: rank applies min-trades filter and splits perfect records
# ============================================================
def test_rank_filters_and_splits():
    """rank() drops trades < min_trades and routes pf=inf to perfect_record."""
    import cross_section as cs
    agg = pd.DataFrame([
        {"ticker": "LOW", "strategy": "S1", "regime": "bull_low_vol",
         "trades": 3, "win_rate": 100.0, "total_pnl": 12.0, "avg_pnl": 4.0,
         "pf": 5.0, "max_dd": 0.5, "avg_holding": 4, "score": 5 * math.sqrt(3)},
        {"ticker": "PERF", "strategy": "S1", "regime": "bull_low_vol",
         "trades": 8, "win_rate": 100.0, "total_pnl": 80.0, "avg_pnl": 10.0,
         "pf": float("inf"), "max_dd": 0.0, "avg_holding": 5, "score": float("inf")},
        {"ticker": "GOOD", "strategy": "S1", "regime": "bull_low_vol",
         "trades": 12, "win_rate": 75.0, "total_pnl": 200.0, "avg_pnl": 16.7,
         "pf": 4.0, "max_dd": 5.0, "avg_holding": 6, "score": 4 * math.sqrt(12)},
        {"ticker": "TIE1", "strategy": "S1", "regime": "bear_high_vol",
         "trades": 6, "win_rate": 50.0, "total_pnl": 30.0, "avg_pnl": 5.0,
         "pf": 2.0, "max_dd": 8.0, "avg_holding": 5, "score": 2 * math.sqrt(6)},
        {"ticker": "TIE2", "strategy": "S1", "regime": "bear_high_vol",
         "trades": 6, "win_rate": 50.0, "total_pnl": 60.0, "avg_pnl": 10.0,
         "pf": 2.0, "max_dd": 8.0, "avg_holding": 5, "score": 2 * math.sqrt(6)},
    ])
    main_rank, perfect = cs.rank(agg, min_trades=5)

    # LOW dropped (trades < 5); PERF in perfect; GOOD in main
    assert "LOW" not in set(main_rank["ticker"])
    assert set(perfect["ticker"]) == {"PERF"}
    assert "GOOD" in set(main_rank["ticker"])

    # Tie break: same score, TIE2 has higher total_pnl → ranks above TIE1
    bear = main_rank[main_rank["regime"] == "bear_high_vol"].reset_index(drop=True)
    assert list(bear["ticker"]) == ["TIE2", "TIE1"]
    print("  PASS: rank filters min-trades and splits perfect records")
```

Append to `tests` list:

```python
        ("rank filters + splits", test_rank_filters_and_splits),
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -c "from test_logic import test_rank_filters_and_splits; test_rank_filters_and_splits()"`
Expected: AttributeError on `cs.rank`.

- [ ] **Step 3: Implement `rank`**

Add to `cross_section.py`, below `aggregate`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -c "from test_logic import test_rank_filters_and_splits; test_rank_filters_and_splits()"`
Expected: `PASS: rank filters min-trades and splits perfect records`.

- [ ] **Step 5: Commit**

```bash
git add cross_section.py test_logic.py
git commit -m "Add rank() with min-trades filter + perfect-record split"
```

---

## Task 10: Markdown report renderer

**Files:**
- Modify: `cross_section.py`

Renders the per-regime top-N tables. No automated test — output is human-formatted; the smoke test in Task 13 will inspect it manually.

- [ ] **Step 1: Implement `render_markdown`**

Add to `cross_section.py`, below `rank`:

```python
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
```

- [ ] **Step 2: Smoke check the renderer with a tiny fake input**

Run:
```bash
python3 - <<'PY'
import pandas as pd, math
import cross_section as cs

main = pd.DataFrame([
    {"ticker": "AAA", "strategy": "S1", "regime": "bull_low_vol", "trades": 8,
     "win_rate": 75.0, "total_pnl": 100.0, "avg_pnl": 12.5,
     "pf": 3.0, "max_dd": 4.0, "avg_holding": 5, "score": 3 * math.sqrt(8)},
])
perfect = main.iloc[0:0]
dates = pd.bdate_range("2020-01-01", periods=10)
regimes = pd.DataFrame({"regime": ["bull_low_vol"] * 10}, index=dates)
md = cs.render_markdown(main, perfect, regimes, top_n=30,
                        universe_size=(1, 0), period=("2020-01-01", "2020-01-15"),
                        strategies=["S1"])
print(md)
PY
```
Expected: a markdown string starts with `# Cross-Section Report — ...`, contains a `## Strategy 1` block, and the AAA row in `bull_low_vol`.

- [ ] **Step 3: Commit**

```bash
git add cross_section.py
git commit -m "Render per-regime top-N markdown report"
```

---

## Task 11: CSV writer + git push helper

**Files:**
- Modify: `cross_section.py`

- [ ] **Step 1: Implement `write_csv` and `git_push_reports`**

Add to `cross_section.py`, below the renderer:

```python
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
```

- [ ] **Step 2: Smoke check the CSV writer**

Run:
```bash
python3 - <<'PY'
import pandas as pd, tempfile
from pathlib import Path
import cross_section as cs

agg = pd.DataFrame([
    {"ticker": "AAA", "strategy": "S1", "regime": "bull_low_vol", "trades": 8,
     "win_rate": 75.0, "total_pnl": 100.0, "avg_pnl": 12.5,
     "pf": 3.0, "max_dd": 4.0, "avg_holding": 5, "score": 8.5},
])
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "test.csv"
    cs.write_csv(agg, p)
    print(p.read_text())
PY
```
Expected: a CSV with the header `ticker,strategy,regime,trades,...` and one data row.

- [ ] **Step 3: Commit**

```bash
git add cross_section.py
git commit -m "Add CSV writer and git push helper"
```

---

## Task 12: Wire `main()` end-to-end

**Files:**
- Modify: `cross_section.py`

- [ ] **Step 1: Replace `main()` with the orchestration body**

Replace the placeholder `main()` in `cross_section.py` with:

```python
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
```

- [ ] **Step 2: Smoke test (no push) on a small slice**

Run: `python3 cross_section.py --limit 5 --no-push`

Expected:
- Six numbered phase headers `[1/6]` … `[6/6]` print.
- A non-empty `reports/cross_section_<today>.md` is produced.
- A non-empty `reports/cross_section_<today>.csv` is produced.
- Process exits 0.

If the OHLC cache is empty (`data/ohlc/_meta.parquet` missing), the smoke test will raise: `RuntimeError: OHLC cache is empty; run python3 download_ohlc.py --init first.` That is the correct user-facing error; do not patch around it.

- [ ] **Step 3: Run the full test suite to confirm nothing else regressed**

Run: `python3 test_logic.py`

Expected: all 14+ tests pass (`Results: N passed, 0 failed`).

- [ ] **Step 4: Commit**

```bash
git add cross_section.py
git commit -m "Wire cross_section main() end-to-end"
```

---

## Task 13: Update README and CLAUDE.md

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Read the current README and CLAUDE.md**

Run: `wc -l README.md CLAUDE.md` then read both files in full to find the right insertion points (the existing `screener.py`, `backtest.py`, and `generate_report.py` sections).

- [ ] **Step 2: Add a `cross_section.py` usage section to `README.md`**

Insert after the existing report-generation section (the one that documents `generate_report.py`). Use this content verbatim:

```markdown
### 横截面回测 (cross_section.py)

跨股票回测 jojo Strategy 1 / 2，按 9 个市场环境（SPX 趋势 × 波动率分位）
统计每只股票在每个环境下的表现，找出"哪些股票最适合哪种策略 + 哪种市况"。

```bash
# 全策略 + 推送 GitHub（默认）
python3 cross_section.py

# 仅 Strategy 1，本地运行不推送
python3 cross_section.py --strategy 1 --no-push

# 调阈值
python3 cross_section.py --top 50 --min-trades 10

# 烟雾测试（前 5 只标的）
python3 cross_section.py --limit 5 --no-push
```

输出:
- `reports/cross_section_<日期>.md` —— 每个 regime × 策略 top-N 表格
- `reports/cross_section_<日期>.csv` —— 完整聚合数据

注意:
- 依赖本地 OHLC 缓存（先跑 `python3 download_ohlc.py --init` 生成）
- 首次运行会从 Wikipedia 抓 Russell 1000 + S&P 500 成分到 `data/index_members.json`
- 仅推 GitHub，不推 S3
```

- [ ] **Step 3: Update `CLAUDE.md`**

Add this row to the "项目文件" table (find the existing row for `generate_report.py` and insert after it):

```markdown
| `cross_section.py` | 横截面回测：按 9 个 SPX 趋势 × 波动率 regime 排名每股 S1/S2 表现 |
```

Add this section to the "可用命令" block, after the existing "生成回测报告" section:

```markdown
### 横截面回测

```bash
# 全策略，按当前股票池 (Russell1000 ∪ SP500 ∩ ≥10y) 跑
python3 cross_section.py

# 单策略 + 不推送
python3 cross_section.py --strategy 1 --no-push

# 烟雾测试
python3 cross_section.py --limit 5 --no-push
```
```

Add to the "架构" block, after the `screener.py` bullet, this bullet:

```markdown
- **`cross_section.py`** — 横截面回测：`build_universe()` 取 cache ∩ Wikipedia 大盘成分；`build_regimes()` 在 SPX 上计算 trend (SMA50/200/225) × 5y rolling vol-rank → 9 桶；每股跑 `run_backtest`，按 entry-date regime 聚合后用 `score = pf × √trades` 排名。仅推 GitHub。
```

- [ ] **Step 4: Verify README + CLAUDE.md render**

Run: `python3 -c "import pathlib; print(pathlib.Path('README.md').read_text()[:200]); print('---'); print(pathlib.Path('CLAUDE.md').read_text()[:200])"`

Expected: prints the first 200 chars of both files; no exception.

Manually inspect the diff:

Run: `git diff -- README.md CLAUDE.md | head -120`

Expected: only the new sections added in Step 2 / Step 3 appear; no other lines changed.

- [ ] **Step 5: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "Document cross_section.py in README and CLAUDE.md"
```

---

## Task 14: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Full test suite**

Run: `python3 test_logic.py`

Expected: `Results: N passed, 0 failed`. If the existing tests already passed before this branch began, the count should grow by exactly **8** new tests: `SPX cache roundtrip`, `trend_state no look-ahead`, `vol_bucket no look-ahead`, `build_regimes + lookup`, `Universe filters`, `classify_trades`, `Aggregate metrics`, `rank filters + splits`. Confirm each appears in the per-test output lines.

- [ ] **Step 2: End-to-end smoke run**

Run: `python3 cross_section.py --limit 25 --no-push`

Expected: completes without errors, writes a markdown report and CSV to `reports/`. Open the markdown file and check that:
- The "Universe" header has `Stocks: 25` (or fewer if some failed) and `Commodities: 0` (commodities pruned by `--limit 25` since most stocks come first alphabetically — this is fine for smoke testing).
- At least one regime block under "Strategy 1" has a populated table.
- The "Regime time distribution" table lists at least `bull_*`, `bear_*`, `neutral_*` rows.

- [ ] **Step 3: Sanity spot-check**

Run: `python3 cross_section.py --no-push` (full run; ~2 min)

Open `reports/cross_section_<today>.md` and verify:
- NVDA or TSLA appears in the top of `bull_low_vol` or `bull_mid_vol` for Strategy 1 (matches priors that S1 is high-momentum bull-leaning).
- `bear_high_vol` row counts in the regime distribution concentrate around 2008-Q4 / 2020-Q1 / 2022 (visible in the trade dates inside the CSV — `grep '2020-03' reports/cross_section_<today>.csv` should return rows).

If any check fails, this is a sanity issue (not a hard test failure); report findings and decide whether to revisit the spec.

- [ ] **Step 4: Final commit + push**

If sanity checks pass and the report looks reasonable:

```bash
git status
git log --oneline -10
```

Confirm the branch has 13 clean commits (one per task). The user can decide whether to push or open a PR; do not push without explicit approval.
