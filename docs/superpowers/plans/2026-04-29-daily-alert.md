# Daily Alert Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `src/daily_alert.py`, a post-close Telegram alert that emits today's jojo Strategy 1 / Strategy 2 signals filtered down to tickers ranked in the top-30 of the latest cross-section CSV for the current 9-bucket SPX regime.

**Architecture:** New orchestration module reusing `screener` (signal scan, FMP profiles, EXTRA_TICKERS, CN_NAMES) and `cross_section` (SPX cache, regime classification). Adds three things of its own: top-30 ranking lookup against the latest `reports/cross_section_*.csv`, an SPX freshness gate that aborts before sending if yfinance is not yet updated, and a Telegram MarkdownV2 sender.

**Tech Stack:** pandas, numpy, requests (Telegram + FMP), yfinance (via `cross_section.load_or_fetch_spx`), `pandas.tseries.holiday.USFederalHolidayCalendar` for the freshness gate. No new pip dependencies.

**Spec:** `docs/superpowers/specs/2026-04-29-daily-alert-design.md`

---

## File Structure

| File | Status | Responsibility |
|------|--------|----------------|
| `src/daily_alert.py` | Create | All daily-alert logic: env, CSV lookup, freshness gate, regime, signal scan, FMP, formatting, Telegram |
| `.env.example` | Create | Committed template; documents required env vars (no real values) |
| `.env` | Create at runtime | Local-only credentials (gitignored) |
| `.gitignore` | Modify | Add `.env` |
| `src/test_logic.py` | Modify | Add 6 tests for the new module |
| `README.md`, `README.zh.md`, `CLAUDE.md` | Modify | Document the new entry point + .env setup |

`daily_alert.py` is sized for one-file orchestration (~350 lines projected). It imports `screener` and `cross_section` directly.

---

## Task 1: Bootstrap module + .env.example + .gitignore

**Files:**
- Create: `src/daily_alert.py`
- Create: `.env.example`
- Modify: `.gitignore`

This task establishes the module skeleton, ships an env template, and makes sure local `.env` files cannot be committed. After this task, `python3 src/daily_alert.py --dry-run` runs without error and prints a placeholder message.

- [ ] **Step 1: Add `.env` to `.gitignore`**

Append a line to `/home/yixiang/jojo_quant/.gitignore` (currently contains `__pycache__/`, `*.pyc`, `*.pyo`, `data/`, `logs/`):

```
.env
```

- [ ] **Step 2: Create `.env.example` at repo root**

```
# Telegram bot credentials for src/daily_alert.py
# Get TOKEN from @BotFather on Telegram.
# Get CHAT_ID by adding the bot to your chat and calling
#   https://api.telegram.org/bot<TOKEN>/getUpdates
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

- [ ] **Step 3: Create `src/daily_alert.py` with module docstring, constants, and CLI skeleton**

```python
"""Daily Telegram alert for jojo signals filtered by cross-section top-30.

Workflow (runs after each US trading-day close):
1. Load .env / env vars for Telegram credentials.
2. Load the latest reports/cross_section_*.csv (monthly cadence).
3. Recompute today's SPX 9-bucket regime; abort if SPX data not yet updated.
4. Download today's OHLC for the cross-section universe (~960 tickers).
5. Detect S1 (overbought momentum) and S2 (oversold reversal) signals.
6. Keep only signals where the ticker is in the top-30 of the matching
   (strategy, regime) row in the CSV.
7. Format a Chinese natural-language MarkdownV2 message with FMP company
   info per ticker, send to the configured Telegram group.

If both strategies have zero qualifying alerts, exit silently with code 0.

Usage:
    python3 src/daily_alert.py                  # default: send Telegram
    python3 src/daily_alert.py --dry-run        # build + print, no POST
    python3 src/daily_alert.py --top 50         # override top-N filter
    python3 src/daily_alert.py --skip-fresh-check
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import cross_section
import screener
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
ENV_PATH = REPO_ROOT / ".env"

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
TELEGRAM_MAX_LEN = 3900   # below the 4096 hard cap, leaves margin

TOP_N_DEFAULT = 30
MIN_TRADES_DEFAULT = 5

US_BDAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())

EXIT_OK = 0
EXIT_SPX_STALE = 1
EXIT_TELEGRAM = 2
EXIT_CONFIG = 3


def main() -> int:
    parser = argparse.ArgumentParser(description="jojo daily Telegram alert")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the rendered message; do not POST to Telegram")
    parser.add_argument("--top", type=int, default=TOP_N_DEFAULT,
                        help="Top-N cutoff for ranking filter")
    parser.add_argument("--skip-fresh-check", action="store_true",
                        help="Bypass SPX freshness gate (testing only)")
    args = parser.parse_args()

    print("daily_alert.py — placeholder; functions added in subsequent tasks")
    print(f"  args = {vars(args)}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Smoke-run the skeleton**

Run: `python3 src/daily_alert.py --dry-run`

Expected output:
```
daily_alert.py — placeholder; functions added in subsequent tasks
  args = {'dry_run': True, 'top': 30, 'skip_fresh_check': False}
```

Exit code: 0.

- [ ] **Step 5: Commit**

```bash
git add .gitignore .env.example src/daily_alert.py
git commit -m "Scaffold daily_alert.py + .env.example"
```

---

## Task 2: `load_env` — read env vars with `.env` fallback

**Files:**
- Modify: `src/daily_alert.py`
- Modify: `src/test_logic.py`

Reads `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` from process environment first, falling back to `<repo>/.env` if either is missing. Pure-Python parser; no new dependencies.

- [ ] **Step 1: Add the failing test**

Append to `src/test_logic.py`, before `if __name__ == "__main__":`:

```python
# ============================================================
# Test: daily_alert.load_env reads from env then .env file
# ============================================================
def test_load_env_reads_dotenv():
    """load_env() prefers process env, falls back to .env file."""
    import tempfile
    from pathlib import Path
    import daily_alert

    # Case 1: both from environment
    os.environ["TELEGRAM_BOT_TOKEN"] = "envtok"
    os.environ["TELEGRAM_CHAT_ID"] = "envchat"
    try:
        token, chat_id = daily_alert.load_env()
        assert token == "envtok" and chat_id == "envchat"
    finally:
        del os.environ["TELEGRAM_BOT_TOKEN"]
        del os.environ["TELEGRAM_CHAT_ID"]

    # Case 2: env missing, .env file provides them
    with tempfile.TemporaryDirectory() as td:
        original = daily_alert.ENV_PATH
        daily_alert.ENV_PATH = Path(td) / ".env"
        daily_alert.ENV_PATH.write_text(
            'TELEGRAM_BOT_TOKEN=filetok\nTELEGRAM_CHAT_ID="filechat"\n'
        )
        try:
            os.environ.pop("TELEGRAM_BOT_TOKEN", None)
            os.environ.pop("TELEGRAM_CHAT_ID", None)
            token, chat_id = daily_alert.load_env()
            assert token == "filetok"
            assert chat_id == "filechat"
        finally:
            daily_alert.ENV_PATH = original

    print("  PASS: load_env reads env then .env")
```

Append to the `tests` list in `__main__`:

```python
        ("daily_alert load_env", test_load_env_reads_dotenv),
```

Add `import os` near the top of `test_logic.py` if not already present (it is — confirm).

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -c "from test_logic import test_load_env_reads_dotenv; test_load_env_reads_dotenv()"`

(From inside the `src/` directory, or with `PYTHONPATH=src`.)

Expected: `AttributeError: module 'daily_alert' has no attribute 'load_env'` or `ENV_PATH`.

- [ ] **Step 3: Implement `load_env`**

Add to `src/daily_alert.py`, after the constants block:

```python
# ---------------------------------------------------------------------------
# Env loading
# ---------------------------------------------------------------------------

def load_env() -> tuple[str, str]:
    """Return (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID).

    Order: process environment first, then .env file fallback.
    Raises RuntimeError if either is still missing afterwards.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if (not token or not chat_id) and ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k == "TELEGRAM_BOT_TOKEN" and not token:
                token = v
            elif k == "TELEGRAM_CHAT_ID" and not chat_id:
                chat_id = v

    if not token or not chat_id:
        raise RuntimeError(
            "Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID. "
            "Set them in the environment or in <repo>/.env."
        )
    return token, chat_id
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -c "from test_logic import test_load_env_reads_dotenv; test_load_env_reads_dotenv()"`

Expected: `PASS: load_env reads env then .env`.

- [ ] **Step 5: Commit**

```bash
git add src/daily_alert.py src/test_logic.py
git commit -m "Add load_env with .env fallback"
```

---

## Task 3: `load_latest_cross_section_csv` — find newest report

**Files:**
- Modify: `src/daily_alert.py`
- Modify: `src/test_logic.py`

Globs `reports/cross_section_*.csv`, returns the lexicographically last (filenames are date-stamped `YYYY-MM-DD`, so lexicographic == chronological).

- [ ] **Step 1: Add the failing test**

Append to `src/test_logic.py`:

```python
# ============================================================
# Test: load_latest_cross_section_csv picks newest by filename
# ============================================================
def test_load_latest_cross_section_csv_picks_newest():
    """When multiple cross_section_*.csv exist, the latest filename wins."""
    import tempfile
    from pathlib import Path
    import daily_alert

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        original = daily_alert.REPORTS_DIR
        daily_alert.REPORTS_DIR = d
        try:
            (d / "cross_section_2025-12-01.csv").write_text(
                "ticker,strategy,regime,trades,win_rate,total_pnl,avg_pnl,pf,max_dd,avg_holding,score\n"
                "AAA,S1,bull_low_vol,10,60,50,5,2.5,5,8,7.9\n"
            )
            (d / "cross_section_2026-04-29.csv").write_text(
                "ticker,strategy,regime,trades,win_rate,total_pnl,avg_pnl,pf,max_dd,avg_holding,score\n"
                "BBB,S2,bear_high_vol,8,55,30,3.7,2.0,4,7,5.7\n"
            )
            df, name = daily_alert.load_latest_cross_section_csv()
            assert name == "cross_section_2026-04-29"
            assert "BBB" in df["ticker"].values
            assert "AAA" not in df["ticker"].values
        finally:
            daily_alert.REPORTS_DIR = original

    print("  PASS: load_latest_cross_section_csv picks newest")
```

Append to `tests` list:

```python
        ("daily_alert load_latest_csv", test_load_latest_cross_section_csv_picks_newest),
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -c "from test_logic import test_load_latest_cross_section_csv_picks_newest; test_load_latest_cross_section_csv_picks_newest()"`

Expected: `AttributeError: module 'daily_alert' has no attribute 'load_latest_cross_section_csv'`.

- [ ] **Step 3: Implement**

Add to `src/daily_alert.py`, below `load_env`:

```python
# ---------------------------------------------------------------------------
# Cross-section CSV
# ---------------------------------------------------------------------------

def load_latest_cross_section_csv() -> tuple[pd.DataFrame, str]:
    """Return (DataFrame, stem) for the newest reports/cross_section_*.csv.

    Filenames are date-stamped (cross_section_YYYY-MM-DD.csv) so a plain
    lexicographic sort == chronological.
    """
    files = sorted(REPORTS_DIR.glob("cross_section_*.csv"))
    if not files:
        raise RuntimeError(
            f"No cross_section_*.csv in {REPORTS_DIR}. "
            "Run `python3 src/cross_section.py` first."
        )
    latest = files[-1]
    df = pd.read_csv(latest)
    return df, latest.stem
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -c "from test_logic import test_load_latest_cross_section_csv_picks_newest; test_load_latest_cross_section_csv_picks_newest()"`

Expected: `PASS: load_latest_cross_section_csv picks newest`.

- [ ] **Step 5: Commit**

```bash
git add src/daily_alert.py src/test_logic.py
git commit -m "Add load_latest_cross_section_csv"
```

---

## Task 4: `filter_top30` — top-N tickers per (strategy, regime)

**Files:**
- Modify: `src/daily_alert.py`
- Modify: `src/test_logic.py`

Filters the CSV down to the strategy and regime, drops `trades < min_trades` and `pf == "inf"` rows (perfect-record entries live in a separate listing), sorts by `score` (then `total_pnl`, then `win_rate`), and returns the top-N tickers as a `set[str]`.

- [ ] **Step 1: Add the failing test**

Append to `src/test_logic.py`:

```python
# ============================================================
# Test: filter_top30 excludes low-trades and inf-pf rows
# ============================================================
def test_filter_top30_excludes_low_trades_and_inf_pf():
    """trades < min_trades and pf == 'inf' rows are excluded; sort by score desc."""
    import daily_alert
    csv_df = pd.DataFrame([
        {"ticker": "GOOD", "strategy": "S1", "regime": "bull_low_vol",
         "trades": 10, "win_rate": 60.0, "total_pnl": 50.0, "avg_pnl": 5.0,
         "pf": 2.5, "max_dd": 5.0, "avg_holding": 8.0, "score": 7.9},
        {"ticker": "BEST", "strategy": "S1", "regime": "bull_low_vol",
         "trades": 20, "win_rate": 70.0, "total_pnl": 200.0, "avg_pnl": 10.0,
         "pf": 4.0, "max_dd": 5.0, "avg_holding": 7.0, "score": 17.9},
        {"ticker": "LOW", "strategy": "S1", "regime": "bull_low_vol",
         "trades": 3, "win_rate": 100.0, "total_pnl": 30.0, "avg_pnl": 10.0,
         "pf": 5.0, "max_dd": 0.0, "avg_holding": 5.0, "score": 8.66},
        {"ticker": "PERF", "strategy": "S1", "regime": "bull_low_vol",
         "trades": 8, "win_rate": 100.0, "total_pnl": 80.0, "avg_pnl": 10.0,
         "pf": "inf", "max_dd": 0.0, "avg_holding": 5.0, "score": "inf"},
        {"ticker": "OTHR", "strategy": "S2", "regime": "bull_low_vol",
         "trades": 30, "win_rate": 65.0, "total_pnl": 100.0, "avg_pnl": 3.3,
         "pf": 3.0, "max_dd": 5.0, "avg_holding": 6.0, "score": 16.4},
    ])
    keep = daily_alert.filter_top30(csv_df, strategy="S1",
                                    regime="bull_low_vol", n=30)
    assert keep == {"GOOD", "BEST"}, f"got {keep}"

    # Top-1 cap
    keep1 = daily_alert.filter_top30(csv_df, strategy="S1",
                                     regime="bull_low_vol", n=1)
    assert keep1 == {"BEST"}, f"top1 should be BEST, got {keep1}"

    print("  PASS: filter_top30 excludes low-trades + inf pf, sorts by score")
```

Append to `tests` list:

```python
        ("daily_alert filter_top30", test_filter_top30_excludes_low_trades_and_inf_pf),
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -c "from test_logic import test_filter_top30_excludes_low_trades_and_inf_pf; test_filter_top30_excludes_low_trades_and_inf_pf()"`

Expected: `AttributeError: module 'daily_alert' has no attribute 'filter_top30'`.

- [ ] **Step 3: Implement**

Add to `src/daily_alert.py`, below `load_latest_cross_section_csv`:

```python
def filter_top30(csv_df: pd.DataFrame, *, strategy: str, regime: str,
                 n: int = TOP_N_DEFAULT,
                 min_trades: int = MIN_TRADES_DEFAULT) -> set[str]:
    """Return the set of tickers ranked in the top-N of (strategy, regime).

    Filters:
      - row.strategy == strategy AND row.regime == regime
      - trades >= min_trades
      - pf finite (rows where pf == 'inf' belong to the perfect-record
        listing, not the main rank — exclude them here to match the
        cross-section markdown report's behaviour)

    Sort key: score desc, total_pnl desc, win_rate desc.
    """
    sub = csv_df[(csv_df["strategy"] == strategy) & (csv_df["regime"] == regime)]
    sub = sub[sub["trades"] >= min_trades]
    sub = sub[sub["pf"].apply(lambda v: not (isinstance(v, str) and v == "inf"))]
    if sub.empty:
        return set()
    sub = sub.copy()
    sub["pf_num"] = pd.to_numeric(sub["pf"], errors="coerce")
    sub = sub.sort_values(["score", "total_pnl", "win_rate"],
                          ascending=[False, False, False])
    return set(sub.head(n)["ticker"].astype(str).tolist())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -c "from test_logic import test_filter_top30_excludes_low_trades_and_inf_pf; test_filter_top30_excludes_low_trades_and_inf_pf()"`

Expected: `PASS: filter_top30 excludes low-trades + inf pf, sorts by score`.

- [ ] **Step 5: Commit**

```bash
git add src/daily_alert.py src/test_logic.py
git commit -m "Add filter_top30 ranking lookup"
```

---

## Task 5: SPX freshness gate — `expected_last_us_trading_day` + `check_spx_fresh`

**Files:**
- Modify: `src/daily_alert.py`
- Modify: `src/test_logic.py`

Decides what date the most recent US trading day should be (using `USFederalHolidayCalendar`), and aborts via `sys.exit(EXIT_SPX_STALE)` if SPX's last bar is older than that.

- [ ] **Step 1: Add the failing test**

Append to `src/test_logic.py`:

```python
# ============================================================
# Test: expected_last_us_trading_day returns a US business day
# ============================================================
def test_expected_last_us_trading_day_returns_business_day():
    """Helper returns a US business day on or before today."""
    import daily_alert
    d = daily_alert.expected_last_us_trading_day()
    today = pd.Timestamp.utcnow().normalize().tz_localize(None)
    assert d <= today, f"{d} should be <= today ({today})"
    # A US business day per the calendar
    assert daily_alert.US_BDAY.is_on_offset(d), (
        f"{d} should land on a US business day"
    )
    print(f"  PASS: expected_last_us_trading_day = {d.date()}")


# ============================================================
# Test: check_spx_fresh aborts when last bar is too old
# ============================================================
def test_check_spx_fresh_aborts_when_stale():
    """check_spx_fresh sys.exits when SPX last bar precedes the expected day."""
    import daily_alert

    # Stale SPX: last bar 30 days before today
    stale = pd.DataFrame(
        {"close": [100.0]},
        index=pd.DatetimeIndex(
            [pd.Timestamp.utcnow().normalize().tz_localize(None) - pd.Timedelta(days=30)]
        ),
    )
    try:
        daily_alert.check_spx_fresh(stale)
    except SystemExit as e:
        assert e.code == daily_alert.EXIT_SPX_STALE
        print("  PASS: check_spx_fresh aborts on stale data")
        return
    raise AssertionError("check_spx_fresh did not abort on stale SPX")
```

Append to `tests` list:

```python
        ("daily_alert expected_last_us_trading_day", test_expected_last_us_trading_day_returns_business_day),
        ("daily_alert check_spx_fresh stale", test_check_spx_fresh_aborts_when_stale),
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -c "from test_logic import test_expected_last_us_trading_day_returns_business_day; test_expected_last_us_trading_day_returns_business_day()"`
Expected: `AttributeError: module 'daily_alert' has no attribute 'expected_last_us_trading_day'`.

Run: `python3 -c "from test_logic import test_check_spx_fresh_aborts_when_stale; test_check_spx_fresh_aborts_when_stale()"`
Expected: similar AttributeError on `check_spx_fresh`.

- [ ] **Step 3: Implement**

Add to `src/daily_alert.py`, below `filter_top30`:

```python
# ---------------------------------------------------------------------------
# SPX freshness gate
# ---------------------------------------------------------------------------

def expected_last_us_trading_day() -> pd.Timestamp:
    """Most recent US trading day on or before today (UTC, midnight)."""
    today = pd.Timestamp.utcnow().normalize().tz_localize(None)
    if US_BDAY.is_on_offset(today):
        return today
    return (today - US_BDAY).normalize()


def check_spx_fresh(spx: pd.DataFrame) -> None:
    """Exit with EXIT_SPX_STALE if SPX last bar precedes the expected day."""
    if spx is None or spx.empty:
        print("[ABORT] SPX DataFrame empty; cannot evaluate freshness.")
        sys.exit(EXIT_SPX_STALE)
    expected = expected_last_us_trading_day()
    last = spx.index[-1].normalize()
    if last < expected:
        print(
            f"[ABORT] SPX last bar {last.date()} < expected "
            f"{expected.date()}. yfinance not updated yet; exiting. "
            "Re-run later."
        )
        sys.exit(EXIT_SPX_STALE)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -c "from test_logic import test_expected_last_us_trading_day_returns_business_day; test_expected_last_us_trading_day_returns_business_day()"`
Expected: `PASS: expected_last_us_trading_day = <some date>`.

Run: `python3 -c "from test_logic import test_check_spx_fresh_aborts_when_stale; test_check_spx_fresh_aborts_when_stale()"`
Expected: `PASS: check_spx_fresh aborts on stale data`.

- [ ] **Step 5: Commit**

```bash
git add src/daily_alert.py src/test_logic.py
git commit -m "Add SPX freshness gate (expected_last_us_trading_day + check_spx_fresh)"
```

---

## Task 6: `compute_today_regime`

**Files:**
- Modify: `src/daily_alert.py`

Thin wrapper that re-fetches SPX (forcing freshness via `max_staleness_days=0`), runs the gate, and returns today's 9-bucket regime label plus its date. No new test — covered by integration smoke later.

- [ ] **Step 1: Implement**

Add to `src/daily_alert.py`, below `check_spx_fresh`:

```python
def compute_today_regime() -> tuple[str, pd.Timestamp]:
    """Return (regime_label, date) for today using cross_section's classifier.

    Forces a yfinance re-fetch (max_staleness_days=0) so each daily run sees
    the latest SPX bar. After fetching, runs check_spx_fresh which will
    sys.exit(EXIT_SPX_STALE) if data is still behind.
    """
    spx = cross_section.load_or_fetch_spx(max_staleness_days=0)
    check_spx_fresh(spx)
    regimes = cross_section.build_regimes(spx)
    last_date = regimes.index[-1]
    return str(regimes.loc[last_date, "regime"]), last_date
```

- [ ] **Step 2: Smoke check**

Run:
```bash
python3 -c "
import sys
sys.path.insert(0, 'src')
import daily_alert
regime, date = daily_alert.compute_today_regime()
print(f'regime={regime} date={date.date()}')
"
```

Expected: prints something like `regime=bull_low_vol date=2026-04-29`. If yfinance is up-to-date the call succeeds. If yfinance is laggy, it sys.exits with code 1 — that's the desired behaviour.

- [ ] **Step 3: Commit**

```bash
git add src/daily_alert.py
git commit -m "Add compute_today_regime (with freshness gate)"
```

---

## Task 7: `get_today_signals`

**Files:**
- Modify: `src/daily_alert.py`

Reuses `screener.scan_signals` against the cross-section universe.

- [ ] **Step 1: Implement**

Add to `src/daily_alert.py`, below `compute_today_regime`:

```python
# ---------------------------------------------------------------------------
# Signal generation (reuses screener)
# ---------------------------------------------------------------------------

def get_today_signals(universe: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download today's data for `universe` and run screener.scan_signals.

    Returns (s1_df, s2_df) of today's S1 and S2 signals (cols match
    screener.scan_signals output: ticker, date, close, jojo, prev,
    plus atr_pct for S1 and recent_low for S2).
    """
    print(f"Downloading {len(universe)} tickers (last 120 days)...")
    all_data = screener.download_ohlc(universe, days=120, batch_size=200)
    print("Scanning signals...")
    s1, s2 = screener.scan_signals(all_data, strategy="all")
    return s1, s2
```

- [ ] **Step 2: Smoke check**

Run:
```bash
python3 -c "
import sys
sys.path.insert(0, 'src')
import daily_alert
s1, s2 = daily_alert.get_today_signals(['NVDA', 'TSLA', 'AAPL'])
print('S1 cols:', list(s1.columns))
print('S2 cols:', list(s2.columns))
print('S1 rows:', len(s1), '| S2 rows:', len(s2))
"
```

Expected: cols match the spec (`ticker, date, close, jojo, prev, atr_pct` for s1; `ticker, date, close, jojo, prev, recent_low` for s2). Row counts may be 0 — that's fine.

- [ ] **Step 3: Commit**

```bash
git add src/daily_alert.py
git commit -m "Add get_today_signals (reuses screener)"
```

---

## Task 8: `fetch_company_info`

**Files:**
- Modify: `src/daily_alert.py`

Per-ticker FMP profile lookup with safe fallbacks for commodity futures and FMP failures. No new test — too dependent on network — but smoke-check it.

- [ ] **Step 1: Implement**

Add to `src/daily_alert.py`, below `get_today_signals`:

```python
# ---------------------------------------------------------------------------
# Company info
# ---------------------------------------------------------------------------

def fetch_company_info(ticker: str) -> dict:
    """Return {name, sector, industry, description} for a ticker.

    Commodity futures use a hardcoded fallback (CN_NAMES + Commodities/Futures).
    Stocks query FMP profile API. Network or 4xx/5xx failures degrade to
    {ticker, '', '', ''} — the alert never blocks on FMP issues.
    """
    if ticker in screener.EXTRA_TICKERS_SET:
        return {
            "name": screener.CN_NAMES.get(ticker, ticker),
            "sector": "Commodities",
            "industry": "Futures",
            "description": "",
        }
    try:
        resp = requests.get(
            screener.FMP_PROFILE_URL,
            params={"symbol": ticker, "apikey": screener.FMP_API_KEY},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data:
                item = data[0]
                desc = (item.get("description") or "").strip()
                if len(desc) > 280:
                    desc = desc[:277] + "..."
                return {
                    "name": item.get("companyName") or ticker,
                    "sector": item.get("sector", "") or "",
                    "industry": item.get("industry", "") or "",
                    "description": desc,
                }
    except Exception as e:
        print(f"  [WARN] FMP fetch {ticker}: {e}")
    return {"name": ticker, "sector": "", "industry": "", "description": ""}
```

- [ ] **Step 2: Smoke check**

Run:
```bash
python3 -c "
import sys
sys.path.insert(0, 'src')
import daily_alert
info = daily_alert.fetch_company_info('NVDA')
print(info['name'], '|', info['sector'], '/', info['industry'])
print(info['description'][:100])
"
```

Expected: prints "NVIDIA Corporation | Technology / Semiconductors" (or similar) + a short description. If FMP is down, prints "NVDA | / " — also acceptable.

- [ ] **Step 3: Commit**

```bash
git add src/daily_alert.py
git commit -m "Add fetch_company_info (FMP + commodity fallback)"
```

---

## Task 9: MarkdownV2 escape + regime descriptor + alert formatter

**Files:**
- Modify: `src/daily_alert.py`
- Modify: `src/test_logic.py`

`_md_escape` is the load-bearing helper; an unescaped `.` or `(` in any interpolated value breaks Telegram with HTTP 400. Test it explicitly.

- [ ] **Step 1: Add the failing test**

Append to `src/test_logic.py`:

```python
# ============================================================
# Test: _md_escape covers all Telegram MarkdownV2 specials
# ============================================================
def test_md_escape_special_chars():
    """_md_escape escapes every Telegram MarkdownV2 special character."""
    import daily_alert
    s = "BRK.B (test) +1 -bar_value!"
    out = daily_alert._md_escape(s)
    for c in "._()+!-":
        assert "\\" + c in out, f"missing escape for {c!r} in {out!r}"
    # No escape for plain alphanumerics
    for c in "BRK Btest1bar":
        assert c in out, f"plain char {c!r} should pass through"
    print(f"  PASS: _md_escape produced {out!r}")
```

Append to `tests` list:

```python
        ("daily_alert _md_escape", test_md_escape_special_chars),
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -c "from test_logic import test_md_escape_special_chars; test_md_escape_special_chars()"`

Expected: `AttributeError: module 'daily_alert' has no attribute '_md_escape'`.

- [ ] **Step 3: Implement**

Add to `src/daily_alert.py`, below `fetch_company_info`:

```python
# ---------------------------------------------------------------------------
# MarkdownV2 helpers
# ---------------------------------------------------------------------------

_MD_SPECIAL = r"_*[]()~`>#+-=|{}.!\\"


def _md_escape(s) -> str:
    """Backslash-escape Telegram MarkdownV2 specials in `s`."""
    text = "" if s is None else str(s)
    return "".join("\\" + c if c in _MD_SPECIAL else c for c in text)


def _describe_regime(regime: str) -> str:
    """Plain-language Chinese description of a regime label."""
    parts = regime.split("_")
    trend = parts[0]
    vol = "_".join(parts[1:]) if len(parts) > 1 else ""
    trend_cn = {
        "bull": "多头趋势",
        "bear": "空头趋势",
        "neutral": "趋势中性",
    }.get(trend, trend)
    vol_cn = {
        "low_vol": "波动率低",
        "mid_vol": "波动率中等",
        "high_vol": "波动率高",
    }.get(vol, vol)
    return f"(SPX 处于{trend_cn}, {vol_cn})"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -c "from test_logic import test_md_escape_special_chars; test_md_escape_special_chars()"`

Expected: `PASS: _md_escape produced 'BRK\\.B \\(test\\) \\+1 \\-bar\\_value\\!'`.

- [ ] **Step 5: Commit**

```bash
git add src/daily_alert.py src/test_logic.py
git commit -m "Add MarkdownV2 escape + regime descriptor"
```

---

## Task 10: `format_message` (per-alert + full message)

**Files:**
- Modify: `src/daily_alert.py`
- Modify: `src/test_logic.py`

Builds the full Chinese MarkdownV2 message body. Receives a list of dicts per strategy with all fields needed (ticker, name, sector, industry, description, jojo numbers, regime, historical metrics).

- [ ] **Step 1: Implement `_format_alert` and `format_message`**

Add to `src/daily_alert.py`, below `_describe_regime`:

```python
# ---------------------------------------------------------------------------
# Message rendering
# ---------------------------------------------------------------------------

def _format_alert(a: dict, *, strategy: str) -> list[str]:
    """Return the lines (already MarkdownV2-escaped) for one alert entry."""
    parts: list[str] = []
    parts.append(f"*{_md_escape(a['ticker'])}* — {_md_escape(a['name'])}")
    if a.get("sector") or a.get("industry"):
        parts.append(
            f"板块: {_md_escape(a.get('sector', ''))} / "
            f"{_md_escape(a.get('industry', ''))}"
        )
    if a.get("description"):
        parts.append(_md_escape(a["description"]))

    # Pre-escape numeric fields into plain locals so we never nest f-strings
    # with conflicting quote characters (Python 3.9 forbids that).
    jojo_str = _md_escape(f"{a['jojo']:.1f}")
    prev_str = _md_escape(f"{a['prev']:.1f}")
    if strategy == "S1":
        atr_str = _md_escape(f"{a['atr_pct']:.1f}")
        parts.append(
            f"今日 jojo 上穿 76 \\(今 {jojo_str} / 昨 {prev_str}\\), "
            f"ATR {atr_str}%\\."
        )
    else:  # S2
        low_str = _md_escape(f"{a['recent_low']:.1f}")
        parts.append(
            f"今日 jojo 在 28 以下拐头 \\(今 {jojo_str} / 昨 {prev_str}\\), "
            f"20 日低 {low_str}\\."
        )

    bt_pf = a["bt_pf"]
    if isinstance(bt_pf, float) and bt_pf == float("inf"):
        pf_str = "inf"
    else:
        pf_str = f"{float(bt_pf):.2f}"
    pf_esc = _md_escape(pf_str)
    win_str = _md_escape(f"{a['bt_win_rate']:.1f}")
    pnl_str = _md_escape(f"{a['bt_total_pnl']:+.1f}")
    hold_str = _md_escape(f"{a['bt_avg_holding']:.1f}")
    parts.append(
        f"历史 `{_md_escape(a['regime'])}` 表现: "
        f"{int(a['bt_trades'])} 笔, 胜率 {win_str}%, "
        f"总收益 {pnl_str}%, "
        f"PF {pf_esc}, 平均持仓 {hold_str} 天\\."
    )
    return parts


def format_message(s1_alerts: list[dict], s2_alerts: list[dict],
                   regime: str, date_str: str) -> str:
    """Build the full Chinese MarkdownV2 message. Returns '' if no alerts."""
    if not s1_alerts and not s2_alerts:
        return ""
    lines: list[str] = []
    lines.append(f"🔔 *jojo 信号提醒* — {_md_escape(date_str)}")
    lines.append("")
    lines.append(f"📊 *当前市场环境*: `{_md_escape(regime)}`")
    lines.append(f"   {_md_escape(_describe_regime(regime))}")
    lines.append("")

    if s1_alerts:
        lines.append("═══════════════════")
        lines.append("🚀 *策略 1 \\(超买动量\\) 信号*")
        lines.append("═══════════════════")
        lines.append("")
        for a in s1_alerts:
            lines.extend(_format_alert(a, strategy="S1"))
            lines.append("")

    if s2_alerts:
        lines.append("═══════════════════")
        lines.append("🔄 *策略 2 \\(超卖反转\\) 信号*")
        lines.append("═══════════════════")
        lines.append("")
        for a in s2_alerts:
            lines.extend(_format_alert(a, strategy="S2"))
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"
```

- [ ] **Step 2: Add the test**

Append to `src/test_logic.py`:

```python
# ============================================================
# Test: format_message returns '' when no alerts; non-empty otherwise
# ============================================================
def test_format_message_empty_and_populated():
    """format_message: zero alerts → '', has alerts → contains ticker."""
    import daily_alert
    assert daily_alert.format_message([], [], "bull_low_vol", "2026-04-29") == ""

    s1 = [{
        "ticker": "NVDA", "name": "NVIDIA Corp", "sector": "Tech",
        "industry": "Semis", "description": "GPU maker",
        "jojo": 78.2, "prev": 75.4, "atr_pct": 4.3,
        "regime": "bull_low_vol",
        "bt_trades": 41, "bt_win_rate": 48.8, "bt_total_pnl": 131.1,
        "bt_pf": 2.88, "bt_avg_holding": 12.8,
    }]
    out = daily_alert.format_message(s1, [], "bull_low_vol", "2026-04-29")
    assert "NVDA" in out
    assert "策略 1" in out
    assert "策略 2" not in out  # only S1 alerts
    assert "*NVDA*" in out
    print("  PASS: format_message handles empty + populated")
```

Append to `tests` list:

```python
        ("daily_alert format_message", test_format_message_empty_and_populated),
```

- [ ] **Step 3: Run test to verify it passes**

Run: `python3 -c "from test_logic import test_format_message_empty_and_populated; test_format_message_empty_and_populated()"`

Expected: `PASS: format_message handles empty + populated`.

- [ ] **Step 4: Commit**

```bash
git add src/daily_alert.py src/test_logic.py
git commit -m "Add format_message + per-alert formatter"
```

---

## Task 11: Telegram sender + chunking

**Files:**
- Modify: `src/daily_alert.py`

`send_telegram` does the POST with one retry on 5xx / network errors. `send_chunked` splits messages above `TELEGRAM_MAX_LEN` along newline boundaries.

- [ ] **Step 1: Implement**

Add to `src/daily_alert.py`, below `format_message`:

```python
# ---------------------------------------------------------------------------
# Telegram delivery
# ---------------------------------------------------------------------------

def send_telegram(token: str, chat_id: str, text: str, *,
                  retries: int = 1, parse_mode: str = "MarkdownV2") -> None:
    """POST to Telegram Bot API. Retries once on 5xx / network errors.

    Raises RuntimeError on permanent failure (non-200 final attempt).
    """
    url = TELEGRAM_API.format(token=token)
    payload = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode}

    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=15)
        except requests.RequestException as e:
            if attempt < retries:
                time.sleep(2)
                continue
            raise RuntimeError(f"Telegram network error: {e}") from e

        if resp.status_code == 200:
            return
        if resp.status_code >= 500 and attempt < retries:
            time.sleep(2)
            continue
        raise RuntimeError(
            f"Telegram send failed {resp.status_code}: {resp.text[:300]}"
        )


def send_chunked(token: str, chat_id: str, text: str,
                 *, max_len: int = TELEGRAM_MAX_LEN) -> None:
    """Split `text` along newline boundaries and send each chunk."""
    if len(text) <= max_len:
        send_telegram(token, chat_id, text)
        return
    chunks: list[str] = []
    cur = ""
    for line in text.split("\n"):
        candidate = (cur + "\n" + line) if cur else line
        if len(candidate) > max_len and cur:
            chunks.append(cur)
            cur = line
        else:
            cur = candidate
    if cur:
        chunks.append(cur)
    for c in chunks:
        send_telegram(token, chat_id, c)
```

- [ ] **Step 2: Smoke check (no test — needs network + token)**

This step verifies Telegram delivery once `.env` is configured locally. Skip if no `.env` is present.

```bash
# With a real .env present at repo root:
python3 -c "
import sys; sys.path.insert(0, 'src')
import daily_alert
token, chat = daily_alert.load_env()
daily_alert.send_telegram(token, chat, '\\\\[smoke test\\\\] daily\\\\_alert\\\\.py is wired up\\\\.')
print('sent')
"
```

Expected: a message arrives in the configured Telegram group; script prints `sent`. If `.env` is missing, the call to `load_env` raises with a clear message — also acceptable, just rerun once `.env` is in place.

- [ ] **Step 3: Commit**

```bash
git add src/daily_alert.py
git commit -m "Add Telegram sender + chunking"
```

---

## Task 12: Wire `main()` end-to-end

**Files:**
- Modify: `src/daily_alert.py`
- Modify: `src/test_logic.py`

Replace the placeholder `main()` with the full pipeline. Add a final test asserting silent zero-alert behaviour by injecting empty signal frames through the formatter path.

- [ ] **Step 1: Implement `main()`**

Replace the placeholder `main()` in `src/daily_alert.py` with:

```python
def _build_alert_record(row: pd.Series, *, strategy: str, regime: str,
                       csv_df: pd.DataFrame) -> dict | None:
    """Combine signal row + CSV historical metrics + FMP info → alert dict.

    Returns None if no matching CSV row is found (defensive — should not
    happen because the universe is the CSV's tickers).
    """
    ticker = str(row["ticker"])
    sub = csv_df[
        (csv_df["ticker"] == ticker)
        & (csv_df["strategy"] == strategy)
        & (csv_df["regime"] == regime)
    ]
    if sub.empty:
        return None
    csv_row = sub.iloc[0]
    info = fetch_company_info(ticker)
    rec: dict = {
        "ticker": ticker,
        "name": info["name"],
        "sector": info["sector"],
        "industry": info["industry"],
        "description": info["description"],
        "jojo": float(row["jojo"]),
        "prev": float(row["prev"]),
        "regime": regime,
        "bt_trades": int(csv_row["trades"]),
        "bt_win_rate": float(csv_row["win_rate"]),
        "bt_total_pnl": float(csv_row["total_pnl"]),
        "bt_avg_holding": float(csv_row["avg_holding"]),
    }
    pf = csv_row["pf"]
    rec["bt_pf"] = float("inf") if (isinstance(pf, str) and pf == "inf") else float(pf)
    if strategy == "S1":
        rec["atr_pct"] = float(row["atr_pct"])
    else:
        rec["recent_low"] = float(row["recent_low"])
    return rec


def main() -> int:
    parser = argparse.ArgumentParser(description="jojo daily Telegram alert")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the rendered message; do not POST to Telegram")
    parser.add_argument("--top", type=int, default=TOP_N_DEFAULT,
                        help="Top-N cutoff for ranking filter")
    parser.add_argument("--skip-fresh-check", action="store_true",
                        help="Bypass SPX freshness gate (testing only)")
    args = parser.parse_args()

    # 1. CSV
    print("[1/6] Loading latest cross_section CSV...")
    try:
        csv_df, csv_name = load_latest_cross_section_csv()
    except RuntimeError as e:
        print(f"  [ERROR] {e}")
        return EXIT_CONFIG
    print(f"  Using {csv_name}.csv ({len(csv_df)} rows)")

    # 2. Regime + freshness
    print("\n[2/6] Computing today's SPX regime...")
    if args.skip_fresh_check:
        spx = cross_section.load_or_fetch_spx()
        regimes = cross_section.build_regimes(spx)
        last_date = regimes.index[-1]
        regime = str(regimes.loc[last_date, "regime"])
    else:
        regime, last_date = compute_today_regime()
    print(f"  Regime: {regime}  |  date: {last_date.date()}")

    # 3. Universe
    universe = sorted(csv_df["ticker"].unique().tolist())
    print(f"\n[3/6] Universe: {len(universe)} tickers from CSV")

    # 4. Today's signals
    print("\n[4/6] Scanning today's signals...")
    s1, s2 = get_today_signals(universe)
    print(f"  S1 raw: {len(s1)}  |  S2 raw: {len(s2)}")

    # 5. Filter to top-30
    print(f"\n[5/6] Filtering by top-{args.top} of cross_section[{regime}]...")
    s1_top = filter_top30(csv_df, strategy="S1", regime=regime, n=args.top)
    s2_top = filter_top30(csv_df, strategy="S2", regime=regime, n=args.top)
    s1_alerts: list[dict] = []
    for _, row in s1.iterrows():
        if str(row["ticker"]) in s1_top:
            rec = _build_alert_record(row, strategy="S1", regime=regime,
                                      csv_df=csv_df)
            if rec is not None:
                s1_alerts.append(rec)
    s2_alerts: list[dict] = []
    for _, row in s2.iterrows():
        if str(row["ticker"]) in s2_top:
            rec = _build_alert_record(row, strategy="S2", regime=regime,
                                      csv_df=csv_df)
            if rec is not None:
                s2_alerts.append(rec)
    print(f"  S1 filtered: {len(s1_alerts)}  |  S2 filtered: {len(s2_alerts)}")

    # 6. Send
    print("\n[6/6] Building + sending message...")
    message = format_message(s1_alerts, s2_alerts, regime,
                             str(last_date.date()))
    if not message:
        print("  No qualifying alerts. Exiting silently (exit 0).")
        return EXIT_OK

    if args.dry_run:
        print("--- BEGIN MESSAGE (dry run) ---")
        print(message)
        print("--- END MESSAGE ---")
        return EXIT_OK

    try:
        token, chat_id = load_env()
    except RuntimeError as e:
        print(f"  [ERROR] {e}")
        return EXIT_CONFIG

    try:
        send_chunked(token, chat_id, message)
    except RuntimeError as e:
        print(f"  [ERROR] {e}")
        return EXIT_TELEGRAM

    print("  Message sent.")
    return EXIT_OK
```

- [ ] **Step 2: Add the silent-on-zero-alerts test**

Append to `src/test_logic.py`:

```python
# ============================================================
# Test: format_message of empty alerts → '' (silent path)
# ============================================================
def test_main_silent_on_zero_alerts():
    """Empty S1+S2 alert lists yield empty message → main() exits silently.

    We test the formatter directly (the silent path is just
    `if not message: return EXIT_OK`).
    """
    import daily_alert
    msg = daily_alert.format_message([], [], "bull_low_vol", "2026-04-29")
    assert msg == ""
    print("  PASS: zero-alert format_message returns '' (main exits silent)")
```

Append to `tests` list:

```python
        ("daily_alert silent on zero alerts", test_main_silent_on_zero_alerts),
```

- [ ] **Step 3: Run the full test suite**

Run: `python3 src/test_logic.py`

Expected: all 15 prior tests + 8 new daily_alert tests pass (`Results: 23 passed, 0 failed`).

- [ ] **Step 4: Smoke run end-to-end with `--dry-run --skip-fresh-check`**

Run: `python3 src/daily_alert.py --dry-run --skip-fresh-check`

Expected:
- Six numbered phase headers `[1/6]` ... `[6/6]`.
- Either prints `No qualifying alerts. Exiting silently (exit 0).` (likely on most days) or `--- BEGIN MESSAGE (dry run) ---` followed by the rendered Chinese message.
- Exit code 0.

If the run fails because there is no `reports/cross_section_*.csv`, the script prints a clear error and returns exit code 3. That is the correct user-facing error; do not patch around it.

- [ ] **Step 5: Commit**

```bash
git add src/daily_alert.py src/test_logic.py
git commit -m "Wire daily_alert main() end-to-end"
```

---

## Task 13: Documentation updates

**Files:**
- Modify: `README.md`
- Modify: `README.zh.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add a Daily-alert section to `README.md`**

Insert immediately **after** the `## Cross-section backtest (cross_section.py)` section (and before `## reports/ policy`). Use this content verbatim:

```markdown
## Daily alert (`daily_alert.py`)

After each US trading day's close, sends a Telegram message listing today's jojo Strategy 1 / Strategy 2 signals — but only for tickers ranked in the top-30 of the latest cross-section CSV for the current 9-bucket SPX regime.

```bash
# Default: scan, filter, send Telegram (silent if no qualifying signals)
python3 src/daily_alert.py

# Build the message and print it; do not POST to Telegram
python3 src/daily_alert.py --dry-run

# Override top-N cutoff (default 30)
python3 src/daily_alert.py --top 50
```

### Setup

1. Create a Telegram bot via `@BotFather` and add it to your target chat.
2. Copy `.env.example` to `.env` and fill in `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` (`.env` is gitignored).
3. Make sure a recent `reports/cross_section_*.csv` exists; the alert reads the latest one.
4. Optional: configure cron to run after each US close. Beijing 09:00 Tue–Sat (machine TZ = `Asia/Shanghai`):

   ```
   0 9 * * 2-6 cd /home/yixiang/jojo_quant && /usr/bin/python3 src/daily_alert.py >> logs/daily_alert.log 2>&1
   ```

   If the machine is on UTC, use `0 1 * * 2-6` (UTC 01:00 = Beijing 09:00).

### Behaviour notes

- If yfinance has not yet published today's SPX bar, the script exits with code 1 and sends nothing — re-run later or rely on the next cron tick.
- If both strategies have zero qualifying signals, the script exits silently (no "no signals" message — quiet days stay quiet).
- Company info comes live from FMP (one request per alerted ticker); commodity futures use a hardcoded name fallback.
```

- [ ] **Step 2: Mirror the section into `README.zh.md`**

Insert immediately after `## 横截面回测 (cross_section.py)` (and before `## reports/ 维护策略`). Use this content verbatim:

```markdown
## 每日提醒 (daily_alert.py)

每个美股交易日收盘后，把当日 jojo Strategy 1 / Strategy 2 信号过滤到 cross-section 当前 9 桶 regime 下 top-30 的票，通过 Telegram 推送。

```bash
# 默认：扫描 + 过滤 + 发 Telegram（无信号则静默退出）
python3 src/daily_alert.py

# 仅本地预览消息，不发 Telegram
python3 src/daily_alert.py --dry-run

# 调整 top-N（默认 30）
python3 src/daily_alert.py --top 50
```

### 配置

1. 通过 `@BotFather` 创建 Telegram bot，加入目标群。
2. 复制 `.env.example` 为 `.env` 并填入 `TELEGRAM_BOT_TOKEN` 与 `TELEGRAM_CHAT_ID`（`.env` 已 gitignored）。
3. 确认 `reports/cross_section_*.csv` 存在；脚本读最新一个。
4. 可选：cron 配置为每个美股收盘后跑。机器时区 = `Asia/Shanghai` 时北京 09:00 Tue–Sat：

   ```
   0 9 * * 2-6 cd /home/yixiang/jojo_quant && /usr/bin/python3 src/daily_alert.py >> logs/daily_alert.log 2>&1
   ```

   机器时区 = UTC 时改为 `0 1 * * 2-6`（UTC 01:00 = 北京 09:00）。

### 行为说明

- 若 yfinance 当日 SPX 数据未更新，脚本以退出码 1 中止，不发消息——稍后手动重跑或等下一次 cron。
- 若 S1 和 S2 都没有符合 top-30 的信号，脚本静默退出（不发"无信号"消息——安静日就让它安静）。
- 公司信息实时从 FMP 拉取（每个 alert ticker 一次请求）；商品期货使用本地硬编码名称回退。
```

- [ ] **Step 3: Update `CLAUDE.md`**

Insert this command snippet block after the existing "### Cross-section backtest" section and before "### Tests and debugging":

```markdown
### Daily Telegram alert

```bash
# Default: scan, filter, send Telegram (silent if no qualifying signals)
python3 src/daily_alert.py

# Dry-run (build + print message only)
python3 src/daily_alert.py --dry-run

# Override top-N cutoff
python3 src/daily_alert.py --top 50
```

Requires `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` in env or `<repo>/.env` (gitignored). Reads the latest `reports/cross_section_*.csv` for ranking. Aborts with exit code 1 if today's SPX bar is not yet on yfinance.
```

Insert a new row in the "Project files" table, after the row for `src/cross_section.py`:

```markdown
| `src/daily_alert.py` | Post-close Telegram alert filtering today's jojo signals by cross-section top-30 of the current regime |
```

Insert a new bullet in the "Architecture" block after the `cross_section.py` bullet:

```markdown
- **`src/daily_alert.py`** — daily post-close Telegram alert. Reuses `screener.scan_signals` for today's S1/S2 signals and `cross_section.{load_or_fetch_spx, build_regimes}` for today's 9-bucket regime; filters signals down to tickers in the top-30 of the latest `reports/cross_section_*.csv` row for that regime; renders a Chinese MarkdownV2 message with FMP company info and POSTs to Telegram. Aborts cleanly if SPX is not yet updated on yfinance.
```

In the "External services" block, add this bullet:

```markdown
- **Telegram Bot API:** `daily_alert.py` POSTs to `api.telegram.org`. Credentials read from environment or the gitignored `<repo>/.env`. Bot must be added to the target chat first; group chat IDs may be negative.
```

- [ ] **Step 4: Verify the diffs**

Run:
```bash
git diff -- README.md README.zh.md CLAUDE.md | head -200
```

Expected: only the new sections from Steps 1–3 appear; no other lines changed.

- [ ] **Step 5: Commit**

```bash
git add README.md README.zh.md CLAUDE.md
git commit -m "Document daily_alert.py in README + CLAUDE.md"
```

---

## Task 14: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Full test suite**

Run: `python3 src/test_logic.py`

Expected: `Results: N passed, 0 failed` where N == prior count + the 8 new daily_alert tests (`load_env`, `load_latest_csv`, `filter_top30`, `expected_last_us_trading_day`, `check_spx_fresh stale`, `_md_escape`, `format_message`, `silent on zero alerts`). Confirm each new test name appears in the per-test output.

- [ ] **Step 2: End-to-end dry run with the freshness gate active**

Run: `python3 src/daily_alert.py --dry-run`

Expected: completes in 1–3 minutes (universe download dominates). One of:

- `No qualifying alerts. Exiting silently (exit 0).` — most likely on a quiet trading day.
- `--- BEGIN MESSAGE (dry run) ---` followed by a non-empty Chinese MarkdownV2 message and `--- END MESSAGE ---`.
- `[ABORT] SPX last bar ... < expected ...` and exit code 1 — yfinance has not yet caught up; rerun later. This is correct behaviour, not a failure.

- [ ] **Step 3: Telegram delivery test (only if `.env` is configured)**

If `.env` exists with valid credentials and the bot is in the target chat, run:

```bash
python3 src/daily_alert.py
```

Expected: any of the three outcomes from Step 2; if a non-empty message is built, Telegram delivers it to the configured chat. Verify the message renders correctly in the Telegram client (no escape errors).

- [ ] **Step 4: Final branch summary**

Run:
```bash
git status
git log --oneline -15
```

Confirm:
- 14 clean commits (one per task) on `feat/daily-alerts`.
- `git status` clean.

The branch is ready for the user's chosen completion path (merge / PR / keep / discard).
