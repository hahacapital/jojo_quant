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
