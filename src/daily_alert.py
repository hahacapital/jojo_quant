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
