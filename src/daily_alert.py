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
