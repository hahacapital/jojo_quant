# Daily Alert — Design

**Date:** 2026-04-29
**Author:** brainstorm session
**Status:** Approved, awaiting implementation plan

## Goal

After each US trading day's close, automatically send a Telegram message listing today's jojo Strategy 1 / Strategy 2 signals — but **only** for tickers ranked in the top 30 of the latest cross-section report for the current 9-bucket SPX regime.

## Decisions

| Topic | Decision |
|-------|----------|
| Delivery channel | Telegram bot (group chat) |
| Ranking source | Latest `reports/cross_section_*.csv` (monthly cadence); SPX regime recomputed daily |
| Regime granularity | 9 buckets (3 trend × 3 vol), exact match (`bull_low_vol`, etc.) |
| Strategy filter | S1 signals filtered by S1 top-30; S2 signals by S2 top-30 |
| Trigger | Local cron, Beijing 09:00 Tue–Sat (after US trading day close) |
| Credentials storage | Env vars `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID`, with `.env` fallback (gitignored) and committed `.env.example` template |
| Universe for daily scan | Tickers present in latest cross-section CSV (~960 today) — limits download time |
| Freshness gate | If SPX last bar < expected last US trading day, exit non-zero without sending |
| Company info | FMP API (live, ~1–10 fetches/day, well under free-tier limit); commodities use hardcoded names |
| Message language | Natural Chinese, MarkdownV2 |

## Non-Goals

- Daily re-computation of cross-section (stays monthly).
- Multi-channel broadcast (Slack / email / SMS).
- Inline buttons or interactive bot UX.
- Auto-scheduling: cron config left to the user.
- Historical alert archive in git (logs cover this).
- Per-stock detailed charts.

## Architecture

### New file: `src/daily_alert.py`

Sibling to `screener.py` / `cross_section.py`. Reuses everything from those modules; adds Telegram delivery and the top-30 ranking lookup.

```
daily_alert.py
├── load_env()                       # read .env / env vars
├── load_latest_cross_section_csv()  # find latest reports/cross_section_*.csv
├── filter_top30(csv, strategy, regime, n=30, min_trades=5)
├── compute_today_regime()           # uses cross_section.load_or_fetch_spx + build_regimes
├── check_spx_fresh(spx)             # exit non-zero if data stale
├── expected_last_us_trading_day()   # USFederalHolidayCalendar-based
├── get_today_signals(universe)      # reuses screener.{download_ohlc, scan_signals}
├── fetch_company_info(ticker)       # FMP profile
├── _md_escape(s)                    # MarkdownV2 escape
├── _format_alert(...) / _describe_regime(...)
├── format_message(s1_alerts, s2_alerts, regime, date)
├── send_telegram(token, chat_id, text, *, retries=1)
├── send_chunked(...)                # split if > ~3900 chars
└── main()                           # CLI
```

### Reused modules (no changes)

- `screener.get_exchange_tickers`, `screener.download_ohlc`, `screener.scan_signals`
- `screener.EXTRA_TICKERS`, `screener.EXTRA_TICKERS_SET`, `screener.CN_NAMES`
- `screener.FMP_PROFILE_URL`, `screener.FMP_API_KEY`
- `cross_section.load_or_fetch_spx`, `cross_section.build_regimes`

### New non-code files

- `.env.example` — committed template (no real values).
- `.env` — local-only, in `.gitignore`.

### `.gitignore` change

Add `.env`.

## Data Flow

```
0. Cron fires Beijing 09:00 Tue–Sat
1. Load .env → (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
2. Load latest reports/cross_section_*.csv
3. SPX freshness gate:
     - load_or_fetch_spx(max_staleness_days=0)  # forces re-fetch
     - assert spx.index[-1] >= expected_last_us_trading_day()
     - if stale → log + sys.exit(1)
4. Compute today's regime (last row of build_regimes(spx))
5. Universe = unique tickers in cross_section CSV (~960)
6. Download today's data (screener.download_ohlc, days=120)
7. Detect signals (screener.scan_signals → s1_df, s2_df)
8. For each strategy:
     - top30 = filter_top30(csv, strategy, regime, n=30)
     - alerts = signals ∩ top30
9. For each alert: fetch_company_info + lookup the (ticker, strategy, regime)
   row from CSV for historical metrics
10. format_message → MarkdownV2 string (single message; chunk if > ~3900 chars)
11. send_telegram (with 1 retry on 5xx / network errors)
```

## Ranking lookup

CSV schema: `ticker, strategy, regime, trades, win_rate, total_pnl, avg_pnl, pf, max_dd, avg_holding, score`.

```python
def filter_top30(csv_df, *, strategy, regime, n=30, min_trades=5):
    sub = csv_df[(csv_df["strategy"] == strategy) & (csv_df["regime"] == regime)]
    sub = sub[sub["trades"] >= min_trades]
    sub = sub[sub["pf"].apply(lambda v: not (isinstance(v, str) and v == "inf"))]
    sub = sub.copy()
    sub["pf_num"] = sub["pf"].astype(float)
    sub = sub.sort_values(["score", "total_pnl", "win_rate"],
                          ascending=[False, False, False])
    return set(sub.head(n)["ticker"].tolist())
```

`pf == "inf"` rows live in the perfect-record listing, not the main rank — exclude them from the alert filter to mirror the markdown report's behaviour.

## Freshness gate

```python
US_BDAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())

def expected_last_us_trading_day():
    today = pd.Timestamp.utcnow().normalize().tz_localize(None)
    return today if US_BDAY.is_on_offset(today) else (today - US_BDAY)

def check_spx_fresh(spx):
    expected = expected_last_us_trading_day()
    last = spx.index[-1].normalize()
    if last < expected:
        print(f"[ABORT] SPX last bar {last.date()} < expected {expected.date()}.")
        sys.exit(1)
```

`load_or_fetch_spx(max_staleness_days=0)` ensures we re-fetch yfinance every run rather than rely on a stale cache.

## Signal generation (reuse)

```python
def get_today_signals():
    csv_df, _ = load_latest_cross_section_csv()
    universe = sorted(csv_df["ticker"].unique().tolist())  # ~960
    all_data = screener.download_ohlc(universe, days=120, batch_size=200)
    s1, s2 = screener.scan_signals(all_data, strategy="all")
    return s1, s2
```

The cross-section universe is a strict subset of what `screener.scan_signals` would otherwise scan; restricting download to that subset cuts time by ~10× and guarantees every detected signal can resolve in the ranking lookup.

## Company info

```python
def fetch_company_info(ticker):
    if ticker in screener.EXTRA_TICKERS_SET:
        return {
            "name": screener.CN_NAMES.get(ticker, ticker),
            "sector": "Commodities",
            "industry": "Futures",
            "description": "",
        }
    try:
        resp = requests.get(screener.FMP_PROFILE_URL,
                            params={"symbol": ticker, "apikey": screener.FMP_API_KEY},
                            timeout=10)
        if resp.status_code == 200 and resp.json():
            item = resp.json()[0]
            desc = (item.get("description") or "").strip()
            if len(desc) > 280:
                desc = desc[:277] + "..."
            return {
                "name": item.get("companyName") or ticker,
                "sector": item.get("sector", ""),
                "industry": item.get("industry", ""),
                "description": desc,
            }
    except Exception as e:
        print(f"  [WARN] FMP fetch {ticker}: {e}")
    return {"name": ticker, "sector": "", "industry": "", "description": ""}
```

Per-ticker FMP fetch with 10-second timeout; failures degrade to ticker-only display, never block the alert.

## Telegram delivery

Endpoint: `https://api.telegram.org/bot<TOKEN>/sendMessage`, JSON body `{"chat_id": "...", "text": "...", "parse_mode": "MarkdownV2"}`.

MarkdownV2 requires escaping `_ * [ ] ( ) ~ \` > # + - = | { } . !`. Every interpolated value (ticker, regime, jojo numbers, descriptions) goes through `_md_escape` before composition.

Length: single message limit 4096; we split at 3900 along newline boundaries to stay safe.

Retries: 1 retry on `requests.RequestException` or HTTP 5xx, with a 2-second backoff. 4xx (e.g. bad escape, bot not in group) raises immediately with the response body in the error.

Bot is already in the target group.

## Message format (Chinese natural language)

```
🔔 *jojo 信号提醒* — 2026-04-29

📊 *当前市场环境*: `bull_low_vol`
   (SPX 处于多头趋势, 波动率低)

═══════════════════
🚀 *策略 1 (超买动量) 信号*
═══════════════════

*NVDA* — NVIDIA Corporation
板块: Technology / Semiconductors
英伟达是图形处理器 (GPU) 龙头,业务覆盖游戏、数据中心、AI 加速、自动驾驶...
今日 jojo 上穿 76 (今 78.2 / 昨 75.4), ATR 4.3%.
历史 `bull_low_vol` 表现: 41 笔, 胜率 48.8%, 总收益 +131.1%, PF 2.88, 平均持仓 12.8 天.

═══════════════════
🔄 *策略 2 (超卖反转) 信号*
═══════════════════

(no signals → "今日策略 2 无符合 top-30 排名的信号。" omitted entirely if both empty)
```

If both strategies have zero alerts, the script exits silently (exit code 0). No "no signals" message is sent — Telegram inbox stays clean on quiet days.

## CLI

```bash
python3 src/daily_alert.py                  # default: send Telegram (silent if no signals)
python3 src/daily_alert.py --dry-run        # build message, print, do not POST
python3 src/daily_alert.py --top 50         # override top-N filter
python3 src/daily_alert.py --skip-fresh-check  # bypass SPX freshness gate (testing only)
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--dry-run` | False | Print the rendered message; no Telegram POST |
| `--top N` | 30 | Top-N cutoff for the ranking filter |
| `--skip-fresh-check` | False | Bypass the SPX freshness gate (testing only) |

## Exit codes

- `0` — normal (alert sent, or zero signals → silent exit).
- `1` — SPX freshness gate failed (data not yet updated).
- `2` — Telegram send failed (after retry).
- `3` — Misconfig (missing env vars, no cross_section CSV, etc.).

## Cron

User configures manually (we do not auto-install). Beijing time entry, assuming machine TZ = Asia/Shanghai:

```
0 9 * * 2-6 cd /home/yixiang/jojo_quant && /usr/bin/python3 src/daily_alert.py >> logs/daily_alert.log 2>&1
```

If machine TZ is UTC, use `0 1 * * 2-6` (UTC 01:00 = Beijing 09:00). Tue–Sat covers Mon–Fri US closes.

## Testing

Add to `src/test_logic.py`:

1. `test_filter_top30_excludes_low_trades_and_inf_pf` — ensures `trades < min_trades` and `pf == "inf"` rows are excluded.
2. `test_md_escape_special_chars` — `_md_escape` covers `._()+!` and other MarkdownV2 specials.
3. `test_expected_last_us_trading_day_returns_business_day` — sanity that the helper returns a US business day on or before today.
4. `test_load_env_reads_dotenv` — env-var override + `.env` fallback parsing.
5. `test_main_silent_on_zero_alerts` — when both S1/S2 alerts are empty, `main()` exits 0 without invoking `send_telegram`.

Smoke: `python3 src/daily_alert.py --dry-run --skip-fresh-check` produces a valid MarkdownV2 string and prints it.

## Documentation updates

- `README.md` / `README.zh.md` — add Daily alert section under cross-section.
- `CLAUDE.md` — add command snippet under "Available commands"; note `.env` setup.
- `.env.example` committed at repo root.

## Risks

1. **FMP rate limit**: free tier 250 req/day. With ~10 alerts/day this is well under, but failures during long alert days degrade gracefully.
2. **MarkdownV2 escape misses**: a missed special character returns HTTP 400 from Telegram. Mitigation: every interpolated value goes through `_md_escape`; the escape table is centrally defined; unit-tested.
3. **CSV stale beyond a month**: if cross-section monthly run is missed, alerts use older rankings. Mitigation: monthly cadence is documented; future enhancement could `print [WARN]` if CSV mtime older than 35 days.
4. **yfinance lag past 09:00 Beijing**: the freshness gate exits cleanly. The user can re-run manually or schedule a backup cron at e.g. Beijing 11:00 — out of scope for the initial implementation.
5. **Cross-section universe drift**: a stock newly added to R1000/SP500 won't appear until the next monthly cross-section run. This is acceptable given the cadence.
6. **No retry on transient yfinance failures during signal scan**: if `download_ohlc` flakes for some tickers, those signals are silently missed. `screener.scan_signals` already tolerates per-ticker failures; we accept this.

## Performance

- Phase 1 (universe + CSV load): <1s.
- Phase 2 (SPX fetch + regime): ~2s (single-ticker yfinance).
- Phase 3 (download 960 tickers × 120 days): ~60–120s.
- Phase 4 (scan_signals): ~30s.
- Phase 5 (FMP + format): <5s for ≤10 alerts.
- Phase 6 (Telegram send): <2s per chunk.

End-to-end target: under 3 minutes.

## Open Questions

None. All decisions confirmed.
