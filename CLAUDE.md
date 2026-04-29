# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# jojo_quant (韭韭量化) — jojo signal screening tool

A full-market screening tool built on the **jojo** composite momentum indicator. The scan covers every NASDAQ + NYSE stock and a fixed list of commodity futures.

## Repo layout

All Python sources live under `src/`. The repo root keeps only:

- `jojo.pine` — TradingView Pine Script v6 reference implementation.
- `README.md` (English) and `README.zh.md` (Chinese mirror).
- `CLAUDE.md` (this file).
- `requirements.txt`.
- `data/`, `reports/`, `logs/` (gitignored).
- `docs/superpowers/specs/` and `docs/superpowers/plans/` for design/implementation docs.

When invoking any script, use `python3 src/<file>.py` from the repo root.

## Available commands

### Daily screen

```bash
# Strategy 1 (overbought momentum)
python3 src/screener.py --strategy 1

# Strategy 2 (oversold reversal)
python3 src/screener.py --strategy 2

# All strategies
python3 src/screener.py --strategy all

# Limit results
python3 src/screener.py --strategy 1 --top 20
```

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--strategy` | `all` | `1` = overbought momentum, `2` = oversold reversal, `all` = both |
| `--top N` | unbounded | show only top N rows |
| `--days N` | 120 | calendar days of history downloaded for signal detection |
| `--batch N` | 200 | tickers per yfinance batch |

### Historical backtest

```bash
# Backtest one or more tickers
python3 src/backtest.py TSLA NVDA HOOD --years 3

# Backtest from a TradingView CSV
python3 src/backtest.py --csv data.csv --use-tv --label TSLA
```

### Backtest report generator

```bash
# Generate the 13-stock report (auto-pushes GitHub + S3)
python3 src/generate_report.py

# Generate locally only
python3 src/generate_report.py --no-push --no-s3
```

> `generate_report.py` pushes to GitHub and S3 by default (see "External dependencies" below). Always pass `--no-push --no-s3` when credentials aren't set or you're debugging.

### Cross-section backtest

```bash
# All strategies, current universe (cache ∩ Russell 1000 ∪ S&P 500 ∩ ≥10y history)
python3 src/cross_section.py

# Single strategy, no push
python3 src/cross_section.py --strategy 1 --no-push

# Smoke test
python3 src/cross_section.py --limit 5 --no-push
```

### Tests and debugging

```bash
# Backtest logic assertions (assert-based, no pytest config; this is the project's only test entry point)
python3 src/test_logic.py

# Cross-check jojo values against a TradingView CSV
python3 src/validate.py

# Print each jojo sub-indicator (RSI / WR / CMO / KD / TSI / ADX) for a ticker
python3 src/debug_indicators.py
```

## Strategies

### Strategy 1 — Overbought Momentum
- **Buy:** jojo crosses above 76
- **Sell:** jojo drops below 68
- **Filter:** ATR%(14) ≥ 2.0 (skip low-volatility names)
- **Stop loss:** -20%
- **Best for:** high-volatility stocks (TSLA, NVDA, RKLB, etc.)

### Strategy 2 — Oversold Reversal
- **Buy:** jojo turns up while still below 28
- **Sell:** jojo crosses above 51, or drops back below 28
- **Stop loss:** -20%
- **Best for:** mean-reversion / oversold-bounce setups

## Output schema

Each signal row contains:

| Field | Meaning |
|-------|---------|
| ticker | Symbol |
| name | English company name |
| cn_name | Chinese name (where available) |
| industry | Industry classification |
| mkt_cap_fmt | Formatted market cap |
| close | Latest close |
| jojo | Current jojo value |
| atr_pct | ATR% (Strategy 1 only) |
| bt_trades | Historical backtest trade count (2009 → present) |
| bt_win_rate | Historical win rate % |
| bt_total_pnl | Historical total PnL % |
| bt_pf | Profit factor |
| bt_max_dd | Max drawdown % |
| `{regime}_trades` | Trade count in the **current** SPX regime |
| `{regime}_win%` | Win rate in the current regime |
| `{regime}_pnl%` | Total PnL in the current regime |
| `{regime}_pf` | Profit factor in the current regime |
| `{regime}_dd%` | Max drawdown in the current regime |

> **Regime detection (screener / generate_report):** SPX close ≥ SMA(225) → `bull`, otherwise `bear`. The `{regime}` placeholder in the output is replaced at runtime, so only the current regime's columns appear. `cross_section.py` uses a finer 9-bucket regime (see "Architecture" below).

## Filters

- Stocks: market cap ≥ 1B USD; ETFs excluded.
- Commodity futures: no market-cap filter.

## Coverage

- **Stocks:** all of NASDAQ + NYSE (~6000+).
- **Commodity futures:** Gold (`GC=F`), Silver (`SI=F`), Crude Oil (`CL=F`), Natural Gas (`NG=F`), Copper (`HG=F`), Platinum (`PL=F`).

## Architecture

Module responsibilities and data flow:

- **`src/indicators.py`** — pure pandas/numpy, no I/O. Exposes `compute_jojo(df)`; internally composes six sub-indicators (`_rsi` / `_willr` / `_cmo` / `_stoch` / `_tsi` / `_dmi_adx`) with `_rma` / `_ema` smoothing helpers. Reused by every other module.
- **`src/backtest.py`** — exposes `backtest_strategy1()` / `backtest_strategy2()` (numpy-vectorised simulation) and the orchestration helper `run_backtest()` (download → indicator → strategy → split metrics by market regime). Called by `screener.py` and `generate_report.py`.
- **`src/screener.py`** — daily-scan entry point: `yfinance` bulk OHLC download → `compute_jojo` → today's signal filter → `run_backtest()` adds historical metrics (full + current-regime subset) → ranked output.
- **`src/cross_section.py`** — cross-section backtest: `build_universe()` returns cache ∩ Wikipedia large-cap membership; `build_regimes()` computes SPX trend (SMA50/200/225) × 5-year rolling vol-rank → 9 buckets; each stock runs `run_backtest`, trades aggregate by entry-date regime, ranked by `score = pf × √trades`. GitHub push only.
- **Regime decision:** `^GSPC` close vs SMA(225) for the screener / generate_report path. `cross_section.py` uses a finer 9-bucket regime (3 trend states × 3 vol buckets).

## Project files

| Path | Description |
|------|-------------|
| `src/screener.py` | Full-market scanner (daily entry point) |
| `src/backtest.py` | Historical backtest engine (`run_backtest`, `backtest_strategy1/2`) |
| `src/indicators.py` | jojo indicator computation (pure pandas/numpy, no I/O) |
| `src/generate_report.py` | Batch backtest report generator (default: push GitHub + S3) |
| `src/cross_section.py` | Cross-section backtest: rank S1/S2 per stock across 9 SPX trend × vol regimes |
| `src/test_logic.py` | Assert-based backtest logic tests (project's only test entry point) |
| `src/validate.py` | Cross-checks jojo against TradingView CSVs |
| `src/debug_indicators.py` | Per-sub-indicator diagnostics |
| `src/data_loader.py` · `src/download_ohlc.py` | Local OHLC parquet cache (read + maintain) |
| `jojo.pine` | TradingView Pine Script reference implementation (kept at repo root) |

## Dependencies

```bash
pip install -r requirements.txt
```

### External services

- **GitHub:** `generate_report.py` and `cross_section.py` default to `git add/commit/push`; local git credentials must be configured.
- **S3:** `generate_report.py` uploads to `s3://staking-ledger-bpt/jojo_quant/reports/` (path hard-coded). Requires the AWS CLI and matching IAM credentials. `cross_section.py` does **not** use S3. Both scripts skip publishing with `--no-push` (and `--no-s3` for `generate_report.py`).
- **yfinance:** anonymous access, no API key.
- **FMP** (company profiles): rate-limited; failures degrade to empty profile and never block the pipeline.
- **Wikipedia:** scraped for Russell 1000 + S&P 500 membership by `cross_section.py` and cached into `data/index_members.json` after the first successful fetch.

## `reports/` policy

`reports/` is committed to git and serves as the published, browsable archive of full historical runs.

- **Only commit full-universe runs.** No `--limit` smoke tests, no debugging slices.
- **Update cadence:** monthly. Each month's full run replaces (or sits alongside) the prior month's file with a new dated filename.
- Smoke tests should always pass `--no-push` so they never reach `reports/` on disk for git, or they should be deleted before commit.
- If a partial / smoke-test report is committed by mistake, delete it from `reports/` and push the deletion. The data is reproducible from the cache.
