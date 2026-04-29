# jojo_quant (韭韭量化) — jojo signal screening tool

> English README · [中文文档](README.zh.md)

Daily scan of NASDAQ + NYSE stocks and commodity futures, ranking signals from two strategies built on the **jojo** composite momentum indicator:

- **Strategy 1 (Overbought Momentum)** — buy when jojo crosses above 76; sell when it falls below 68 (ATR% ≥ 2.0 filter, 20% stop loss).
- **Strategy 2 (Oversold Reversal)** — buy when jojo turns up below 28; sell when it crosses above 51 or drops back below 28 (20% stop loss).

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Daily scan
python3 src/screener.py --strategy 1 --top 20   # Strategy 1
python3 src/screener.py --strategy 2 --top 20   # Strategy 2
python3 src/screener.py                          # Both strategies

# Historical backtest
python3 src/backtest.py TSLA NVDA HOOD --years 3
python3 src/backtest.py --csv your_data.csv --use-tv  # Use a TradingView export

# Generate the per-stock backtest report
python3 src/generate_report.py

# Cross-section backtest (rank stocks across 9 SPX trend × volatility regimes)
python3 src/cross_section.py
python3 src/cross_section.py --strategy 1 --no-push
python3 src/cross_section.py --limit 5 --no-push   # smoke test

# Fund-level backtest (Top-N portfolio)
python3 src/fund_backtest.py --strategy 1 --universe sp500+
python3 src/fund_backtest.py --compare --universe sp500+   # 4-way config comparison
```

> All Python source lives under `src/`. The repo root keeps only `jojo.pine`, the documentation files, and the data / reports / logs directories.

## Project layout

| Path | Description |
|------|-------------|
| `jojo.pine` | TradingView Pine Script v6 implementation, 1:1 with the Python code (kept at repo root) |
| `src/indicators.py` | Core jojo computation (pure pandas / numpy, no third-party TA library) |
| `src/screener.py` | Full-market scanner (stocks + futures), enriches each signal with overall + current-regime backtest stats |
| `src/backtest.py` | Historical backtest engine, with optimised variants (stop loss + trend filter + volatility filter) |
| `src/generate_report.py` | Batch backtest report generator (bull/bear regime split, pushes to GitHub + S3) |
| `src/cross_section.py` | Cross-section backtest: per-stock S1/S2 ranking across 9 SPX trend × vol regimes (GitHub push only) |
| `src/fund_backtest.py` | Top-N portfolio simulator with rolling-PF ranking, optional bear-market hedge / ATR%-based sizing |
| `src/data_loader.py` · `src/download_ohlc.py` | Local OHLC parquet cache (one file per ticker) |
| `src/test_logic.py` | The project's only test entry point (assert-based, no pytest configuration) |
| `data/` · `reports/` · `logs/` | Cache, generated reports, run logs (all gitignored) |
| `CLAUDE.md` · `README.md` · `README.zh.md` | Documentation (English + Chinese) |
| `requirements.txt` | Python dependencies |

## Scanner output fields

Each signal includes: basic info (ticker, English name, Chinese name, industry, market cap, last close, jojo value), full-history backtest metrics (trades, win rate, total PnL, profit factor, max drawdown), and **current-regime backtest metrics** — SPX vs SMA(225) determines bull/bear, and only the matching regime's stats are shown.

## Filters

- Stocks: market cap ≥ 1B USD, ETFs excluded.
- Commodity futures: no market-cap filter.

## Coverage

- **Stocks:** every NASDAQ + NYSE listing (~6000+).
- **Commodity futures:** Gold (`GC=F`), Silver (`SI=F`), Crude Oil (`CL=F`), Natural Gas (`NG=F`), Copper (`HG=F`), Platinum (`PL=F`).

## Fund backtest (`fund_backtest`)

Simulates a Top-N portfolio fund that scans daily for jojo buy/sell signals and manages positions automatically (entries, exits, stop loss).

### Core features

1. **Multiple ranking methods** — largest / mid / smallest market cap, jojo value, rolling profit factor.
2. **Historical S&P 500 membership** — reconstructs constituents from Wikipedia change tables to remove survivorship bias.
3. **SPX benchmark comparison** — alpha vs SPX, monthly heatmap, drawdown analysis.
4. **Optional features** — bear-market sizing reduction, ATR%-based dynamic sizing, configurable stop loss and position count.

### Backtest results (historical constituents, no survivorship bias)

| Configuration | Annual % | Max DD % | Sharpe |
|---------------|---------:|---------:|-------:|
| Top 10 by largest market cap | +7.3 | 25.3 | 0.55 |
| Top 10 by mid market cap | +1.8 | 34.9 | 0.19 |
| Top 10 by smallest market cap | +0.3 | 38.9 | 0.10 |
| SPX benchmark | +10.8 | 33.9 | 0.66 |

> **Note:** running with current constituents inflates small-cap returns dramatically due to survivorship bias (Top 1 smallest market cap drops from +38.5% to -13.7% once you switch). Always pass `--historical`.

```bash
# Compare ranking method × Top-N (recommended with --historical)
python3 src/fund_backtest.py --rank-compare --universe sp500+ --historical

# 4-way comparison (baseline / bear hedge / ATR sizing / both stacked)
python3 src/fund_backtest.py --compare --universe sp500+

# Single configuration
python3 src/fund_backtest.py --strategy 1 --universe sp500+ --max-positions 10 --rank-method mktcap --historical
```

## Cross-section backtest (`cross_section.py`)

Cross-stock backtest of jojo Strategy 1 / 2 across 9 market regimes (SPX trend × volatility quantile), surfacing **which stocks fit which strategy under which regime**.

```bash
# All strategies, push to GitHub (default)
python3 src/cross_section.py

# Strategy 1 only, no push
python3 src/cross_section.py --strategy 1 --no-push

# Tune thresholds
python3 src/cross_section.py --top 50 --min-trades 10

# Smoke test (first 5 tickers)
python3 src/cross_section.py --limit 5 --no-push
```

Outputs:

- `reports/cross_section_<date>.md` — top-N table per regime × strategy.
- `reports/cross_section_<date>.csv` — full per-(ticker, strategy, regime) aggregate.

Notes:

- Requires the local OHLC cache; build it first with `python3 src/download_ohlc.py --init`.
- The first run scrapes Russell 1000 + S&P 500 membership from Wikipedia into `data/index_members.json`.
- GitHub push only (no S3).

## jojo indicator deep dive

jojo is a **composite momentum oscillator** that normalises six sub-indicators to the 0–100 range, blends them with fixed weights, and EMA-smooths the result. The final value sits in 0–100:

- **> 76** — overbought zone (Strategy 1 buy line).
- **68** — Strategy 1 sell line.
- **51** — midline (Strategy 2 sell line).
- **< 28** — oversold zone (Strategy 2 buy zone).

### Formula

```
index_raw = RSI × 0.1 + WR × 0.2 + CMO × 0.1 + KD × 0.3 + TSI × 0.2 + ADXRSI × 0.1
jojo      = EMA(index_raw, 3)
```

### The six sub-indicators

#### 1. RSI — Relative Strength Index

- **Weight:** 10%
- **Parameter:** `length = 14`
- **Range:** 0–100
- **Meaning:** measures the relative strength of recent gains vs losses. RSI > 70 is conventionally overbought; < 30 is oversold.
- **Formula:**
  ```
  RS  = RMA(gains, 14) / RMA(losses, 14)
  RSI = 100 - 100 / (1 + RS)
  ```
  RMA is Wilder's smoothing (an EMA variant with `alpha = 1/length`).

#### 2. WR — Williams %R

- **Weight:** 20%
- **Parameter:** `length = 14`
- **Raw range:** -100–0 (standard Williams %R)
- **Normalised:** add 100 to map onto 0–100.
- **Meaning:** measures where the close sits inside the recent high/low range. Higher = closer to the top of the range (stronger).
- **Formula:**
  ```
  WR = -100 × (highest - close) / (highest - lowest) + 100
  ```
  where `highest` / `lowest` are the rolling 14-bar extrema.

> **Note:** after normalisation, WR is identical to Stochastic %K: `100 × (close - lowest) / (highest - lowest)`.

#### 3. CMO — Chande Momentum Oscillator

- **Weight:** 10%
- **Parameter:** `length = 14`
- **Raw range:** -100–100
- **Normalised:** `(CMO + 100) / 2` maps to 0–100.
- **Meaning:** similar to RSI, but directly uses the difference between gain and loss totals over the same window — more sensitive to momentum shifts.
- **Formula:**
  ```
  sum_gain = SUM(gains, 14)
  sum_loss = SUM(losses, 14)
  CMO      = 100 × (sum_gain - sum_loss) / (sum_gain + sum_loss)
  norm     = (CMO + 100) / 2
  ```

#### 4. KD — Stochastic %K

- **Weight:** 30% (highest weight)
- **Parameter:** `length = 14`
- **Range:** 0–100
- **Meaning:** position of the close inside the recent price channel. %K > 80 ≈ near the channel top (strong); < 20 ≈ near the bottom (weak). This is the highest-weighted sub-indicator and is the most sensitive to short-term price location changes.
- **Formula:**
  ```
  %K = 100 × (close - lowest_14) / (highest_14 - lowest_14)
  ```

#### 5. TSI — True Strength Index

- **Weight:** 20%
- **Parameters:** `short_length = 7, long_length = 14`
- **Raw range:** -1–1 (Pine Script's `ta.tsi()` output)
- **Normalised:** `(TSI + 1) / 2 × 100` maps to 0–100.
- **Meaning:** double-EMA-smoothed momentum; smoother than RSI, captures trend direction and strength while filtering short-term noise.
- **Formula:**
  ```
  diff              = close - close[1]                # daily price change
  double_smooth     = EMA(EMA(diff, 14), 7)           # double-smoothed change
  abs_double_smooth = EMA(EMA(|diff|, 14), 7)         # double-smoothed |change|
  TSI_raw           = double_smooth / abs_double_smooth   # [-1, 1]
  TSI               = (TSI_raw + 1) / 2 × 100             # [0, 100]
  ```

#### 6. ADXRSI — RSI of ADX (directional filter)

- **Weight:** 10%
- **Parameters:** `DI length = 14, ADX smoothing = 18, RSI length = 14`
- **Range:** 0–100
- **Meaning:** computes ADX (average directional index — trend strength), then RSI of ADX, then signs the result by candle direction (up vs down bar). This lets jojo distinguish ADX strength inside up-trends from down-trends.
- **Formula:**
  ```
  # 1. DMI (directional movement index)
  +DM  = max(high - high[1], 0)   if high move > low move
  -DM  = max(low[1] - low, 0)     if low move > high move
  ATR  = RMA(TrueRange, 14)
  +DI  = 100 × RMA(+DM, 14) / ATR
  -DI  = 100 × RMA(-DM, 14) / ATR

  # 2. ADX (average directional index)
  DX   = 100 × |+DI - -DI| / (+DI + -DI)
  ADX  = RMA(DX, 18)

  # 3. ADXRSI (directional adjustment)
  sign   = +1 (up bar) or -1 (down bar)
  ADXRSI = (RSI(ADX, 14) × sign + 100) / 2
  ```

### Smoothing

jojo uses two smoothing methods, both matching TradingView exactly:

| Method | Formula | Used in |
|--------|---------|---------|
| **RMA (Wilder smoothing)** | `rma[i] = val × (1/n) + rma[i-1] × (1 - 1/n)` | RSI gain/loss smoothing, ATR, DI, ADX |
| **EMA (exponential moving average)** | `ema[i] = val × (2/(n+1)) + ema[i-1] × (1 - 2/(n+1))` | TSI's double smoothing, final EMA(3) |

Both are seeded with the **simple mean** of the first `n` values — that seeding is the key to matching TradingView's results.

## Dependencies

- **yfinance** — bulk OHLC download from Yahoo Finance.
- **pandas** — data manipulation and time-series ops.
- **numpy** — numerical primitives.
- **requests** — pulls the NASDAQ-listed ticker file.
- **lxml** — backend for Wikipedia table parsing via `pd.read_html`.

> No `pandas_ta` or other TA libraries — every indicator is implemented from scratch.
