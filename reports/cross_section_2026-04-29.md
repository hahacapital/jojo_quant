# Cross-Section Report — 2026-04-29

## Universe

Stocks: 3  |  Commodities: 0  |  Min history: 10y

Effective period: 2008-01-02 → 2026-04-28


*Note*: Universe drawn from current Russell 1000 + S&P 500 membership snapshot, intersected with the local OHLC cache. Survivorship bias is acknowledged; delisted names are not in this table.


## Regime definitions

Each regime is `<trend>_<vol>` — combination of an SPX **trend state** and a **volatility bucket**. All inputs are computed using only data on or before each date (no look-ahead).


### Trend state (3 levels)

- **bull**    — SPX close ≥ SMA225 AND SMA50 ≥ SMA200
- **bear**    — SPX close < SMA225 AND SMA50 < SMA200
- **neutral** — mixed signals (price and 50/200 cross disagree)

### Volatility bucket (3 levels)

30-day realized log-return vol of SPX, annualized × √252, then ranked over a 5-year **rolling** window (percentile rank).

- **low_vol**  — rolling rank ≤ 33%
- **mid_vol**  — 33% < rolling rank ≤ 67%
- **high_vol** — rolling rank > 67%

**warmup**: rows where SPX history is too short for the longest window (SMA225 or 5-year vol rank). Trades whose entry-date falls in warmup are dropped from the analysis.


## Regime time distribution (trading days)

| regime | days |
|--------|------|
| bull_low_vol | 1898 |
| bull_mid_vol | 949 |
| bull_high_vol | 435 |
| neutral_low_vol | 63 |
| neutral_mid_vol | 156 |
| neutral_high_vol | 199 |
| bear_low_vol | 12 |
| bear_mid_vol | 124 |
| bear_high_vol | 492 |
| warmup | 281 |

## Strategy 1 — Overbought Momentum


### bull_low_vol  (top 3 by score, min trades=5)

| rank | ticker | trades | win% | total_pnl% | pf | max_dd% | score |
|------|--------|--------|------|------------|----|---------|-------|
| 1 | A | 43 | 48.8 | +33.6 | 1.62 | 21.4 | 10.64 |
| 2 | AAL | 38 | 44.7 | +33.7 | 1.37 | 29.6 | 8.47 |
| 3 | AA | 38 | 31.6 | +12.7 | 1.15 | 30.4 | 7.11 |

### bull_mid_vol  (top 3 by score, min trades=5)

| rank | ticker | trades | win% | total_pnl% | pf | max_dd% | score |
|------|--------|--------|------|------------|----|---------|-------|
| 1 | AA | 13 | 61.5 | +37.0 | 2.98 | 17.6 | 10.75 |
| 2 | A | 16 | 31.2 | +16.3 | 1.76 | 13.7 | 7.05 |
| 3 | AAL | 18 | 33.3 | -32.1 | 0.50 | 39.1 | 2.14 |

### bull_high_vol  (top 3 by score, min trades=5)

| rank | ticker | trades | win% | total_pnl% | pf | max_dd% | score |
|------|--------|--------|------|------------|----|---------|-------|
| 1 | A | 8 | 87.5 | +12.8 | 13.09 | 1.1 | 37.01 |
| 2 | AA | 12 | 25.0 | +19.5 | 1.49 | 15.9 | 5.14 |
| 3 | AAL | 11 | 36.4 | -16.6 | 0.52 | 26.2 | 1.73 |

### neutral_high_vol  (top 2 by score, min trades=5)

| rank | ticker | trades | win% | total_pnl% | pf | max_dd% | score |
|------|--------|--------|------|------------|----|---------|-------|
| 1 | AA | 6 | 16.7 | -33.2 | 0.09 | 29.6 | 0.22 |
| 2 | AAL | 6 | 0.0 | -43.5 | 0.00 | 37.3 | 0.00 |

### bear_high_vol  (top 2 by score, min trades=5)

| rank | ticker | trades | win% | total_pnl% | pf | max_dd% | score |
|------|--------|--------|------|------------|----|---------|-------|
| 1 | A | 11 | 36.4 | -7.2 | 0.73 | 13.6 | 2.41 |
| 2 | AAL | 7 | 42.9 | -15.8 | 0.40 | 17.2 | 1.05 |

## Strategy 2 — Oversold Reversal


### bull_low_vol  (top 3 by score, min trades=5)

| rank | ticker | trades | win% | total_pnl% | pf | max_dd% | score |
|------|--------|--------|------|------------|----|---------|-------|
| 1 | A | 42 | 69.0 | +22.3 | 1.33 | 31.8 | 8.59 |
| 2 | AAL | 62 | 37.1 | +10.5 | 1.06 | 53.7 | 8.34 |
| 3 | AA | 55 | 56.4 | -6.6 | 0.95 | 41.4 | 7.07 |

### bull_mid_vol  (top 3 by score, min trades=5)

| rank | ticker | trades | win% | total_pnl% | pf | max_dd% | score |
|------|--------|--------|------|------------|----|---------|-------|
| 1 | AA | 35 | 65.7 | +90.1 | 3.50 | 14.6 | 20.70 |
| 2 | A | 36 | 58.3 | +17.4 | 1.44 | 16.7 | 8.65 |
| 3 | AAL | 25 | 60.0 | -8.5 | 0.90 | 41.5 | 4.50 |

### bull_high_vol  (top 3 by score, min trades=5)

| rank | ticker | trades | win% | total_pnl% | pf | max_dd% | score |
|------|--------|--------|------|------------|----|---------|-------|
| 1 | AAL | 19 | 68.4 | +63.4 | 3.73 | 15.8 | 16.24 |
| 2 | A | 7 | 71.4 | +9.6 | 3.19 | 3.0 | 8.45 |
| 3 | AA | 13 | 38.5 | -10.4 | 0.78 | 29.0 | 2.81 |

### neutral_mid_vol  (top 2 by score, min trades=5)

| rank | ticker | trades | win% | total_pnl% | pf | max_dd% | score |
|------|--------|--------|------|------------|----|---------|-------|
| 1 | AA | 6 | 50.0 | +5.9 | 1.38 | 15.1 | 3.37 |
| 2 | A | 5 | 20.0 | -7.8 | 0.44 | 13.3 | 0.99 |

### neutral_high_vol  (top 3 by score, min trades=5)

| rank | ticker | trades | win% | total_pnl% | pf | max_dd% | score |
|------|--------|--------|------|------------|----|---------|-------|
| 1 | A | 13 | 69.2 | +30.4 | 3.08 | 12.2 | 11.11 |
| 2 | AA | 8 | 50.0 | -8.9 | 0.66 | 24.0 | 1.86 |
| 3 | AAL | 9 | 55.6 | -64.0 | 0.25 | 59.0 | 0.74 |

### bear_mid_vol  (top 1 by score, min trades=5)

| rank | ticker | trades | win% | total_pnl% | pf | max_dd% | score |
|------|--------|--------|------|------------|----|---------|-------|
| 1 | AAL | 6 | 33.3 | -24.3 | 0.27 | 30.6 | 0.67 |

### bear_high_vol  (top 3 by score, min trades=5)

| rank | ticker | trades | win% | total_pnl% | pf | max_dd% | score |
|------|--------|--------|------|------------|----|---------|-------|
| 1 | A | 17 | 52.9 | +34.9 | 2.68 | 12.1 | 11.04 |
| 2 | AAL | 19 | 52.6 | +40.2 | 2.34 | 11.8 | 10.22 |
| 3 | AA | 15 | 53.3 | +3.8 | 1.08 | 29.5 | 4.20 |

### Perfect-record entries (pf = inf)

| ticker | regime | trades | total_pnl% |
|--------|--------|--------|------------|
| AAL | neutral_mid_vol | 7 | +35.4 |
