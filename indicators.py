"""
jojo 复合动量指标 — pure pandas/numpy implementation matching TradingView Pine Script.

No external TA library required. All indicators implemented from scratch using
Wilder's smoothing (RMA) and EMA to match TradingView's calculations.

Key: TradingView initializes EMA/RMA with SMA of the first `length` values.
Standard pandas ewm does NOT do this. We implement custom versions to match.
"""

import pandas as pd
import numpy as np


def _rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder's smoothing (RMA), matching Pine's ta.rma().
    Initialized with SMA of first `length` values, then:
        rma[i] = (source - rma[i-1]) * (1/length) + rma[i-1]
    """
    alpha = 1.0 / length
    vals = series.values.astype(float)
    result = np.full_like(vals, np.nan)

    # Find the first window of `length` non-NaN values for SMA seed
    count = 0
    seed_end = -1
    for i in range(len(vals)):
        if not np.isnan(vals[i]):
            count += 1
            if count == length:
                seed_end = i
                break
        else:
            count = 0

    if seed_end < 0:
        return pd.Series(result, index=series.index)

    seed_start = seed_end - length + 1
    result[seed_end] = np.mean(vals[seed_start:seed_end + 1])

    for i in range(seed_end + 1, len(vals)):
        if np.isnan(vals[i]):
            result[i] = result[i - 1]
        else:
            result[i] = alpha * vals[i] + (1 - alpha) * result[i - 1]

    return pd.Series(result, index=series.index)


def _ema(series: pd.Series, length: int) -> pd.Series:
    """EMA matching TradingView's ta.ema().
    Initialized with SMA of first `length` values, then:
        ema[i] = (source - ema[i-1]) * (2/(length+1)) + ema[i-1]
    """
    alpha = 2.0 / (length + 1)
    vals = series.values.astype(float)
    result = np.full_like(vals, np.nan)

    count = 0
    seed_end = -1
    for i in range(len(vals)):
        if not np.isnan(vals[i]):
            count += 1
            if count == length:
                seed_end = i
                break
        else:
            count = 0

    if seed_end < 0:
        return pd.Series(result, index=series.index)

    seed_start = seed_end - length + 1
    result[seed_end] = np.mean(vals[seed_start:seed_end + 1])

    for i in range(seed_end + 1, len(vals)):
        if np.isnan(vals[i]):
            result[i] = result[i - 1]
        else:
            result[i] = alpha * vals[i] + (1 - alpha) * result[i - 1]

    return pd.Series(result, index=series.index)


def _rsi(close: pd.Series, length: int) -> pd.Series:
    """RSI using Wilder's smoothing (RMA), matching Pine's ta.rsi()."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = _rma(gain, length)
    avg_loss = _rma(loss, length)
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def _willr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    """Williams %R, returns [-100, 0]. Pine's ta.wpr()."""
    highest = high.rolling(length).max()
    lowest = low.rolling(length).min()
    return -100 * (highest - close) / (highest - lowest)


def _cmo(close: pd.Series, length: int) -> pd.Series:
    """Chande Momentum Oscillator, returns [-100, 100]. Pine's ta.cmo()."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    sum_gain = gain.rolling(length).sum()
    sum_loss = loss.rolling(length).sum()
    return 100 * (sum_gain - sum_loss) / (sum_gain + sum_loss)


def _stoch(close: pd.Series, high: pd.Series, low: pd.Series, length: int) -> pd.Series:
    """Raw Stochastic %K (unsmoothed), returns [0, 100]. Pine's ta.stoch()."""
    lowest = low.rolling(length).min()
    highest = high.rolling(length).max()
    return 100 * (close - lowest) / (highest - lowest)


def _tsi(close: pd.Series, short_length: int, long_length: int) -> pd.Series:
    """True Strength Index matching Pine's ta.tsi().
    Pine returns [-1, 1]. We return the same scale: [-1, 1].
    """
    diff = close.diff()
    smooth1 = _ema(diff, long_length)
    double_smooth = _ema(smooth1, short_length)
    abs_smooth1 = _ema(diff.abs(), long_length)
    abs_double_smooth = _ema(abs_smooth1, short_length)
    return double_smooth / abs_double_smooth


def _dmi_adx(high: pd.Series, low: pd.Series, close: pd.Series,
             di_length: int, adx_length: int) -> pd.Series:
    """ADX from DMI, matching Pine's ta.dmi(di_length, adx_length).
    Returns only the ADX series."""
    up = high.diff()
    down = -low.diff()

    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0), index=high.index)

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = _rma(tr, di_length)
    plus_di = 100 * _rma(plus_dm, di_length) / atr
    minus_di = 100 * _rma(minus_dm, di_length) / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = _rma(dx, adx_length)
    return adx


def compute_jojo(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """
    Compute the jojo composite momentum indicator.

    Parameters
    ----------
    df : DataFrame with columns: open, high, low, close
    length : lookback period (default 14)

    Returns
    -------
    pd.Series : jojo index values [0, 100]
    """
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)

    # 1. RSI — [0, 100]
    value_rsi = _rsi(close, length)

    # 2. Williams %R + 100 — shift [-100, 0] to [0, 100]
    value_wr = _willr(high, low, close, length) + 100

    # 3. CMO normalized — shift [-100, 100] to [0, 100]
    value_cmo = (_cmo(close, length) + 100) / 2

    # 4. Stochastic %K (raw, unsmoothed) — [0, 100]
    value_kd = _stoch(close, high, low, length)

    # 5. TSI normalized — Pine returns [-1, 1], formula: (tsi+1)/2*100 => [0, 100]
    short_len = round(length / 2)  # math.round(14/2) = 7
    tsi_raw = _tsi(close, short_len, length)  # [-1, 1]
    value_tsi = (tsi_raw + 1) / 2 * 100

    # 6. ADX — Pine: ta.dmi(len, len+4)
    value_adx = _dmi_adx(high, low, close, di_length=length, adx_length=length + 4)

    # 7. adxrsi = (RSI(ADX, 14) * sign(close - open) + 100) / 2
    rsi_of_adx = _rsi(value_adx, 14)
    sign = np.sign(close.values - open_.values)
    adxrsi = (rsi_of_adx * sign + 100) / 2

    # 8. Weighted composite
    index_raw = (
        value_rsi * 0.1
        + value_wr * 0.2
        + value_cmo * 0.1
        + value_kd * 0.3
        + value_tsi * 0.2
        + adxrsi * 0.1
    )

    # 9. EMA(3) smoothing
    index = _ema(index_raw, 3)

    return index


