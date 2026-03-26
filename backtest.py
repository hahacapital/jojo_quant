"""
Backtest — test historical performance of 温度计 strategies.

Strategy 1 (超买动量):
    BUY  when thermometer crosses above 76
    SELL when thermometer drops below 68

Strategy 2 (超卖反转):
    BUY  when thermometer was below 28 and turns upward (today > yesterday)
    SELL when thermometer crosses above 51, OR drops below 28 again

Usage:
    # Backtest single stock
    python backtest.py TSLA

    # Backtest multiple stocks
    python backtest.py TSLA NVDA HOOD AAPL

    # Backtest with custom history period
    python backtest.py TSLA --years 3

    # Backtest from a TradingView CSV (uses 温度计 column directly)
    python backtest.py --csv "/tmp/BATS_TSLA, 1D_35f99.csv" --label TSLA
"""

import argparse
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf

from indicators import compute_thermometer


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """A single round-trip trade."""
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    holding_days: int
    pnl_pct: float  # percentage return
    exit_reason: str


@dataclass
class StrategyResult:
    """Aggregated backtest result for one strategy on one stock."""
    symbol: str
    strategy: str
    trades: list[Trade] = field(default_factory=list)

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.pnl_pct > 0)
        return wins / len(self.trades) * 100

    @property
    def avg_pnl(self) -> float:
        if not self.trades:
            return 0.0
        return np.mean([t.pnl_pct for t in self.trades])

    @property
    def total_pnl(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.pnl_pct for t in self.trades)

    @property
    def avg_holding(self) -> float:
        if not self.trades:
            return 0.0
        return np.mean([t.holding_days for t in self.trades])

    @property
    def min_holding(self) -> int:
        if not self.trades:
            return 0
        return min(t.holding_days for t in self.trades)

    @property
    def max_holding(self) -> int:
        if not self.trades:
            return 0
        return max(t.holding_days for t in self.trades)

    @property
    def max_win(self) -> float:
        if not self.trades:
            return 0.0
        return max(t.pnl_pct for t in self.trades)

    @property
    def max_loss(self) -> float:
        if not self.trades:
            return 0.0
        return min(t.pnl_pct for t in self.trades)

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl_pct for t in self.trades if t.pnl_pct > 0)
        gross_loss = abs(sum(t.pnl_pct for t in self.trades if t.pnl_pct < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def backtest_strategy1(dates, closes, therm_vals) -> list[Trade]:
    """
    Strategy 1 (超买动量):
        BUY:  thermometer crosses above 76 (today > 76 AND yesterday <= 76)
        SELL: thermometer drops below 68 (today < 68 AND yesterday >= 68)
    """
    trades = []
    in_position = False
    entry_date = None
    entry_price = None
    entry_idx = None

    for i in range(1, len(therm_vals)):
        t_today = therm_vals[i]
        t_yesterday = therm_vals[i - 1]

        if np.isnan(t_today) or np.isnan(t_yesterday):
            continue

        if not in_position:
            # BUY signal: cross above 76
            if t_today > 76 and t_yesterday <= 76:
                in_position = True
                entry_date = str(dates[i])
                entry_price = closes[i]
                entry_idx = i
        else:
            # SELL signal: drop below 68
            if t_today < 68 and t_yesterday >= 68:
                exit_price = closes[i]
                pnl = (exit_price / entry_price - 1) * 100
                trades.append(Trade(
                    entry_date=entry_date,
                    entry_price=round(entry_price, 2),
                    exit_date=str(dates[i]),
                    exit_price=round(exit_price, 2),
                    holding_days=i - entry_idx,
                    pnl_pct=round(pnl, 2),
                    exit_reason="下穿68",
                ))
                in_position = False

    # Close open position at last bar
    if in_position:
        exit_price = closes[-1]
        pnl = (exit_price / entry_price - 1) * 100
        trades.append(Trade(
            entry_date=entry_date,
            entry_price=round(entry_price, 2),
            exit_date=str(dates[-1]),
            exit_price=round(exit_price, 2),
            holding_days=len(therm_vals) - 1 - entry_idx,
            pnl_pct=round(pnl, 2),
            exit_reason="持仓中",
        ))

    return trades


def backtest_strategy2(dates, closes, therm_vals) -> list[Trade]:
    """
    Strategy 2 (超卖反转):
        BUY:  thermometer was below 28, then turns up (today > yesterday, yesterday < 28)
        SELL: thermometer crosses above 51 OR drops below 28 again
    """
    trades = []
    in_position = False
    entry_date = None
    entry_price = None
    entry_idx = None

    for i in range(1, len(therm_vals)):
        t_today = therm_vals[i]
        t_yesterday = therm_vals[i - 1]

        if np.isnan(t_today) or np.isnan(t_yesterday):
            continue

        if not in_position:
            # BUY: was below 28 and turning up
            if t_yesterday < 28 and t_today > t_yesterday:
                in_position = True
                entry_date = str(dates[i])
                entry_price = closes[i]
                entry_idx = i
        else:
            # SELL: cross above 51
            if t_today > 51 and t_yesterday <= 51:
                exit_price = closes[i]
                pnl = (exit_price / entry_price - 1) * 100
                trades.append(Trade(
                    entry_date=entry_date,
                    entry_price=round(entry_price, 2),
                    exit_date=str(dates[i]),
                    exit_price=round(exit_price, 2),
                    holding_days=i - entry_idx,
                    pnl_pct=round(pnl, 2),
                    exit_reason="上穿51",
                ))
                in_position = False
            # SELL: drop below 28 again (止损)
            elif t_today < 28 and t_yesterday >= 28:
                exit_price = closes[i]
                pnl = (exit_price / entry_price - 1) * 100
                trades.append(Trade(
                    entry_date=entry_date,
                    entry_price=round(entry_price, 2),
                    exit_date=str(dates[i]),
                    exit_price=round(exit_price, 2),
                    holding_days=i - entry_idx,
                    pnl_pct=round(pnl, 2),
                    exit_reason="再次下穿28",
                ))
                in_position = False

    # Close open position at last bar
    if in_position:
        exit_price = closes[-1]
        pnl = (exit_price / entry_price - 1) * 100
        trades.append(Trade(
            entry_date=entry_date,
            entry_price=round(entry_price, 2),
            exit_date=str(dates[-1]),
            exit_price=round(exit_price, 2),
            holding_days=len(therm_vals) - 1 - entry_idx,
            pnl_pct=round(pnl, 2),
            exit_reason="持仓中",
        ))

    return trades


# ---------------------------------------------------------------------------
# Run backtest on one stock
# ---------------------------------------------------------------------------

def run_backtest(symbol: str, df: pd.DataFrame, therm_vals: np.ndarray = None) -> tuple[StrategyResult, StrategyResult]:
    """
    Run both strategies on a single stock.

    Parameters
    ----------
    symbol : ticker symbol (for display)
    df : OHLC DataFrame
    therm_vals : optional pre-computed thermometer values (e.g. from TV CSV).
                 If None, computed from df using our formula.
    """
    if therm_vals is None:
        therm_vals = compute_thermometer(df).values

    closes = df["close"].astype(float).values
    dates = df.index if hasattr(df.index, 'strftime') else range(len(df))

    # Skip warmup period
    start = 0
    for i in range(len(therm_vals)):
        if not np.isnan(therm_vals[i]):
            start = i
            break

    dates = dates[start:]
    closes = closes[start:]
    therm_vals = therm_vals[start:]

    trades1 = backtest_strategy1(dates, closes, therm_vals)
    trades2 = backtest_strategy2(dates, closes, therm_vals)

    r1 = StrategyResult(symbol=symbol, strategy="超买动量 (76→68)", trades=trades1)
    r2 = StrategyResult(symbol=symbol, strategy="超卖反转 (28→51)", trades=trades2)

    return r1, r2


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_result(r: StrategyResult):
    """Pretty-print a strategy result."""
    print(f"\n{'=' * 65}")
    print(f"  {r.symbol} — {r.strategy}")
    print(f"{'=' * 65}")

    if r.num_trades == 0:
        print("  No trades.")
        return

    print(f"  Trades: {r.num_trades}  |  Win rate: {r.win_rate:.1f}%  "
          f"|  Avg PnL: {r.avg_pnl:+.2f}%  |  Total PnL: {r.total_pnl:+.2f}%")
    print(f"  Holding: avg {r.avg_holding:.1f}d / min {r.min_holding}d / max {r.max_holding}d  |  "
          f"Max win: {r.max_win:+.2f}%  |  Max loss: {r.max_loss:+.2f}%  |  "
          f"Profit factor: {r.profit_factor:.2f}")
    print()

    # Trade list
    rows = []
    for t in r.trades:
        rows.append({
            "entry_date": t.entry_date[:10],
            "entry_price": t.entry_price,
            "exit_date": t.exit_date[:10],
            "exit_price": t.exit_price,
            "days": t.holding_days,
            "pnl%": f"{t.pnl_pct:+.2f}",
            "exit_reason": t.exit_reason,
        })
    print(pd.DataFrame(rows).to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="温度计 backtest")
    parser.add_argument("symbols", nargs="*", help="Ticker symbols to backtest")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to TradingView CSV (uses 温度计 column directly)")
    parser.add_argument("--label", type=str, default=None,
                        help="Symbol label for CSV mode (default: filename)")
    parser.add_argument("--years", type=int, default=2,
                        help="Years of history for yfinance download (default: 2)")
    parser.add_argument("--use-tv", action="store_true",
                        help="When using --csv, use the 温度计 column instead of computing our own")
    args = parser.parse_args()

    if not args.symbols and not args.csv:
        parser.print_help()
        sys.exit(1)

    # --- CSV mode ---
    if args.csv:
        label = args.label or args.csv.split("/")[-1][:20]
        print(f"Loading CSV: {args.csv}")
        df = pd.read_csv(args.csv)

        # Normalize columns
        cols = [c.strip().lower() for c in df.columns]
        seen = {}
        for i, c in enumerate(cols):
            if c in seen:
                seen[c] += 1
                cols[i] = f"{c}_{seen[c]}"
            else:
                seen[c] = 0
        df.columns = cols

        therm_vals = None
        if args.use_tv and "温度计" in df.columns:
            therm_vals = df["温度计"].values
            print(f"  Using TV 温度计 column ({np.sum(~np.isnan(therm_vals))} values)")
        else:
            print("  Computing thermometer from OHLC...")

        r1, r2 = run_backtest(label, df, therm_vals)
        print_result(r1)
        print_result(r2)
        return

    # --- yfinance mode ---
    for sym in args.symbols:
        print(f"\nDownloading {sym} ({args.years}y history)...")
        try:
            df = yf.download(sym, period=f"{args.years}y", auto_adjust=True, progress=False)
            if df.empty or len(df) < 60:
                print(f"  [SKIP] Not enough data for {sym}")
                continue
            # yfinance may return MultiIndex columns even for single ticker
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel("Ticker", axis=1)
            df.columns = [c.lower() for c in df.columns]
            print(f"  Got {len(df)} bars ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")
        except Exception as e:
            print(f"  [ERROR] {sym}: {e}")
            continue

        r1, r2 = run_backtest(sym, df)
        print_result(r1)
        print_result(r2)


if __name__ == "__main__":
    main()
