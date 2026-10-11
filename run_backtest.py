# ============================================================
# SMT + ORDER BLOCK BACKTEST v2
# SMT ke baad 1-10 candles ke andar OB aaye to trade
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_DIR = Path.home() / "gold_quant" / "data"
RESULTS_DIR = Path.home() / "gold_quant" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

LOOKBACK = 20
SMT_TO_OB_WINDOW = 10    # SMT ke baad kitni candles mein OB dhoondna
RR = 2.0
RISK_PCT = 0.005
STOP_ATR_MULT = 1.0
ACCOUNT = 10000
TIMEFRAME = "5m"


def load_data():
    gold_file = DATA_DIR / f"XAUUSD_{TIMEFRAME}.parquet"
    silver_file = DATA_DIR / f"XAGUSD_{TIMEFRAME}.parquet"

    if not gold_file.exists() or not silver_file.exists():
        print(f"X Data files nahi milin")
        return None

    gold = pd.read_parquet(gold_file)
    silver = pd.read_parquet(silver_file)

    df = pd.DataFrame({
        'gold': gold['close'],
        'silver': silver['close']
    }).dropna()

    df['tr'] = df['gold'].diff().abs()
    df['atr'] = df['tr'].rolling(14).mean()

    print(f"  Bars: {len(df):,}")
    print(f"  Range: {df.index.min()} to {df.index.max()}")
    return df


def find_smt(df, lookback=LOOKBACK):
    df = df.copy()
    df['gold_high'] = df['gold'].rolling(lookback).max()
    df['gold_low'] = df['gold'].rolling(lookback).min()
    df['silver_high'] = df['silver'].rolling(lookback).max()
    df['silver_low'] = df['silver'].rolling(lookback).min()

    df['bullish_smt'] = (
        (df['gold'] <= df['gold_low'].shift(1)) &
        (df['silver'] > df['silver_low'].shift(1))
    )
    df['bearish_smt'] = (
        (df['gold'] >= df['gold_high'].shift(1)) &
        (df['silver'] < df['silver_high'].shift(1))
    )
    return df


def find_ob(df):
    df = df.copy()
    df['candle_range'] = df['gold'].diff().abs()
    df['avg_range'] = df['candle_range'].rolling(20).mean()

    df['bullish_ob'] = (
        (df['gold'].shift(1) < df['gold'].shift(2)) &
        (df['gold'] > df['gold'].shift(1)) &
        ((df['gold'] - df['gold'].shift(1)) > df['avg_range'])
    )
    df['bearish_ob'] = (
        (df['gold'].shift(1) > df['gold'].shift(2)) &
        (df['gold'] < df['gold'].shift(1)) &
        ((df['gold'].shift(1) - df['gold']) > df['avg_range'])
    )
    return df


def combine_signals(df, window=SMT_TO_OB_WINDOW):
    """SMT ke baad window candles ke andar OB aaye to signal."""
    df = df.copy()
    df['long_signal'] = False
    df['short_signal'] = False

    # Bullish
    smt_indices = df.index[df['bullish_smt']].tolist()
    for smt_idx in smt_indices:
        pos = df.index.get_loc(smt_idx)
        for j in range(pos+1, min(pos+window+1, len(df))):
            if df['bullish_ob'].iloc[j]:
                df.iloc[j, df.columns.get_loc('long_signal')] = True
                break

    # Bearish
    smt_indices = df.index[df['bearish_smt']].tolist()
    for smt_idx in smt_indices:
        pos = df.index.get_loc(smt_idx)
        for j in range(pos+1, min(pos+window+1, len(df))):
            if df['bearish_ob'].iloc[j]:
                df.iloc[j, df.columns.get_loc('short_signal')] = True
                break

    return df


def backtest(df, rr=RR, stop_mult=STOP_ATR_MULT, risk_pct=RISK_PCT, account=ACCOUNT):
    trades = []
    in_trade = False
    entry_price = stop = target = direction = 0
    entry_time = None
    entry_idx = 0

    for i in range(1, len(df)):
        row = df.iloc[i]

        if in_trade:
            exit_price = None
            reason = None
            if direction == 1:
                if row['gold'] <= stop:
                    exit_price, reason = stop, "STOP"
                elif row['gold'] >= target:
                    exit_price, reason = target, "TARGET"
            else:
                if row['gold'] >= stop:
                    exit_price, reason = stop, "STOP"
                elif row['gold'] <= target:
                    exit_price, reason = target, "TARGET"

            if exit_price is not None:
                stop_dist = abs(entry_price - stop)
                if stop_dist > 0:
                    lots = (risk_pct * account) / (stop_dist * 100)
                    pnl = ((exit_price - entry_price) if direction == 1
                           else (entry_price - exit_price)) * lots * 100
                else:
                    pnl = 0

                trades.append({
                    'entry_time': entry_time, 'exit_time': row.name,
                    'direction': 'LONG' if direction == 1 else 'SHORT',
                    'entry': entry_price, 'exit': exit_price,
                    'stop': stop, 'target': target,
                    'pnl': pnl, 'reason': reason,
                    'bars_held': i - entry_idx,
                })
                in_trade = False
                account += pnl
        else:
            if pd.isna(row['atr']) or row['atr'] == 0:
                continue
            stop_dist = stop_mult * row['atr']

            if row['long_signal']:
                direction = 1
                entry_price = row['gold']
                stop = entry_price - stop_dist
                target = entry_price + stop_dist * rr
                entry_time, entry_idx, in_trade = row.name, i, True
            elif row['short_signal']:
                direction = -1
                entry_price = row['gold']
                stop = entry_price + stop_dist
                target = entry_price - stop_dist * rr
                entry_time, entry_idx, in_trade = row.name, i, True

    return pd.DataFrame(trades), account


def metrics(trades, final_acc, initial=ACCOUNT):
    if len(trades) == 0:
        return {"total_trades": 0}
    wins = trades[trades['pnl'] > 0]
    losses = trades[trades['pnl'] <= 0]

    gp = wins['pnl'].sum()
    gl = abs(losses['pnl'].sum())
    pf = gp / gl if gl > 0 else np.inf

    trades = trades.copy()
    trades['cum'] = trades['pnl'].cumsum()
    eq = initial + trades['cum']
    peak = eq.cummax()
    dd = ((eq - peak) / peak * 100).min()

    return {
        'total_trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate_%': round(len(wins) / len(trades) * 100, 2),
        'avg_win': round(wins['pnl'].mean(), 2) if len(wins) else 0,
        'avg_loss': round(losses['pnl'].mean(), 2) if len(losses) else 0,
        'expectancy': round(trades['pnl'].mean(), 2),
        'profit_factor': round(pf, 2),
        'total_pnl': round(trades['pnl'].sum(), 2),
        'total_return_%': round((final_acc - initial) / initial * 100, 2),
        'max_drawdown_%': round(dd, 2),
        'final_account': round(final_acc, 2),
    }


if __name__ == "__main__":
    print("=" * 60)
    print(" SMT + ORDER BLOCK BACKTEST v2")
    print("=" * 60)

    df = load_data()
    if df is None:
        exit(1)

    print("\nDetecting SMT...")
    df = find_smt(df)
    print(f"  Bullish SMT: {df['bullish_smt'].sum()}")
    print(f"  Bearish SMT: {df['bearish_smt'].sum()}")

    print("\nDetecting Order Blocks...")
    df = find_ob(df)
    print(f"  Bullish OB: {df['bullish_ob'].sum()}")
    print(f"  Bearish OB: {df['bearish_ob'].sum()}")

    print(f"\nCombining signals (SMT + OB within {SMT_TO_OB_WINDOW} candles)...")
    df = combine_signals(df)
    print(f"  Long signals: {df['long_signal'].sum()}")
    print(f"  Short signals: {df['short_signal'].sum()}")

    print("\nRunning backtest...")
    trades, final_acc = backtest(df)

    if len(trades) == 0:
        print("\nX Koi trade nahi bani. Window badhao ya conditions looser karo.")
        exit(0)

    m = metrics(trades, final_acc)

    print("\n" + "=" * 60)
    print(" RESULTS")
    print("=" * 60)
    for k, v in m.items():
        print(f"  {k:.<30} {v}")
    print("=" * 60)

    trades.to_csv(RESULTS_DIR / "trades.csv", index=False)
    with open(RESULTS_DIR / "summary.txt", "w") as f:
        f.write(f"Run: {datetime.now()}\n")
        f.write(f"Symbol: XAUUSD + XAGUSD ({TIMEFRAME})\n")
        f.write(f"Bars: {len(df)}\n\n")
        for k, v in m.items():
            f.write(f"{k}: {v}\n")

    print(f"\nTrades saved: {RESULTS_DIR / 'trades.csv'}")
    print(f"Summary saved: {RESULTS_DIR / 'summary.txt'}")
