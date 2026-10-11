# ============================================================
# DELTA DIVERGENCE BACKTEST (Tick Count Proxy)
# Up-ticks vs Down-ticks per candle
# Divergence: price new high/low, delta confirm nahi
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path.home() / "gold_quant" / "data"
RESULTS_DIR = Path.home() / "gold_quant" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TZ = 'US/Eastern'
ACCOUNT = 10000
RISK_PCT = 0.005
SPREAD = 0.30


def build_delta_candles(tf='5min'):
    """Tick CSV se candles + tick-count delta banao."""
    files = sorted(DATA_DIR.glob("xauusd*.csv"))
    if not files:
        print(f"X Koi CSV nahi mili")
        return None

    print(f"  Loading {len(files)} CSV file(s)...")
    chunks = []
    for f in files:
        print(f"    {f.name}")
        for chunk in pd.read_csv(f, chunksize=2_000_000):
            chunk.columns = [c.lower().strip() for c in chunk.columns]
            chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], unit='ms', utc=True)
            chunk['mid'] = (chunk['bidprice'] + chunk['askprice']) / 2
            chunk['up'] = (chunk['mid'].diff() > 0).astype(int)
            chunk['down'] = (chunk['mid'].diff() < 0).astype(int)
            chunks.append(chunk[['timestamp', 'mid', 'up', 'down']])

    df = pd.concat(chunks, ignore_index=True)
    df = df.sort_values('timestamp').set_index('timestamp')

    # Resample to candles
    ohlc = df['mid'].resample(tf).agg(
        open='first', high='max', low='min', close='last'
    )
    ohlc['up_ticks'] = df['up'].resample(tf).sum()
    ohlc['down_ticks'] = df['down'].resample(tf).sum()
    ohlc = ohlc.dropna()

    ohlc['delta'] = ohlc['up_ticks'] - ohlc['down_ticks']
    ohlc['cum_delta'] = ohlc['delta'].cumsum()

    # EST timezone
    if ohlc.index.tz is None:
        ohlc.index = ohlc.index.tz_localize('UTC')
    ohlc.index = ohlc.index.tz_convert(TZ)
    ohlc['date'] = ohlc.index.date

    # Per-session cumulative delta
    ohlc['session_delta'] = ohlc.groupby('date')['delta'].cumsum()

    print(f"  Total candles: {len(ohlc):,}")
    print(f"  Range: {ohlc.index.min()} to {ohlc.index.max()}")

    out = DATA_DIR / "candles" / f"XAUUSD_{tf.replace('min','m')}_delta.parquet"
    ohlc.to_parquet(out)
    print(f"  Saved: {out}")
    return ohlc


def detect_divergence(df, lookback=10):
    """
    Bullish: price lower low, delta higher low
    Bearish: price higher high, delta lower high
    """
    df = df.copy()
    df['price_low'] = df['low'].rolling(lookback).min()
    df['price_high'] = df['high'].rolling(lookback).max()
    df['delta_low'] = df['session_delta'].rolling(lookback).min()
    df['delta_high'] = df['session_delta'].rolling(lookback).max()

    # Price naya low, delta naya low NAHI (delta higher low)
    df['bull_div'] = (
        (df['low'] <= df['price_low']) &
        (df['session_delta'] > df['delta_low'])
    )

    # Price naya high, delta naya high NAHI
    df['bear_div'] = (
        (df['high'] >= df['price_high']) &
        (df['session_delta'] < df['delta_high'])
    )

    return df


def backtest_delta(df, lookback=10, sl_dollars=5.0, rr=2.0):
    trades = []
    in_trade = False
    direction = 0
    entry = stop = target = 0
    entry_time = None
    entry_idx = 0
    account = ACCOUNT

    for i in range(lookback + 5, len(df)):
        row = df.iloc[i]

        if in_trade:
            # EXIT
            if direction == 1:
                if row['low'] <= stop:
                    pnl = (stop - entry) - SPREAD
                    trades.append({'entry_time': entry_time, 'exit_time': row.name,
                                   'direction': 'LONG', 'entry': entry, 'exit': stop,
                                   'pnl': pnl, 'reason': 'STOP', 'bars': i - entry_idx})
                    in_trade = False
                elif row['high'] >= target:
                    pnl = (target - entry) - SPREAD
                    trades.append({'entry_time': entry_time, 'exit_time': row.name,
                                   'direction': 'LONG', 'entry': entry, 'exit': target,
                                   'pnl': pnl, 'reason': 'TARGET', 'bars': i - entry_idx})
                    in_trade = False
            else:
                if row['high'] >= stop:
                    pnl = (entry - stop) - SPREAD
                    trades.append({'entry_time': entry_time, 'exit_time': row.name,
                                   'direction': 'SHORT', 'entry': entry, 'exit': stop,
                                   'pnl': pnl, 'reason': 'STOP', 'bars': i - entry_idx})
                    in_trade = False
                elif row['low'] <= target:
                    pnl = (entry - target) - SPREAD
                    trades.append({'entry_time': entry_time, 'exit_time': row.name,
                                   'direction': 'SHORT', 'entry': entry, 'exit': target,
                                   'pnl': pnl, 'reason': 'TARGET', 'bars': i - entry_idx})
                    in_trade = False
        else:
            # ENTRY
            if row['bull_div']:
                direction = 1
                entry = row['close']
                stop = entry - sl_dollars
                target = entry + sl_dollars * rr
                entry_time = row.name
                entry_idx = i
                in_trade = True
            elif row['bear_div']:
                direction = -1
                entry = row['close']
                stop = entry + sl_dollars
                target = entry - sl_dollars * rr
                entry_time = row.name
                entry_idx = i
                in_trade = True

    return pd.DataFrame(trades)


def calc_metrics(trades, initial=ACCOUNT, sl_dollars=5.0):
    if len(trades) == 0:
        return None
    t = trades.copy()
    t['lots'] = (RISK_PCT * initial) / (sl_dollars * 100)
    t['pnl_usd'] = t['pnl'] * t['lots'] * 100

    wins = t[t['pnl_usd'] > 0]
    losses = t[t['pnl_usd'] <= 0]
    gp = wins['pnl_usd'].sum()
    gl = abs(losses['pnl_usd'].sum())
    pf = gp / gl if gl > 0 else float('inf')

    t['cum'] = t['pnl_usd'].cumsum()
    eq = initial + t['cum']
    dd = ((eq - eq.cummax()) / eq.cummax() * 100).min()

    return {
        'trades': len(t),
        'wins': len(wins),
        'losses': len(losses),
        'WR_%': round(len(wins) / len(t) * 100, 1) if len(t) else 0,
        'avg_win': round(wins['pnl_usd'].mean(), 2) if len(wins) else 0,
        'avg_loss': round(losses['pnl_usd'].mean(), 2) if len(losses) else 0,
        'expectancy': round(t['pnl_usd'].mean(), 2),
        'PF': round(pf, 2),
        'total_pnl': round(t['pnl_usd'].sum(), 2),
        'return_%': round(t['pnl_usd'].sum() / initial * 100, 2),
        'DD_%': round(dd, 2),
    }, t


def show(label, trades, sl_dollars=5.0):
    result = calc_metrics(trades, sl_dollars=sl_dollars)
    if result is None:
        print(f"  {label:<30} : no trades")
        return None
    m, t = result
    print(f"  {label:<30} T={m['trades']:>3}  WR={m['WR_%']:>5}%  "
          f"PF={m['PF']:>5}  Exp=${m['expectancy']:>7}  "
          f"DD={m['DD_%']:>6}%  Ret={m['return_%']:>6}%")
    return m, t


if __name__ == "__main__":
    print("=" * 80)
    print(" DELTA DIVERGENCE BACKTEST (Tick Count Proxy)")
    print("=" * 80)

    cache = DATA_DIR / "candles" / "XAUUSD_5m_delta.parquet"
    if cache.exists():
        print(f"\nLoading cached delta candles...")
        df = pd.read_parquet(cache)
        df.index = pd.to_datetime(df.index)
        print(f"  Candles: {len(df):,}")
    else:
        print("\nBuilding delta candles from tick CSVs...")
        df = build_delta_candles('5min')
        if df is None:
            exit(1)

    print(f"\n  Date range: {df.index.min()} to {df.index.max()}")

    print("\n" + "=" * 80)
    print(" PARAMETER SWEEP")
    print("=" * 80)

    results = {}
    best_pf = -1
    best_trades = None
    best_cfg = None

    for lookback in [5, 10, 20]:
        for sl in [4.0, 5.0, 6.0]:
            for rr in [1.5, 2.0, 3.0]:
                d = detect_divergence(df, lookback=lookback)
                t = backtest_delta(d, lookback=lookback, sl_dollars=sl, rr=rr)
                m = calc_metrics(t, sl_dollars=sl)
                if m is None:
                    continue
                m_dict, m_t = m
                label = f"LB={lookback} SL=${sl} RR={rr}"
                print(f"  {label:<30} T={m_dict['trades']:>3}  "
                      f"WR={m_dict['WR_%']:>5}%  PF={m_dict['PF']:>5}  "
                      f"Exp=${m_dict['expectancy']:>7}  DD={m_dict['DD_%']:>6}%")
                if m_dict['PF'] > best_pf:
                    best_pf = m_dict['PF']
                    best_trades = m_t
                    best_cfg = (lookback, sl, rr)

    if best_trades is not None and len(best_trades) > 0:
        print("\n" + "=" * 80)
        print(f" BEST: LB={best_cfg[0]} SL=${best_cfg[1]} RR={best_cfg[2]}  "
              f"(PF={best_pf:.2f})")
        print("=" * 80)

        m, t = calc_metrics(best_trades, sl_dollars=best_cfg[1])
        for k, v in m.items():
            print(f"  {k:.<20} {v}")

        print("\n--- Direction breakdown ---")
        for d in t['direction'].unique():
            sub = t[t['direction'] == d]
            w = (sub['pnl_usd'] > 0).sum()
            print(f"  {d:.<10} trades={len(sub):>3}  WR={w/len(sub)*100:.1f}%")

        print("\n--- Exit reason breakdown ---")
        for r in t['reason'].unique():
            sub = t[t['reason'] == r]
            w = (sub['pnl_usd'] > 0).sum()
            print(f"  {r:.<10} trades={len(sub):>3}  WR={w/len(sub)*100:.1f}%")

        t.to_csv(RESULTS_DIR / "delta_trades.csv", index=False)
        print(f"\nTrades saved: {RESULTS_DIR / 'delta_trades.csv'}")
