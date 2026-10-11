# ============================================================
# MULTI-TIMEFRAME: 15m Bias + 5m Sweep Entry
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path.home() / "gold_quant" / "data" / "candles"
RESULTS_DIR = Path.home() / "gold_quant" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TZ = 'US/Eastern'
ACCOUNT = 10000
RISK_PCT = 0.005
SPREAD = 0.30

# Config
HTF_LOOKBACK = 20       # 15m candles for swing structure
LTF_SWEEP_LOOKBACK = 10 # 5m candles for sweep
SL_DOLLARS = 4.0
RR = 2.0
SESSION_START = 2       # 2:00 EST (London)
SESSION_END = 12        # 12:00 EST (end of NY AM)


def load_pair():
    f5 = DATA_DIR / "XAUUSD_5m.parquet"
    f15 = DATA_DIR / "XAUUSD_15m.parquet"
    if not f5.exists() or not f15.exists():
        print("X Candle files nahi mili")
        return None, None

    df5 = pd.read_parquet(f5)
    df15 = pd.read_parquet(f15)
    for df in (df5, df15):
        df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert(TZ)
        df['date'] = df.index.date

    # VWAP on 5m
    df5['tp'] = (df5['high'] + df5['low'] + df5['close']) / 3
    df5['vol'] = df5['ticks'] if 'ticks' in df5.columns else 1
    df5['cum_vol'] = df5.groupby('date')['vol'].cumsum()
    df5['cum_tpv'] = (df5['tp'] * df5['vol']).groupby(df5['date']).cumsum()
    df5['vwap'] = df5['cum_tpv'] / df5['cum_vol']

    print(f"  5m bars: {len(df5):,}")
    print(f"  15m bars: {len(df15):,}")
    print(f"  Range: {df5.index.min()} to {df5.index.max()}")
    return df5, df15


def compute_htf_bias(df15):
    """15m trend via recent high/low BOS."""
    df = df15.copy()
    df['recent_high'] = df['high'].rolling(HTF_LOOKBACK).max()
    df['recent_low'] = df['low'].rolling(HTF_LOOKBACK).min()
    df['bos_up'] = df['close'] > df['recent_high'].shift(1)
    df['bos_down'] = df['close'] < df['recent_low'].shift(1)

    # Carry trend forward
    trend = 0
    trends = []
    for i in range(len(df)):
        if df['bos_up'].iloc[i]:
            trend = 1
        elif df['bos_down'].iloc[i]:
            trend = -1
        trends.append(trend)
    df['htf_trend'] = trends

    # Latest HTF candle trend (forward fill to 5m later)
    return df[['htf_trend']]


def backtest(df5, df15, sl=SL_DOLLARS, rr=RR,
             use_vwap_filter=True, use_session_filter=True):
    htf = compute_htf_bias(df15)

    trades = []
    in_trade = False
    direction = 0
    entry = stop = target = 0
    entry_time = None
    entry_idx = 0

    for i in range(LTF_SWEEP_LOOKBACK + 5, len(df5)):
        row = df5.iloc[i]
        ts = row.name

        # Session filter
        if use_session_filter:
            if not (SESSION_START <= ts.hour < SESSION_END):
                if in_trade:
                    # Force close at session end
                    pass
                continue

        # ---- Get latest 15m bias ----
        prev_15 = htf[htf.index <= ts]
        if len(prev_15) == 0:
            continue
        htf_trend = prev_15['htf_trend'].iloc[-1]

        # ---- EXIT ----
        if in_trade:
            if direction == 1:
                if row['low'] <= stop:
                    pnl = (stop - entry) - SPREAD
                    trades.append({
                        'entry_time': entry_time, 'exit_time': ts,
                        'direction': 'LONG', 'entry': entry, 'exit': stop,
                        'pnl': pnl, 'reason': 'STOP', 'bars': i - entry_idx,
                    })
                    in_trade = False
                elif row['high'] >= target:
                    pnl = (target - entry) - SPREAD
                    trades.append({
                        'entry_time': entry_time, 'exit_time': ts,
                        'direction': 'LONG', 'entry': entry, 'exit': target,
                        'pnl': pnl, 'reason': 'TARGET', 'bars': i - entry_idx,
                    })
                    in_trade = False
            else:
                if row['high'] >= stop:
                    pnl = (entry - stop) - SPREAD
                    trades.append({
                        'entry_time': entry_time, 'exit_time': ts,
                        'direction': 'SHORT', 'entry': entry, 'exit': stop,
                        'pnl': pnl, 'reason': 'STOP', 'bars': i - entry_idx,
                    })
                    in_trade = False
                elif row['low'] <= target:
                    pnl = (entry - target) - SPREAD
                    trades.append({
                        'entry_time': entry_time, 'exit_time': ts,
                        'direction': 'SHORT', 'entry': entry, 'exit': target,
                        'pnl': pnl, 'reason': 'TARGET', 'bars': i - entry_idx,
                    })
                    in_trade = False
            continue

        # ---- ENTRY ----
        # Need htf bias
        if htf_trend == 0:
            continue

        # 5m sweep levels
        lookback = df5.iloc[i - LTF_SWEEP_LOOKBACK:i]
        recent_5m_low = lookback['low'].min()
        recent_5m_high = lookback['high'].max()

        # Bullish: 15m uptrend + 5m sweep of recent low
        if htf_trend == 1:
            # Sweep happened: wick below recent low, close above
            if row['low'] < recent_5m_low and row['close'] > recent_5m_low:
                if use_vwap_filter and row['close'] < row['vwap']:
                    continue
                direction = 1
                entry = row['close']
                stop = row['low'] - 0.5  # 0.5 below sweep wick
                target = entry + sl * rr
                entry_time = ts
                entry_idx = i
                in_trade = True

        # Bearish: 15m downtrend + 5m sweep of recent high
        elif htf_trend == -1:
            if row['high'] > recent_5m_high and row['close'] < recent_5m_high:
                if use_vwap_filter and row['close'] > row['vwap']:
                    continue
                direction = -1
                entry = row['close']
                stop = row['high'] + 0.5
                target = entry - sl * rr
                entry_time = ts
                entry_idx = i
                in_trade = True

        # Skip if risk too big
        if in_trade:
            risk = abs(entry - stop)
            if risk > sl * 2:  # stop more than 2x intended
                in_trade = False

    return pd.DataFrame(trades)


def calc_metrics(trades, initial=ACCOUNT):
    if len(trades) == 0:
        return None
    t = trades.copy()
    t['stop_dist'] = (t['entry'] - t['exit']).abs()
    # Risk-based sizing
    t['stop_dist'] = (t['entry'] - t['entry']).abs() + 1  # placeholder
    # Simple: fixed risk per trade
    t['pnl_usd'] = t['pnl'] * (RISK_PCT * initial) / SL_DOLLARS

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
        'return_%': round(t['pnl_usd'].sum() / initial * 100, 2),
        'DD_%': round(dd, 2),
    }, t


def show(label, trades):
    r = calc_metrics(trades)
    if r is None:
        print(f"  {label:<30} : no trades")
        return None
    m, t = r
    print(f"  {label:<30} T={m['trades']:>3}  WR={m['WR_%']:>5}%  "
          f"PF={m['PF']:>5}  Exp=${m['expectancy']:>7}  "
          f"DD={m['DD_%']:>6}%  Ret={m['return_%']:>6}%")
    return m, t


if __name__ == "__main__":
    print("=" * 80)
    print(" MULTI-TIMEFRAME: 15m Bias + 5m Sweep")
    print("=" * 80)

    df5, df15 = load_pair()
    if df5 is None:
        exit(1)

    print("\n" + "=" * 80)
    print(" CONFIG COMPARISON")
    print("=" * 80)

    results = {}

    print("\n[1] Basic (no filters)")
    r = show("Basic",
             backtest(df5, df15, use_vwap_filter=False, use_session_filter=False))
    if r:
        results['Basic'] = r

    print("\n[2] + VWAP filter")
    r = show("+ VWAP",
             backtest(df5, df15, use_vwap_filter=True, use_session_filter=False))
    if r:
        results['VWAP'] = r

    print("\n[3] + Session filter")
    r = show("+ Session",
             backtest(df5, df15, use_vwap_filter=False, use_session_filter=True))
    if r:
        results['Session'] = r

    print("\n[4] + VWAP + Session")
    r = show("+ VWAP + Session",
             backtest(df5, df15, use_vwap_filter=True, use_session_filter=True))
    if r:
        results['Both'] = r

    if results:
        print("\n" + "=" * 80)
        print(" SUMMARY")
        print("=" * 80)
        print(f"{'Config':<20} {'Trades':>7} {'WR%':>7} {'PF':>7} "
              f"{'Exp$':>8} {'DD%':>7} {'Ret%':>7}")
        print("-" * 80)
        best = None
        best_pf = -1
        for name, (m, t) in results.items():
            print(f"{name:<20} {m['trades']:>7} {m['WR_%']:>7} "
                  f"{m['PF']:>7} {m['expectancy']:>8} "
                  f"{m['DD_%']:>7} {m['return_%']:>7}")
            if m['PF'] > best_pf:
                best_pf = m['PF']
                best = (name, m, t)

        if best:
            name, m, t = best
            print(f"\nBEST: {name}  (PF={m['PF']}, WR={m['WR_%']}%)")
            t.to_csv(RESULTS_DIR / "mtf_trades.csv", index=False)
            print(f"Saved: {RESULTS_DIR / 'mtf_trades.csv'}")

            print("\n--- Direction breakdown ---")
            for d in t['direction'].unique():
                sub = t[t['direction'] == d]
                w = (sub['pnl_usd'] > 0).sum()
                print(f"  {d:<10} trades={len(sub):>3}  WR={w/len(sub)*100:.1f}%")
