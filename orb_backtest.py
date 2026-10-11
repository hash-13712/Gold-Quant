# ============================================================
# ORB (Opening Range Breakout) — BACKTEST
# NY Open 9:30-9:45 EST range, breakout entry
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_DIR = Path.home() / "gold_quant" / "data" / "candles"
RESULTS_DIR = Path.home() / "gold_quant" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TZ = 'US/Eastern'
TIMEFRAME = "5m"
SYMBOL = "XAUUSD"

OR_START_H = 9
OR_START_M = 30
OR_MINUTES = 15

TRADE_END_H = 15

RISK_PCT = 0.005
ACCOUNT = 10000
SPREAD = 0.30

MIN_RR = 0.8
TARGET_MULT = 1.5


def load_data():
    f = DATA_DIR / f"{SYMBOL}_{TIMEFRAME}.parquet"
    if not f.exists():
        print(f"X Nahi mili: {f}")
        return None
    df = pd.read_parquet(f)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert(TZ)
    df = df.copy()
    df['date'] = df.index.date
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['vol'] = df['ticks'] if 'ticks' in df.columns else 1
    df['cum_vol'] = df.groupby('date')['vol'].cumsum()
    df['cum_tpv'] = (df['tp'] * df['vol']).groupby(df['date']).cumsum()
    df['vwap'] = df['cum_tpv'] / df['cum_vol']
    df['avg_vol'] = df['vol'].rolling(20).mean()
    print(f"  Bars: {len(df):,}")
    print(f"  Range: {df.index.min()} to {df.index.max()}")
    return df


def get_opening_range(day_df):
    orb = day_df[
        (day_df.index.hour == OR_START_H) &
        (day_df.index.minute >= OR_START_M) &
        (day_df.index.minute < OR_START_M + OR_MINUTES)
    ]
    if len(orb) < 2:
        return None, None
    return orb['high'].max(), orb['low'].min()


def backtest_orb(df, use_vwap_filter=False, use_vol_filter=False,
                 target_mult=TARGET_MULT, min_rr=MIN_RR):
    trades = []
    dates = sorted(set(df.index.date))
    days_scanned = 0

    for date in dates:
        day = df[df.index.date == date]
        if len(day) < 20:
            continue

        or_high, or_low = get_opening_range(day)
        if or_high is None:
            continue

        or_range = or_high - or_low
        if or_range < 1.0:
            continue

        days_scanned += 1

        after_or = (day.index.hour > OR_START_H) | (
            (day.index.hour == OR_START_H) &
            (day.index.minute >= OR_START_M + OR_MINUTES)
        )
        trade_window = day[after_or]
        trade_window = trade_window[trade_window.index.hour <= TRADE_END_H]

        if len(trade_window) == 0:
            continue

        in_trade = False
        direction = 0
        entry = stop = target = 0
        entry_time = None

        for ts, c in trade_window.iterrows():
            if in_trade:
                if direction == 1:
                    if c['low'] <= stop:
                        pnl = (stop - entry) - SPREAD
                        trades.append({
                            'date': date, 'time': ts, 'direction': 'LONG',
                            'entry': entry, 'exit': stop, 'stop': stop,
                            'target': target, 'pnl': pnl, 'reason': 'STOP',
                        })
                        in_trade = False
                    elif c['high'] >= target:
                        pnl = (target - entry) - SPREAD
                        trades.append({
                            'date': date, 'time': ts, 'direction': 'LONG',
                            'entry': entry, 'exit': target, 'stop': stop,
                            'target': target, 'pnl': pnl, 'reason': 'TARGET',
                        })
                        in_trade = False
                else:
                    if c['high'] >= stop:
                        pnl = (entry - stop) - SPREAD
                        trades.append({
                            'date': date, 'time': ts, 'direction': 'SHORT',
                            'entry': entry, 'exit': stop, 'stop': stop,
                            'target': target, 'pnl': pnl, 'reason': 'STOP',
                        })
                        in_trade = False
                    elif c['low'] <= target:
                        pnl = (entry - target) - SPREAD
                        trades.append({
                            'date': date, 'time': ts, 'direction': 'SHORT',
                            'entry': entry, 'exit': target, 'stop': stop,
                            'target': target, 'pnl': pnl, 'reason': 'TARGET',
                        })
                        in_trade = False
                continue

            if c['close'] > or_high:
                if use_vwap_filter and c['close'] < c['vwap']:
                    continue
                if use_vol_filter and not pd.isna(c['avg_vol']):
                    if c['vol'] < c['avg_vol'] * 1.2:
                        continue

                direction = 1
                entry = c['close']
                entry_time = ts
                stop = or_low
                target = entry + or_range * target_mult

                risk = entry - stop
                reward = target - entry
                if risk <= 0:
                    continue
                rr_ratio = reward / risk
                if rr_ratio < min_rr:
                    continue

                in_trade = True

            elif c['close'] < or_low:
                if use_vwap_filter and c['close'] > c['vwap']:
                    continue
                if use_vol_filter and not pd.isna(c['avg_vol']):
                    if c['vol'] < c['avg_vol'] * 1.2:
                        continue

                direction = -1
                entry = c['close']
                entry_time = ts
                stop = or_high
                target = entry - or_range * target_mult

                risk = stop - entry
                reward = entry - target
                if risk <= 0:
                    continue
                rr_ratio = reward / risk
                if rr_ratio < min_rr:
                    continue

                in_trade = True

        if in_trade:
            last = trade_window.iloc[-1]
            pnl = ((last['close'] - entry) if direction == 1
                   else (entry - last['close'])) - SPREAD
            trades.append({
                'date': date, 'time': last.name,
                'direction': 'LONG' if direction == 1 else 'SHORT',
                'entry': entry, 'exit': last['close'], 'stop': stop,
                'target': target, 'pnl': pnl, 'reason': 'EOD',
            })

    print(f"  Days scanned: {days_scanned}")
    return pd.DataFrame(trades)


def calc_metrics(trades, initial=ACCOUNT):
    if len(trades) == 0:
        return None
    t = trades.copy()
    t['stop_dist'] = (t['entry'] - t['stop']).abs()
    t = t[t['stop_dist'] > 0].copy()
    t['lots'] = (RISK_PCT * initial) / (t['stop_dist'] * 100)
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
        'final': round(initial + t['pnl_usd'].sum(), 2),
    }, t


def show(label, trades):
    result = calc_metrics(trades)
    if result is None:
        print(f"  {label:<25} : no trades")
        return None
    m, t = result
    print(f"  {label:<25} T={m['trades']:>3}  WR={m['WR_%']:>5}%  "
          f"PF={m['PF']:>5}  Exp=${m['expectancy']:>7}  "
          f"DD={m['DD_%']:>6}%  Ret={m['return_%']:>6}%")
    return m, t


if __name__ == "__main__":
    print("=" * 80)
    print(" ORB BACKTEST — GOLD")
    print("=" * 80)
    df = load_data()
    if df is None:
        exit(1)

    print(f"\nOpening Range: {OR_START_H}:{OR_START_M:02d} to "
          f"{OR_START_H}:{OR_START_M + OR_MINUTES:02d} EST")
    print(f"Target: {TARGET_MULT}x OR range, Min RR: {MIN_RR}")

    print("\n" + "=" * 80)
    print(" CONFIG COMPARISON")
    print("=" * 80)

    results = {}

    print("\n[1] Basic ORB (no filters)")
    r = show("Basic", backtest_orb(df))
    if r:
        results['Basic'] = r

    print("\n[2] ORB + VWAP filter")
    r = show("VWAP filter", backtest_orb(df, use_vwap_filter=True))
    if r:
        results['VWAP'] = r

    print("\n[3] ORB + Volume filter")
    r = show("Volume filter", backtest_orb(df, use_vol_filter=True))
    if r:
        results['Volume'] = r

    print("\n[4] ORB + VWAP + Volume")
    r = show("VWAP + Volume",
             backtest_orb(df, use_vwap_filter=True, use_vol_filter=True))
    if r:
        results['Both'] = r

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
        t.to_csv(RESULTS_DIR / "orb_trades.csv", index=False)
        print(f"Trades saved: {RESULTS_DIR / 'orb_trades.csv'}")

        print("\n--- Best config details ---")
        for k, v in m.items():
            print(f"  {k:.<20} {v}")

        print("\n--- Direction breakdown ---")
        for d in t['direction'].unique():
            sub = t[t['direction'] == d]
            w = (sub['pnl_usd'] > 0).sum()
            print(f"  {d:.<10} trades={len(sub):>3}  WR={w/len(sub)*100:.1f}%")

        print("\n--- Exit reason breakdown ---")
        for r_ in t['reason'].unique():
            sub = t[t['reason'] == r_]
            w = (sub['pnl_usd'] > 0).sum()
            print(f"  {r_:.<10} trades={len(sub):>3}  WR={w/len(sub)*100:.1f}%")
