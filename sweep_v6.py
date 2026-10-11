# ============================================================
# LIQUIDITY SWEEP v6 — Breakeven + Trailing + Bug Fixed
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_DIR = Path.home() / "gold_quant" / "data"
RESULTS_DIR = Path.home() / "gold_quant" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TIMEFRAME = "5m"
SL_DOLLARS = 4.0
TP_DOLLARS = 8.0
RISK_PCT = 0.005
ACCOUNT = 10000
SPREAD = 0.30

NY_START, NY_END = 7, 15
SWEEP_START, SWEEP_END = 1, 4
MAX_TRADES_DAY = 1

# Breakeven settings
BE_TRIGGER_PIPS = 40      # 40 pips = $4 ke baad SL ko BE pe le jao
BE_OFFSET_PIPS = 5        # BE pe 5 pips profit lock (spread cover)

# Trailing settings
TRAIL_AFTER_PIPS = 60     # 60 pips ke baad trail start
TRAIL_DISTANCE_PIPS = 30  # 30 pips peeche trail


def load_gold():
    f = DATA_DIR / f"XAUUSD_{TIMEFRAME}.parquet"
    if not f.exists():
        print(f"X Nahi mili: {f}")
        return None
    df = pd.read_parquet(f)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert('US/Eastern')
    print(f"  Bars: {len(df):,}")
    print(f"  Range: {df.index.min()} to {df.index.max()}")
    return df


def get_daily_levels(df):
    levels = {}
    for date, group in df.groupby(df.index.date):
        ny = group[(group.index.hour >= NY_START) & (group.index.hour < NY_END)]
        if len(ny) > 50:
            levels[date] = {
                'NY_high': ny['high'].max(),
                'NY_low': ny['low'].min(),
            }
    return levels


def backtest(df, sl=SL_DOLLARS, tp=TP_DOLLARS, spread=SPREAD,
             long_only=True, short_only=False,
             use_breakeven=True, use_trailing=True,
             max_per_day=MAX_TRADES_DAY):
    levels = get_daily_levels(df)
    dates = sorted(levels.keys())

    trades = []
    days_scanned = 0
    days_with_trades = 0

    for i in range(1, len(dates)):
        prev_date = dates[i-1]
        curr_date = dates[i]
        prev_lvl = levels[prev_date]

        curr_day = df[df.index.date == curr_date].copy()
        scan_window = curr_day[(curr_day.index.hour >= SWEEP_START) &
                               (curr_day.index.hour < SWEEP_END)]
        if len(scan_window) == 0:
            continue

        days_scanned += 1
        in_trade = False
        trades_today = 0
        swept = set()
        entry_price = stop = target = direction = 0
        entry_time = None
        level_hit = None
        entry_pos = None
        be_triggered = False  # Breakeven flag

        # Phase 1: Scan for entry
        for pos, (ts, candle) in enumerate(scan_window.iterrows()):
            if trades_today >= max_per_day:
                break
            for lvl_name, lvl_price in prev_lvl.items():
                if lvl_name in swept:
                    continue

                # HIGH sweep → SHORT
                if lvl_name.endswith('high') and not long_only:
                    if candle['high'] > lvl_price and candle['close'] < lvl_price:
                        direction = -1
                        entry_price = candle['close']
                        stop = entry_price + sl
                        target = entry_price - tp
                        entry_time = ts
                        level_hit = lvl_name
                        swept.add(lvl_name)
                        in_trade = True
                        trades_today += 1
                        entry_pos = curr_day.index.get_loc(ts)
                        break

                # LOW sweep → LONG
                if lvl_name.endswith('low') and not short_only:
                    if candle['low'] < lvl_price and candle['close'] > lvl_price:
                        direction = 1
                        entry_price = candle['close']
                        stop = entry_price - sl
                        target = entry_price + tp
                        entry_time = ts
                        level_hit = lvl_name
                        swept.add(lvl_name)
                        in_trade = True
                        trades_today += 1
                        entry_pos = curr_day.index.get_loc(ts)
                        break
            if in_trade:
                break

        if not in_trade:
            continue

        # Phase 2: Trade management with BE + Trailing
        remaining = curr_day.iloc[entry_pos + 1:]
        exit_price = None
        exit_reason = None
        exit_time = None

        for ts, candle in remaining.iterrows():
            # Current price in pips
            if direction == 1:
                profit_pips = (candle['high'] - entry_price) / 0.10
                loss_check = candle['low']

                # Breakeven check
                if use_breakeven and not be_triggered:
                    if profit_pips >= BE_TRIGGER_PIPS:
                        be_price = entry_price + BE_OFFSET_PIPS * 0.10
                        if be_price > stop:
                            stop = be_price
                            be_triggered = True

                # Trailing check
                if use_trailing and be_triggered:
                    if profit_pips >= TRAIL_AFTER_PIPS:
                        new_stop = candle['high'] - TRAIL_DISTANCE_PIPS * 0.10
                        if new_stop > stop:
                            stop = new_stop

                # Stop hit
                if loss_check <= stop:
                    if be_triggered and stop >= entry_price:
                        exit_reason = 'BE'
                    else:
                        exit_reason = 'STOP'
                    exit_price = stop
                    exit_time = ts
                    break

                # Target hit
                if candle['high'] >= target:
                    exit_price = target
                    exit_reason = 'TARGET'
                    exit_time = ts
                    break

            else:  # SHORT
                profit_pips = (entry_price - candle['low']) / 0.10
                loss_check = candle['high']

                # Breakeven
                if use_breakeven and not be_triggered:
                    if profit_pips >= BE_TRIGGER_PIPS:
                        be_price = entry_price - BE_OFFSET_PIPS * 0.10
                        if be_price < stop:
                            stop = be_price
                            be_triggered = True

                # Trailing
                if use_trailing and be_triggered:
                    if profit_pips >= TRAIL_AFTER_PIPS:
                        new_stop = candle['low'] + TRAIL_DISTANCE_PIPS * 0.10
                        if new_stop < stop:
                            stop = new_stop

                # Stop hit
                if loss_check >= stop:
                    if be_triggered and stop <= entry_price:
                        exit_reason = 'BE'
                    else:
                        exit_reason = 'STOP'
                    exit_price = stop
                    exit_time = ts
                    break

                # Target hit
                if candle['low'] <= target:
                    exit_price = target
                    exit_reason = 'TARGET'
                    exit_time = ts
                    break

        if exit_price is None:
            last = remaining.iloc[-1] if len(remaining) > 0 else curr_day.iloc[-1]
            exit_price = last['close']
            exit_reason = 'EOD'
            exit_time = last.name

        raw_pnl = ((exit_price - entry_price) if direction == 1
                   else (entry_price - exit_price)) - spread

        trades.append({
            'date': curr_date, 'direction': 'LONG' if direction == 1 else 'SHORT',
            'level': level_hit, 'entry_time': entry_time, 'exit_time': exit_time,
            'entry': entry_price, 'exit': exit_price,
            'pnl': raw_pnl, 'reason': exit_reason,
            'be_triggered': be_triggered,
        })
        days_with_trades += 1

    print(f"\n  Days scanned: {days_scanned}")
    print(f"  Days with trades: {days_with_trades}")
    return pd.DataFrame(trades)


def metrics(trades, initial=ACCOUNT):
    if len(trades) == 0:
        return {"total_trades": 0}
    risk_per_trade = RISK_PCT * initial
    lots = risk_per_trade / (SL_DOLLARS * 100)
    trades = trades.copy()
    trades['pnl_usd'] = trades['pnl'] * lots * 100

    wins = trades[trades['pnl_usd'] > 0]
    losses = trades[trades['pnl_usd'] <= 0]
    gp = wins['pnl_usd'].sum()
    gl = abs(losses['pnl_usd'].sum())
    pf = gp / gl if gl > 0 else np.inf
    trades['cum'] = trades['pnl_usd'].cumsum()
    eq = initial + trades['cum']
    dd = ((eq - eq.cummax()) / eq.cummax() * 100).min()

    return {
        'total_trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate_%': round(len(wins) / len(trades) * 100, 2),
        'avg_win_$': round(wins['pnl_usd'].mean(), 2) if len(wins) else 0,
        'avg_loss_$': round(losses['pnl_usd'].mean(), 2) if len(losses) else 0,
        'expectancy_$': round(trades['pnl_usd'].mean(), 2),
        'profit_factor': round(pf, 2),
        'total_pnl_$': round(trades['pnl_usd'].sum(), 2),
        'total_return_%': round(trades['pnl_usd'].sum() / initial * 100, 2),
        'max_drawdown_%': round(dd, 2),
        'final_account_$': round(initial + trades['pnl_usd'].sum(), 2),
    }


def show_results(label, trades):
    m = metrics(trades)
    print(f"\n--- {label} ---")
    print(f"  Trades: {m.get('total_trades',0)}  WR: {m.get('win_rate_%',0)}%  "
          f"PF: {m.get('profit_factor',0)}  Exp: ${m.get('expectancy_$',0)}  "
          f"DD: {m.get('max_drawdown_%',0)}%")
    return m


if __name__ == "__main__":
    print("=" * 60)
    print(" LIQUIDITY SWEEP v6 — BE + Trailing")
    print("=" * 60)
    df = load_gold()
    if df is None:
        exit(1)

    print("\nTesting 6 configurations...")
    results = {}

    # 1. Baseline — no BE, no trail
    print("\n[1] Baseline (no BE, no trail, LONG only)")
    t1 = backtest(df, use_breakeven=False, use_trailing=False, long_only=True)
    results['Baseline'] = show_results("Baseline", t1)

    # 2. BE only
    print("\n[2] BE after 40 pips")
    t2 = backtest(df, use_breakeven=True, use_trailing=False, long_only=True)
    results['BE only'] = show_results("BE only", t2)

    # 3. BE + Trailing
    print("\n[3] BE + Trailing")
    t3 = backtest(df, use_breakeven=True, use_trailing=True, long_only=True)
    results['BE + Trail'] = show_results("BE + Trail", t3)

    # 4. Both directions, BE + trail
    print("\n[4] BE + Trail, both directions")
    t4 = backtest(df, use_breakeven=True, use_trailing=True,
                  long_only=False, short_only=False)
    results['Both + BE+Trail'] = show_results("Both + BE+Trail", t4)

    # 5. SHORT only
    print("\n[5] SHORT only, BE + trail")
    t5 = backtest(df, use_breakeven=True, use_trailing=True,
                  long_only=False, short_only=True)
    results['SHORT only'] = show_results("SHORT only", t5)

    # Save best trades
    t3.to_csv(RESULTS_DIR / "sweep_v6_trades.csv", index=False)
    print(f"\nTrades saved: {RESULTS_DIR / 'sweep_v6_trades.csv'}")

    print("\n" + "=" * 60)
    print(" SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Config':<20} {'Trades':>7} {'WR%':>7} {'PF':>7} {'Exp$':>8} {'DD%':>7}")
    print("-" * 60)
    for name, m in results.items():
        print(f"{name:<20} {m.get('total_trades',0):>7} "
              f"{m.get('win_rate_%',0):>7} {m.get('profit_factor',0):>7} "
              f"{m.get('expectancy_$',0):>8} {m.get('max_drawdown_%',0):>7}")
