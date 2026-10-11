# ============================================================
# LIQUIDITY SWEEP v3 — GOLD
# Levels: 07:00-15:00 NY range + 21:00-22:00 range
# Sweep window: 01:00-02:00 EST next day
# SL = 40 pips ($4), TP = 70-100 pips ($7-10)
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_DIR = Path.home() / "gold_quant" / "data"
RESULTS_DIR = Path.home() / "gold_quant" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TIMEFRAME = "5m"
SL_DOLLARS = 4.0          # 40 pips
TP_DOLLARS = 8.0          # 80 pips (middle of 70-100)
RISK_PCT = 0.005
ACCOUNT = 10000
SPREAD = 0.30

# Levels definition (EST hours)
NY_START = 7
NY_END = 15
LATE_START = 21
LATE_END = 22

# Sweep watch window (EST)
SWEEP_START = 1
SWEEP_END = 2

MAX_TRADES_DAY = 1


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
    """Har din ke NY range aur late evening range ka High/Low."""
    levels = {}
    for date, group in df.groupby(df.index.date):
        day_l = {}
        # NY range 07:00-15:00
        ny = group[(group.index.hour >= NY_START) & (group.index.hour < NY_END)]
        if len(ny) > 50:
            day_l['NY_high'] = ny['high'].max()
            day_l['NY_low'] = ny['low'].min()
        # Late 21:00-22:00
        late = group[(group.index.hour >= LATE_START) & (group.index.hour < LATE_END)]
        if len(late) > 6:
            day_l['Late_high'] = late['high'].max()
            day_l['Late_low'] = late['low'].min()
        # Dono ranges chahiye
        if len(day_l) == 4:
            levels[date] = day_l
    return levels


def backtest(df, sl=SL_DOLLARS, tp=TP_DOLLARS, spread=SPREAD,
             max_per_day=MAX_TRADES_DAY):
    levels = get_daily_levels(df)
    dates = sorted(levels.keys())

    trades = []
    days_scanned = 0
    days_with_data = 0

    for i in range(1, len(dates)):
        prev_date = dates[i - 1]
        curr_date = dates[i]
        prev_lvl = levels[prev_date]

        # Curr day 01:00-02:00 window
        curr_day = df[df.index.date == curr_date]
        scan = curr_day[(curr_day.index.hour >= SWEEP_START) &
                        (curr_day.index.hour < SWEEP_END)]
        if len(scan) == 0:
            continue

        days_scanned += 1
        in_trade = False
        trades_today = 0
        swept = set()
        entry_price = stop = target = direction = 0
        entry_time = None
        level_hit = None

        for ts, candle in scan.iterrows():
            if in_trade:
                # EXIT
                if direction == 1:  # LONG
                    if candle['low'] <= stop:
                        trades.append({
                            'date': curr_date, 'direction': 'LONG',
                            'level': level_hit, 'entry_time': entry_time,
                            'exit_time': ts, 'entry': entry_price,
                            'exit': stop, 'pnl': -(sl + spread),
                            'reason': 'STOP'})
                        in_trade = False
                    elif candle['high'] >= target:
                        trades.append({
                            'date': curr_date, 'direction': 'LONG',
                            'level': level_hit, 'entry_time': entry_time,
                            'exit_time': ts, 'entry': entry_price,
                            'exit': target, 'pnl': tp - spread,
                            'reason': 'TARGET'})
                        in_trade = False
                else:  # SHORT
                    if candle['high'] >= stop:
                        trades.append({
                            'date': curr_date, 'direction': 'SHORT',
                            'level': level_hit, 'entry_time': entry_time,
                            'exit_time': ts, 'entry': entry_price,
                            'exit': stop, 'pnl': -(sl + spread),
                            'reason': 'STOP'})
                        in_trade = False
                    elif candle['low'] <= target:
                        trades.append({
                            'date': curr_date, 'direction': 'SHORT',
                            'level': level_hit, 'entry_time': entry_time,
                            'exit_time': ts, 'entry': entry_price,
                            'exit': target, 'pnl': tp - spread,
                            'reason': 'TARGET'})
                        in_trade = False
                continue

            if trades_today >= max_per_day:
                continue

            # SWEEP DETECTION — kisi bhi level ka
            for lvl_name, lvl_price in prev_lvl.items():
                if lvl_name in swept:
                    continue

                # HIGH sweep → SHORT
                if lvl_name.endswith('high'):
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
                        break

                # LOW sweep → LONG
                if lvl_name.endswith('low'):
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
                        break

        # Window khatam — open trade close karo
        if in_trade:
            last = scan.iloc[-1]
            pnl = ((last['close'] - entry_price) if direction == 1
                   else (entry_price - last['close'])) - spread
            trades.append({
                'date': curr_date,
                'direction': 'LONG' if direction == 1 else 'SHORT',
                'level': level_hit, 'entry_time': entry_time,
                'exit_time': last.name, 'entry': entry_price,
                'exit': last['close'], 'pnl': pnl, 'reason': 'WINDOW_END'})

        if trades_today > 0:
            days_with_data += 1

    print(f"\n  Days scanned: {days_scanned}")
    print(f"  Days with trades: {days_with_data}")
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


if __name__ == "__main__":
    print("=" * 60)
    print(" LIQUIDITY SWEEP v3 — GOLD")
    print("=" * 60)
    df = load_gold()
    if df is None:
        exit(1)

    print(f"\nLevels:")
    print(f"  NY range:     07:00-15:00 EST")
    print(f"  Late range:   21:00-22:00 EST")
    print(f"\nSweep window: 01:00-02:00 EST (next day)")
    print(f"SL=${SL_DOLLARS} (40 pips), TP=${TP_DOLLARS} (80 pips)")
    print(f"Max {MAX_TRADES_DAY} trade/day, Spread=${SPREAD}")

    print("\nRunning backtest...")
    trades = backtest(df)
    if len(trades) == 0:
        print("\nX Koi trade nahi bani.")
        exit(0)

    m = metrics(trades)
    print("\n" + "=" * 60)
    print(" RESULTS")
    print("=" * 60)
    for k, v in m.items():
        print(f"  {k:.<30} {v}")
    print("=" * 60)

    print("\n Level Breakdown:")
    for lvl in ['NY_high', 'NY_low', 'Late_high', 'Late_low']:
        sub = trades[trades['level'] == lvl]
        if len(sub) > 0:
            wins = (sub['pnl_usd'] > 0).sum()
            print(f"  {lvl:.<15} trades={len(sub):>3}  "
                  f"WR={wins/len(sub)*100:.1f}%")

    print("\n Direction Breakdown:")
    for d in ['LONG', 'SHORT']:
        sub = trades[trades['direction'] == d]
        if len(sub) > 0:
            wins = (sub['pnl_usd'] > 0).sum()
            print(f"  {d:.<15} trades={len(sub):>3}  "
                  f"WR={wins/len(sub)*100:.1f}%")

    trades.to_csv(RESULTS_DIR / "sweep_v3_trades.csv", index=False)
    print(f"\nTrades saved: {RESULTS_DIR / 'sweep_v3_trades.csv'}")
