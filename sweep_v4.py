# ============================================================
# LIQUIDITY SWEEP v4 — GOLD
# Fix: Entry 01:00-02:00, lekin trade poore din chale
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
LATE_START, LATE_END = 21, 22
SWEEP_START, SWEEP_END = 1, 2

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
    levels = {}
    for date, group in df.groupby(df.index.date):
        day_l = {}
        ny = group[(group.index.hour >= NY_START) & (group.index.hour < NY_END)]
        if len(ny) > 50:
            day_l['NY_high'] = ny['high'].max()
            day_l['NY_low'] = ny['low'].min()
        late = group[(group.index.hour >= LATE_START) & (group.index.hour < LATE_END)]
        if len(late) > 6:
            day_l['Late_high'] = late['high'].max()
            day_l['Late_low'] = late['low'].min()
        if len(day_l) == 4:
            levels[date] = day_l
    return levels


def backtest(df, sl=SL_DOLLARS, tp=TP_DOLLARS, spread=SPREAD, max_per_day=MAX_TRADES_DAY):
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
        entry_pos = None  # curr_day mein index position

        # Phase 1: Scan window mein entry
        for pos, (ts, candle) in enumerate(scan_window.iterrows()):
            if trades_today >= max_per_day:
                break
            for lvl_name, lvl_price in prev_lvl.items():
                if lvl_name in swept:
                    continue
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
                        entry_pos = curr_day.index.get_loc(ts)
                        break
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
                        entry_pos = curr_day.index.get_loc(ts)
                        break
            if in_trade:
                break

        if not in_trade:
            continue

        # Phase 2: Poora din trade ko chalao
        remaining = curr_day.iloc[entry_pos + 1:]
        exit_price = None
        exit_reason = None
        exit_time = None

        for ts, candle in remaining.iterrows():
            if direction == 1:
                if candle['low'] <= stop:
                    exit_price, exit_reason, exit_time = stop, 'STOP', ts
                    break
                elif candle['high'] >= target:
                    exit_price, exit_reason, exit_time = target, 'TARGET', ts
                    break
            else:
                if candle['high'] >= stop:
                    exit_price, exit_reason, exit_time = stop, 'STOP', ts
                    break
                elif candle['low'] <= target:
                    exit_price, exit_reason, exit_time = target, 'TARGET', ts
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


if __name__ == "__main__":
    print("=" * 60)
    print(" LIQUIDITY SWEEP v4 — Trade runs full day")
    print("=" * 60)
    df = load_gold()
    if df is None:
        exit(1)

    print(f"\nLevels: NY 07:00-15:00 + Late 21:00-22:00 (prev day)")
    print(f"Entry: 01:00-02:00 EST only")
    print(f"Exit: SL/TP anytime, or EOD")
    print(f"SL=${SL_DOLLARS}, TP=${TP_DOLLARS}")

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

    # Manual pnl_usd add karo breakdown ke liye
    risk_per_trade = RISK_PCT * ACCOUNT
    lots = risk_per_trade / (SL_DOLLARS * 100)
    trades['pnl_usd'] = trades['pnl'] * lots * 100

    print("\n Level Breakdown:")
    for lvl in ['NY_high', 'NY_low', 'Late_high', 'Late_low']:
        sub = trades[trades['level'] == lvl]
        if len(sub) > 0:
            wins = (sub['pnl_usd'] > 0).sum()
            print(f"  {lvl:.<15} trades={len(sub):>3}  WR={wins/len(sub)*100:.1f}%")

    print("\n Direction Breakdown:")
    for d in ['LONG', 'SHORT']:
        sub = trades[trades['direction'] == d]
        if len(sub) > 0:
            wins = (sub['pnl_usd'] > 0).sum()
            print(f"  {d:.<15} trades={len(sub):>3}  WR={wins/len(sub)*100:.1f}%")

    print("\n Exit Reason Breakdown:")
    for r in trades['reason'].unique():
        sub = trades[trades['reason'] == r]
        wins = (sub['pnl_usd'] > 0).sum()
        print(f"  {r:.<15} trades={len(sub):>3}  WR={wins/len(sub)*100:.1f}%")

    trades.to_csv(RESULTS_DIR / "sweep_v4_trades.csv", index=False)
    print(f"\nTrades saved: {RESULTS_DIR / 'sweep_v4_trades.csv'}")
