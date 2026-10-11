# ============================================================
# LIQUIDITY SWEEP v7 — NY sweep → Late target (DOL)
# Min target 70 pips, max 200 pips
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
RISK_PCT = 0.005
ACCOUNT = 10000
SPREAD = 0.30

NY_START, NY_END = 7, 15
LATE_START, LATE_END = 21, 22
SWEEP_START, SWEEP_END = 1, 4

MIN_TARGET_PIPS = 70
MAX_TARGET_PIPS = 200
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


def pick_target(direction, prev_lvl, entry_price):
    candidates = []
    if direction == 1:
        for name in ['Late_high', 'NY_high']:
            if name in prev_lvl:
                t = prev_lvl[name]
                if t > entry_price:
                    candidates.append((name, t))
    else:
        for name in ['Late_low', 'NY_low']:
            if name in prev_lvl:
                t = prev_lvl[name]
                if t < entry_price:
                    candidates.append((name, t))
    if not candidates:
        return None, None
    late = [c for c in candidates if 'Late' in c[0]]
    if late:
        return late[0]
    return candidates[0]


def backtest(df, sl=SL_DOLLARS, spread=SPREAD, max_per_day=MAX_TRADES_DAY):
    levels = get_daily_levels(df)
    dates = sorted(levels.keys())

    trades = []
    days_scanned = 0
    skipped_small_target = 0
    skipped_no_target = 0

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
        target_level = None
        entry_pos = None

        for pos, (ts, candle) in enumerate(scan_window.iterrows()):
            if trades_today >= max_per_day:
                break
            for lvl_name, lvl_price in prev_lvl.items():
                if lvl_name in swept:
                    continue

                if lvl_name == 'NY_high':
                    if candle['high'] > lvl_price and candle['close'] < lvl_price:
                        direction = -1
                        entry_price = candle['close']
                        stop = entry_price + sl
                        t_name, t_price = pick_target(direction, prev_lvl, entry_price)
                        if t_price is None:
                            skipped_no_target += 1
                            swept.add(lvl_name)
                            break
                        target_pips = (entry_price - t_price) / 0.10
                        if target_pips < MIN_TARGET_PIPS:
                            skipped_small_target += 1
                            swept.add(lvl_name)
                            break
                        if target_pips > MAX_TARGET_PIPS:
                            t_price = entry_price - MAX_TARGET_PIPS * 0.10
                        target = t_price
                        target_level = t_name
                        entry_time = ts
                        level_hit = lvl_name
                        swept.add(lvl_name)
                        in_trade = True
                        trades_today += 1
                        entry_pos = curr_day.index.get_loc(ts)
                        break

                if lvl_name == 'NY_low':
                    if candle['low'] < lvl_price and candle['close'] > lvl_price:
                        direction = 1
                        entry_price = candle['close']
                        stop = entry_price - sl
                        t_name, t_price = pick_target(direction, prev_lvl, entry_price)
                        if t_price is None:
                            skipped_no_target += 1
                            swept.add(lvl_name)
                            break
                        target_pips = (t_price - entry_price) / 0.10
                        if target_pips < MIN_TARGET_PIPS:
                            skipped_small_target += 1
                            swept.add(lvl_name)
                            break
                        if target_pips > MAX_TARGET_PIPS:
                            t_price = entry_price + MAX_TARGET_PIPS * 0.10
                        target = t_price
                        target_level = t_name
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
            'level': level_hit, 'target_level': target_level,
            'entry_time': entry_time, 'exit_time': exit_time,
            'entry': entry_price, 'exit': exit_price,
            'target': target, 'stop': stop,
            'pnl': raw_pnl, 'reason': exit_reason,
        })

    print(f"\n  Days scanned: {days_scanned}")
    print(f"  Skipped (target < {MIN_TARGET_PIPS} pips): {skipped_small_target}")
    print(f"  Skipped (no valid target): {skipped_no_target}")
    return pd.DataFrame(trades)


if __name__ == "__main__":
    print("=" * 60)
    print(" LIQUIDITY SWEEP v7 — NY sweep → Late target (DOL)")
    print("=" * 60)
    df = load_gold()
    if df is None:
        exit(1)

    print(f"\nLogic:")
    print(f"  NY_low swept  -> LONG  -> target Late_high (DOL)")
    print(f"  NY_high swept -> SHORT -> target Late_low  (DOL)")
    print(f"  Fallback: NY opposite side")
    print(f"  Entry: {SWEEP_START}:00-{SWEEP_END}:00 EST")
    print(f"  SL: ${SL_DOLLARS} (40 pips)")
    print(f"  Target: {MIN_TARGET_PIPS}-{MAX_TARGET_PIPS} pips")

    print("\nRunning backtest...")
    trades = backtest(df)
    if len(trades) == 0:
        print("\nX Koi trade nahi bani.")
        exit(0)

    # pnl_usd calculate karo PEHLE
    risk_per_trade = RISK_PCT * ACCOUNT
    lots = risk_per_trade / (SL_DOLLARS * 100)
    trades['pnl_usd'] = trades['pnl'] * lots * 100

    # ---- Metrics ----
    wins = trades[trades['pnl_usd'] > 0]
    losses = trades[trades['pnl_usd'] <= 0]
    gp = wins['pnl_usd'].sum()
    gl = abs(losses['pnl_usd'].sum())
    pf = gp / gl if gl > 0 else np.inf
    trades['cum'] = trades['pnl_usd'].cumsum()
    eq = ACCOUNT + trades['cum']
    dd = ((eq - eq.cummax()) / eq.cummax() * 100).min()
    final_acc = ACCOUNT + trades['pnl_usd'].sum()

    print("\n" + "=" * 60)
    print(" RESULTS")
    print("=" * 60)
    print(f"  total_trades.................. {len(trades)}")
    print(f"  wins.......................... {len(wins)}")
    print(f"  losses........................ {len(losses)}")
    print(f"  win_rate_%.................... {round(len(wins)/len(trades)*100, 2)}")
    print(f"  avg_win_$..................... {round(wins['pnl_usd'].mean(), 2) if len(wins) else 0}")
    print(f"  avg_loss_$.................... {round(losses['pnl_usd'].mean(), 2) if len(losses) else 0}")
    print(f"  expectancy_$.................. {round(trades['pnl_usd'].mean(), 2)}")
    print(f"  profit_factor................. {round(pf, 2)}")
    print(f"  total_pnl_$................... {round(trades['pnl_usd'].sum(), 2)}")
    print(f"  total_return_%................ {round(trades['pnl_usd'].sum()/ACCOUNT*100, 2)}")
    print(f"  max_drawdown_%................ {round(dd, 2)}")
    print(f"  final_account_$............... {round(final_acc, 2)}")
    print("=" * 60)

    # ---- Level Breakdown ----
    print("\n Level Breakdown:")
    for lvl in trades['level'].unique():
        sub = trades[trades['level'] == lvl]
        w = (sub['pnl_usd'] > 0).sum()
        print(f"  {lvl:.<15} trades={len(sub):>3}  WR={w/len(sub)*100:.1f}%")

    print("\n Target Level Breakdown:")
    for tl in trades['target_level'].dropna().unique():
        sub = trades[trades['target_level'] == tl]
        w = (sub['pnl_usd'] > 0).sum()
        print(f"  {tl:.<15} trades={len(sub):>3}  WR={w/len(sub)*100:.1f}%")

    print("\n Direction Breakdown:")
    for d in ['LONG', 'SHORT']:
        sub = trades[trades['direction'] == d]
        if len(sub) > 0:
            w = (sub['pnl_usd'] > 0).sum()
            print(f"  {d:.<15} trades={len(sub):>3}  WR={w/len(sub)*100:.1f}%")

    print("\n Exit Reason Breakdown:")
    for r in trades['reason'].unique():
        sub = trades[trades['reason'] == r]
        print(f"  {r:.<15} trades={len(sub):>3}")

    trades['target_pips'] = (abs(trades['target'] - trades['entry']) / 0.10).round(0)
    print(f"\n Target distance: mean={trades['target_pips'].mean():.0f} pips, "
          f"median={trades['target_pips'].median():.0f} pips")

    trades.to_csv(RESULTS_DIR / "sweep_v7_trades.csv", index=False)
    print(f"\nTrades saved: {RESULTS_DIR / 'sweep_v7_trades.csv'}")
