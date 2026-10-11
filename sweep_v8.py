# ============================================================
# LIQUIDITY SWEEP v8 — FINAL
# Fixes: No fallback (skip trade), LONG only option, cap 250 pips
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
MAX_TARGET_PIPS = 250
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


def backtest(df, sl=SL_DOLLARS, spread=SPREAD,
             long_only=False, short_only=False,
             allow_fallback=False, max_per_day=MAX_TRADES_DAY):
    """
    long_only / short_only: direction filter
    allow_fallback: agar False, sirf Late level targets use karo
    """
    levels = get_daily_levels(df)
    dates = sorted(levels.keys())

    trades = []
    days_scanned = 0
    skipped_small = 0
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

                # NY HIGH sweep -> SHORT
                if lvl_name == 'NY_high' and not long_only:
                    if candle['high'] > lvl_price and candle['close'] < lvl_price:
                        # Only use Late_low as target
                        if 'Late_low' in prev_lvl:
                            t_price = prev_lvl['Late_low']
                            if t_price < candle['close']:
                                target_pips = (candle['close'] - t_price) / 0.10
                                if target_pips < MIN_TARGET_PIPS:
                                    skipped_small += 1
                                    swept.add(lvl_name)
                                    break
                                if target_pips > MAX_TARGET_PIPS:
                                    t_price = candle['close'] - MAX_TARGET_PIPS * 0.10
                                direction = -1
                                entry_price = candle['close']
                                stop = entry_price + sl
                                target = t_price
                                target_level = 'Late_low'
                                entry_time = ts
                                level_hit = lvl_name
                                swept.add(lvl_name)
                                in_trade = True
                                trades_today += 1
                                entry_pos = curr_day.index.get_loc(ts)
                                break
                            else:
                                skipped_no_target += 1
                                swept.add(lvl_name)
                                break
                        else:
                            skipped_no_target += 1
                            swept.add(lvl_name)
                            break

                # NY LOW sweep -> LONG
                if lvl_name == 'NY_low' and not short_only:
                    if candle['low'] < lvl_price and candle['close'] > lvl_price:
                        if 'Late_high' in prev_lvl:
                            t_price = prev_lvl['Late_high']
                            if t_price > candle['close']:
                                target_pips = (t_price - candle['close']) / 0.10
                                if target_pips < MIN_TARGET_PIPS:
                                    skipped_small += 1
                                    swept.add(lvl_name)
                                    break
                                if target_pips > MAX_TARGET_PIPS:
                                    t_price = candle['close'] + MAX_TARGET_PIPS * 0.10
                                direction = 1
                                entry_price = candle['close']
                                stop = entry_price - sl
                                target = t_price
                                target_level = 'Late_high'
                                entry_time = ts
                                level_hit = lvl_name
                                swept.add(lvl_name)
                                in_trade = True
                                trades_today += 1
                                entry_pos = curr_day.index.get_loc(ts)
                                break
                            else:
                                skipped_no_target += 1
                                swept.add(lvl_name)
                                break
                        else:
                            skipped_no_target += 1
                            swept.add(lvl_name)
                            break
            if in_trade:
                break

        if not in_trade:
            continue

        # Trade management
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
            'level_price': prev_lvl.get(level_hit),
            'NY_high': prev_lvl.get('NY_high'),
            'NY_low': prev_lvl.get('NY_low'),
            'Late_high': prev_lvl.get('Late_high'),
            'Late_low': prev_lvl.get('Late_low'),
            'entry_time': entry_time, 'exit_time': exit_time,
            'entry': entry_price, 'exit': exit_price,
            'target': target, 'stop': stop,
            'pnl': raw_pnl, 'reason': exit_reason,
        })

    print(f"\n  Days scanned: {days_scanned}")
    print(f"  Skipped (<{MIN_TARGET_PIPS} pips): {skipped_small}")
    print(f"  Skipped (no Late target): {skipped_no_target}")
    return pd.DataFrame(trades)


def calc_metrics(trades):
    if len(trades) == 0:
        return None
    risk = RISK_PCT * ACCOUNT
    lots = risk / (SL_DOLLARS * 100)
    t = trades.copy()
    t['pnl_usd'] = t['pnl'] * lots * 100

    wins = t[t['pnl_usd'] > 0]
    losses = t[t['pnl_usd'] <= 0]
    gp = wins['pnl_usd'].sum()
    gl = abs(losses['pnl_usd'].sum())
    pf = gp / gl if gl > 0 else np.inf
    t['cum'] = t['pnl_usd'].cumsum()
    eq = ACCOUNT + t['cum']
    dd = ((eq - eq.cummax()) / eq.cummax() * 100).min()

    return {
        'trades': len(t), 'wins': len(wins), 'losses': len(losses),
        'WR': round(len(wins) / len(t) * 100, 1),
        'avg_win': round(wins['pnl_usd'].mean(), 2) if len(wins) else 0,
        'avg_loss': round(losses['pnl_usd'].mean(), 2) if len(losses) else 0,
        'expectancy': round(t['pnl_usd'].mean(), 2),
        'PF': round(pf, 2),
        'total_pnl': round(t['pnl_usd'].sum(), 2),
        'return_%': round(t['pnl_usd'].sum() / ACCOUNT * 100, 2),
        'DD_%': round(dd, 2),
        'final': round(ACCOUNT + t['pnl_usd'].sum(), 2),
    }, t


def show(label, trades):
    r = calc_metrics(trades)
    if r is None:
        print(f"  {label}: no trades")
        return None
    m, _ = r
    print(f"  {label:<28} T={m['trades']:>3}  WR={m['WR']:>5}%  "
          f"PF={m['PF']:>5}  Exp=${m['expectancy']:>7}  DD={m['DD_%']:>6}%")
    return m, _


if __name__ == "__main__":
    print("=" * 70)
    print(" LIQUIDITY SWEEP v8 — FINAL")
    print("=" * 70)
    df = load_gold()
    if df is None:
        exit(1)

    print(f"\nParameters:")
    print(f"  Levels: NY 07:00-15:00 + Late 21:00-22:00 (prev day)")
    print(f"  Entry: 01:00-04:00 EST")
    print(f"  SL: ${SL_DOLLARS} (40 pips)")
    print(f"  Target: Late level only ({MIN_TARGET_PIPS}-{MAX_TARGET_PIPS} pips)")
    print(f"  No fallback: trade skip if no valid Late target")

    print("\n" + "=" * 70)
    print(" CONFIG COMPARISON")
    print("=" * 70)

    results = {}

    print("\n[1] Baseline v8 — Both directions, no fallback")
    t1 = backtest(df)
    r = show("v8 baseline", t1)
    if r:
        results['v8 both'] = r

    print("\n[2] LONG only")
    t2 = backtest(df, long_only=True)
    r = show("v8 LONG only", t2)
    if r:
        results['v8 LONG'] = r

    print("\n[3] SHORT only")
    t3 = backtest(df, short_only=True)
    r = show("v8 SHORT only", t3)
    if r:
        results['v8 SHORT'] = r

    # Save the best — use LONG only (bull market bias)
    best_trades = t2 if len(t2) > 0 else t1
    m, t_final = calc_metrics(best_trades)

    print("\n" + "=" * 70)
    print(" DETAILED RESULTS — v8 LONG only")
    print("=" * 70)
    for k, v in m.items():
        print(f"  {k:.<25} {v}")

    print("\n Target Level Breakdown:")
    for tl in t_final['target_level'].dropna().unique():
        sub = t_final[t_final['target_level'] == tl]
        w = (sub['pnl_usd'] > 0).sum()
        print(f"  {tl:.<15} trades={len(sub):>3}  WR={w/len(sub)*100:.1f}%")

    print("\n Exit Reason Breakdown:")
    for r_ in t_final['reason'].unique():
        sub = t_final[t_final['reason'] == r_]
        print(f"  {r_:.<15} trades={len(sub):>3}")

    t_final['target_pips'] = (abs(t_final['target'] - t_final['entry']) / 0.10).round(0)
    print(f"\n Target distance: mean={t_final['target_pips'].mean():.0f} pips, "
          f"median={t_final['target_pips'].median():.0f} pips")

    # Per month breakdown
    print("\n Per-Month Performance:")
    t_final['month'] = pd.to_datetime(t_final['date']).dt.to_period('M')
    for month in t_final['month'].unique():
        sub = t_final[t_final['month'] == month]
        w = (sub['pnl_usd'] > 0).sum()
        pnl = sub['pnl_usd'].sum()
        print(f"  {month}  trades={len(sub):>2}  WR={w/len(sub)*100:>5.1f}%  "
              f"PnL=${pnl:>8.2f}")

    t_final.to_csv(RESULTS_DIR / "sweep_v8_trades.csv", index=False)
    print(f"\nTrades saved: {RESULTS_DIR / 'sweep_v8_trades.csv'}")

    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    print(f"{'Config':<20} {'Trades':>7} {'WR%':>7} {'PF':>7} {'Exp$':>8} {'DD%':>7}")
    print("-" * 70)
    for name, (m_dict, _) in results.items():
        print(f"{name:<20} {m_dict['trades']:>7} {m_dict['WR']:>7} "
              f"{m_dict['PF']:>7} {m_dict['expectancy']:>8} {m_dict['DD_%']:>7}")
