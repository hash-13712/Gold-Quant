# ============================================================
# LIQUIDITY SWEEP STRATEGY — GOLD ONLY
# 07:00 & 21:00 EST levels → next day 01:00 AM se sweep
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_DIR = Path.home() / "gold_quant" / "data"
RESULTS_DIR = Path.home() / "gold_quant" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---- Parameters ----
TIMEFRAME = "5m"
SL_DOLLARS = 4.0          # 40 pips (1 pip = $0.10 for Gold)
RR = 2.0                  # TP = 2 x SL = $8
RISK_PCT = 0.005
ACCOUNT = 10000
SCAN_START_HOUR = 1       # 01:00 AM EST
SCAN_END_HOUR = 23        # Din ke end tak


def load_gold():
    """Gold 5m candles load karo, EST timezone mein."""
    f = DATA_DIR / f"XAUUSD_{TIMEFRAME}.parquet"
    if not f.exists():
        print(f"X Nahi mili: {f}")
        return None

    df = pd.read_parquet(f)
    df.index = pd.to_datetime(df.index)

    # UTC -> EST
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert('US/Eastern')

    print(f"  Bars: {len(df):,}")
    print(f"  Range: {df.index.min()} to {df.index.max()}")
    return df


def get_daily_levels(df):
    """
    Har din ke 07:00 aur 21:00 EST candles ka High/Low nikaalo.
    Returns: {date: {'07_high':..., '07_low':..., '21_high':..., '21_low':...}}
    """
    levels = {}
    for date, group in df.groupby(df.index.date):
        day_l = {}

        # 07:00 candle
        morning = group[group.index.hour == 7]
        if len(morning) > 0:
            day_l['07_high'] = morning['high'].max()
            day_l['07_low'] = morning['low'].min()

        # 21:00 candle
        evening = group[group.index.hour == 21]
        if len(evening) > 0:
            day_l['21_high'] = evening['high'].max()
            day_l['21_low'] = evening['low'].min()

        if day_l:
            levels[date] = day_l

    return levels


def backtest_sweep(df, sl_dollars=SL_DOLLARS, rr=RR):
    """Main backtest loop."""
    levels_by_date = get_daily_levels(df)
    dates = sorted(levels_by_date.keys())

    trades = []
    total_days = 0
    days_with_trades = 0

    for i in range(1, len(dates)):
        prev_date = dates[i - 1]
        curr_date = dates[i]
        prev_levels = levels_by_date[prev_date]

        if not prev_levels:
            continue

        # Aaj ka data 01:00 AM se
        curr_day = df[df.index.date == curr_date]
        scan = curr_day[(curr_day.index.hour >= SCAN_START_HOUR) &
                        (curr_day.index.hour <= SCAN_END_HOUR)]

        if len(scan) == 0:
            continue

        total_days += 1
        day_had_trade = False
        swept = set()
        in_trade = False
        entry_price = stop = target = direction = 0
        entry_time = None
        level_hit = None

        for ts, candle in scan.iterrows():
            if in_trade:
                # ---- EXIT ----
                if direction == 1:  # LONG
                    if candle['low'] <= stop:
                        pnl = -(entry_price - stop)
                        trades.append({
                            'date': curr_date, 'direction': 'LONG',
                            'level': level_hit,
                            'entry_time': entry_time, 'exit_time': ts,
                            'entry': entry_price, 'exit': stop,
                            'pnl_dollars': pnl, 'reason': 'STOP',
                        })
                        in_trade = False
                    elif candle['high'] >= target:
                        pnl = target - entry_price
                        trades.append({
                            'date': curr_date, 'direction': 'LONG',
                            'level': level_hit,
                            'entry_time': entry_time, 'exit_time': ts,
                            'entry': entry_price, 'exit': target,
                            'pnl_dollars': pnl, 'reason': 'TARGET',
                        })
                        in_trade = False
                else:  # SHORT
                    if candle['high'] >= stop:
                        pnl = -(stop - entry_price)
                        trades.append({
                            'date': curr_date, 'direction': 'SHORT',
                            'level': level_hit,
                            'entry_time': entry_time, 'exit_time': ts,
                            'entry': entry_price, 'exit': stop,
                            'pnl_dollars': pnl, 'reason': 'STOP',
                        })
                        in_trade = False
                    elif candle['low'] <= target:
                        pnl = entry_price - target
                        trades.append({
                            'date': curr_date, 'direction': 'SHORT',
                            'level': level_hit,
                            'entry_time': entry_time, 'exit_time': ts,
                            'entry': entry_price, 'exit': target,
                            'pnl_dollars': pnl, 'reason': 'TARGET',
                        })
                        in_trade = False
            else:
                # ---- SWEEP DETECTION ----
                for lvl_name, lvl_price in prev_levels.items():
                    if lvl_name in swept:
                        continue

                    # HIGH sweep → SHORT
                    if lvl_name.endswith('high'):
                        if candle['high'] > lvl_price and candle['close'] < lvl_price:
                            direction = -1
                            entry_price = candle['close']
                            stop = entry_price + sl_dollars
                            target = entry_price - sl_dollars * rr
                            entry_time = ts
                            level_hit = lvl_name
                            swept.add(lvl_name)
                            in_trade = True
                            day_had_trade = True
                            break

                    # LOW sweep → LONG
                    if lvl_name.endswith('low'):
                        if candle['low'] < lvl_price and candle['close'] > lvl_price:
                            direction = 1
                            entry_price = candle['close']
                            stop = entry_price - sl_dollars
                            target = entry_price + sl_dollars * rr
                            entry_time = ts
                            level_hit = lvl_name
                            swept.add(lvl_name)
                            in_trade = True
                            day_had_trade = True
                            break

        # End of day — open trade close karo
        if in_trade:
            last = scan.iloc[-1]
            pnl = ((last['close'] - entry_price) if direction == 1
                   else (entry_price - last['close']))
            trades.append({
                'date': curr_date,
                'direction': 'LONG' if direction == 1 else 'SHORT',
                'level': level_hit,
                'entry_time': entry_time, 'exit_time': last.name,
                'entry': entry_price, 'exit': last['close'],
                'pnl_dollars': pnl, 'reason': 'EOD',
            })

        if day_had_trade:
            days_with_trades += 1

    print(f"\n  Days scanned: {total_days}")
    print(f"  Days with trades: {days_with_trades}")
    return pd.DataFrame(trades)


def metrics(trades, initial=ACCOUNT):
    if len(trades) == 0:
        return {"total_trades": 0}

    # Position sizing: SL $4, risk 0.5% of account
    risk_per_trade = RISK_PCT * initial  # $50
    lots_per_trade = risk_per_trade / (SL_DOLLARS * 100)  # XAUUSD: 1 lot = $100/move

    trades = trades.copy()
    trades['pnl'] = trades['pnl_dollars'] * lots_per_trade * 100

    wins = trades[trades['pnl'] > 0]
    losses = trades[trades['pnl'] <= 0]

    gp = wins['pnl'].sum()
    gl = abs(losses['pnl'].sum())
    pf = gp / gl if gl > 0 else np.inf

    trades['cum'] = trades['pnl'].cumsum()
    eq = initial + trades['cum']
    peak = eq.cummax()
    dd = ((eq - peak) / peak * 100).min()

    return {
        'total_trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate_%': round(len(wins) / len(trades) * 100, 2),
        'avg_win_$': round(wins['pnl'].mean(), 2) if len(wins) else 0,
        'avg_loss_$': round(losses['pnl'].mean(), 2) if len(losses) else 0,
        'expectancy_$': round(trades['pnl'].mean(), 2),
        'profit_factor': round(pf, 2),
        'total_pnl_$': round(trades['pnl'].sum(), 2),
        'total_return_%': round((initial + trades['pnl'].sum() - initial) / initial * 100, 2),
        'max_drawdown_%': round(dd, 2),
        'final_account_$': round(initial + trades['pnl'].sum(), 2),
    }


if __name__ == "__main__":
    print("=" * 60)
    print(" LIQUIDITY SWEEP STRATEGY — GOLD")
    print("=" * 60)

    df = load_gold()
    if df is None:
        exit(1)

    print(f"\nReference levels:")
    print(f"  07:00 AM EST candle High/Low")
    print(f"  21:00 EST candle High/Low")
    print(f"\nScan window: 01:00 AM onwards (next day)")
    print(f"Stop Loss: ${SL_DOLLARS:.2f}")
    print(f"Take Profit: ${SL_DOLLARS * RR:.2f} (RR = {RR})")

    print("\nRunning backtest...")
    trades = backtest_sweep(df)

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

    # Breakdown by level
    print("\n Level Breakdown:")
    for lvl in ['07_high', '07_low', '21_high', '21_low']:
        sub = trades[trades['level'] == lvl]
        if len(sub) > 0:
            wins = (sub['pnl_dollars'] > 0).sum()
            print(f"  {lvl:.<15} trades={len(sub):>3}  wins={wins:>3}  "
                  f"WR={wins/len(sub)*100:.1f}%")

    trades.to_csv(RESULTS_DIR / "sweep_trades.csv", index=False)
    with open(RESULTS_DIR / "sweep_summary.txt", "w") as f:
        f.write(f"Run: {datetime.now()}\n")
        f.write(f"Strategy: Liquidity Sweep (07:00 & 21:00 levels)\n")
        f.write(f"SL=${SL_DOLLARS}, RR={RR}\n\n")
        for k, v in m.items():
            f.write(f"{k}: {v}\n")

    print(f"\nTrades saved: {RESULTS_DIR / 'sweep_trades.csv'}")
    print(f"Summary saved: {RESULTS_DIR / 'sweep_summary.txt'}")
