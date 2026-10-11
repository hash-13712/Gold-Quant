# ============================================================
# CSV TICKS → 1m / 5m / 15m CANDLES
# dukascopy-node CSV ko parquet candles mein convert karta hai
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
import glob

DATA_DIR = Path.home() / "gold_quant" / "data"


def find_csv(symbol_pattern):
    """CSV file dhoondo."""
    files = list(DATA_DIR.glob(f"{symbol_pattern}*.csv"))
    if not files:
        print(f"  ✗ Nahi mili: {symbol_pattern}*.csv")
        return None
    return files[0]


def load_and_convert(symbol, timeframe='5min'):
    """CSV load karo, candles banao."""
    csv_file = find_csv(symbol)
    if csv_file is None:
        return None

    print(f"\n[{symbol.upper()}] Loading: {csv_file.name}")

    # CSV load karo
    df = pd.read_csv(csv_file)
    print(f"  Columns: {list(df.columns)}")
    print(f"  Rows: {len(df):,}")

    # Column names normalize karo (different formats handle karne ke liye)
    df.columns = [c.lower().strip() for c in df.columns]

    # Timestamp column dhoondo
    ts_col = None
    for c in ['timestamp', 'time', 'date', 'datetime']:
        if c in df.columns:
            ts_col = c
            break
    if ts_col is None:
        print(f"  ✗ Timestamp column nahi mila")
        return None

    # Bid/Ask columns dhoondo
    bid_col = None
    ask_col = None
    for c in df.columns:
        if 'bid' in c and 'vol' not in c and 'price' in c:
            bid_col = c
        elif 'bid' in c and 'vol' not in c:
            if bid_col is None:
                bid_col = c
        if 'ask' in c and 'vol' not in c and 'price' in c:
            ask_col = c
        elif 'ask' in c and 'vol' not in c:
            if ask_col is None:
                ask_col = c

    if bid_col is None or ask_col is None:
        print(f"  ✗ Bid/Ask column nahi mile")
        return None

    print(f"  Timestamp: {ts_col}, Bid: {bid_col}, Ask: {ask_col}")

    # Timestamp parse karo
    # Agar integer milliseconds hain
    if df[ts_col].dtype in ['int64', 'int32', 'float64']:
        df['timestamp'] = pd.to_datetime(df[ts_col], unit='ms', utc=True)
    else:
        df['timestamp'] = pd.to_datetime(df[ts_col], utc=True)

    # Mid price
    df['mid'] = (df[bid_col] + df[ask_col]) / 2

    # Index set
    df = df.set_index('timestamp').sort_index()

    # Candles banao
    candles = df['mid'].resample(timeframe).agg(
        open='first',
        high='max',
        low='min',
        close='last'
    ).dropna()

    # Volume (agar available hai)
    vol_col = None
    for c in df.columns:
        if 'bid' in c and 'vol' in c:
            vol_col = c
            break
    if vol_col:
        candles['volume'] = df[vol_col].resample(timeframe).sum()

    # Save
    out_file = DATA_DIR / f"{symbol.upper()}_{timeframe.replace('min','m')}.parquet"
    candles.to_parquet(out_file)

    print(f"  ✓ Saved: {out_file.name}")
    print(f"    Candles: {len(candles):,}")
    print(f"    Range: {candles.index.min()} → {candles.index.max()}")

    return candles


if __name__ == "__main__":
    print("=" * 60)
    print("CSV → CANDLES CONVERTER")
    print("=" * 60)

    # Gold aur Silver ke liye candles banao
    for symbol in ['xauusd', 'xagusd']:
        for tf in ['1min', '5min', '15min']:
            load_and_convert(symbol, tf)

    print("\n" + "=" * 60)
    print("SAB CANDLES BAN GAYI!")
    print("=" * 60)
    print(f"\nFiles in {DATA_DIR}:")
    for f in sorted(DATA_DIR.glob("*.parquet")):
        print(f"  {f.name}")
