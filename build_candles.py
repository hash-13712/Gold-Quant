# ============================================================
# TICK DATA → MULTI-TIMEFRAME CANDLES
# Saari CSVs ek saath merge, 7 timeframes banao
# ============================================================

import pandas as pd
from pathlib import Path

DATA_DIR = Path.home() / "gold_quant" / "data"
CANDLE_DIR = DATA_DIR / "candles"
CANDLE_DIR.mkdir(exist_ok=True)

TIMEFRAMES = {
    "1m":  "1min",
    "5m":  "5min",
    "15m": "15min",
    "30m": "30min",
    "1h":  "1h",
    "4h":  "4h",
    "1D":  "1D",
}

SYMBOLS = ["xauusd", "xagusd"]


def load_all_ticks(symbol):
    files = sorted(DATA_DIR.glob(f"{symbol}*.csv"))
    if not files:
        print(f"  X Koi CSV nahi mili: {symbol}*.csv")
        return None

    print(f"  {len(files)} CSV file(s) mili:")
    for f in files:
        print(f"    {f.name}")

    all_chunks = []
    chunk_size = 2_000_000

    for f in files:
        print(f"  Loading: {f.name}")
        total = 0
        for chunk in pd.read_csv(f, chunksize=chunk_size):
            chunk.columns = [c.lower().strip() for c in chunk.columns]
            chunk['timestamp'] = pd.to_datetime(
                chunk['timestamp'], unit='ms', utc=True
            )
            chunk['mid'] = (chunk['bidprice'] + chunk['askprice']) / 2
            all_chunks.append(chunk[['timestamp', 'mid']])
            total += len(chunk)
        print(f"    {total:,} ticks")

    df = pd.concat(all_chunks, ignore_index=True)
    df = df.sort_values('timestamp').set_index('timestamp')
    return df


def build_for_symbol(symbol):
    print(f"\n{'=' * 60}")
    print(f"Processing {symbol.upper()}")
    print(f"{'=' * 60}")

    df = load_all_ticks(symbol)
    if df is None:
        return

    print(f"\n  Total ticks: {len(df):,}")
    print(f"  Range: {df.index.min()} → {df.index.max()}")

    for tf_label, tf_code in TIMEFRAMES.items():
        print(f"\n  Building {tf_label}...")

        candles = df['mid'].resample(tf_code).agg(
            open='first', high='max', low='min', close='last'
        )
        candles['ticks'] = df['mid'].resample(tf_code).count()
        candles = candles.dropna()

        out = CANDLE_DIR / f"{symbol.upper()}_{tf_label}.parquet"
        candles.to_parquet(out)
        print(f"    OK {len(candles):>7,} candles → {out.name}")


if __name__ == "__main__":
    print("=" * 60)
    print("BUILDING CANDLES — ALL TIMEFRAMES")
    print("=" * 60)

    for s in SYMBOLS:
        build_for_symbol(s)

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nFiles in {CANDLE_DIR}:")
    for f in sorted(CANDLE_DIR.glob("*.parquet")):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name:.<30} {size_mb:>6.2f} MB")
