# ============================================================
# DUKASCOPY DOWNLOADER — FINAL FIXED VERSION
# User-Agent + Retry + Parallel
# ============================================================

import lzma
import struct
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

DATA_DIR = Path.home() / "gold_quant" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = ["XAUUSD", "XAGUSD"]
BASE_URL = "https://datafeed.dukascopy.com/datafeed"

START_DATE = datetime(2026, 6, 1).date()
END_DATE   = datetime(2026, 9, 24).date()

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
    "Accept": "*/*",
    "Referer": "https://www.dukascopy.com/",
}

session = requests.Session()
session.headers.update(HEADERS)


def parse_bi5(content, symbol):
    try:
        decompressed = lzma.decompress(content)
    except Exception:
        return pd.DataFrame()
    point = 1e3 if symbol.startswith(("XAU", "XAG")) else 1e5
    ticks = []
    for i in range(0, len(decompressed), 20):
        chunk = decompressed[i:i + 20]
        if len(chunk) < 20:
            break
        try:
            t_off, ask, bid, av, bv = struct.unpack('>IIIff', chunk)
            ticks.append({'time_offset_ms': t_off, 'ask': ask / point,
                          'bid': bid / point, 'ask_vol': av, 'bid_vol': bv})
        except struct.error:
            continue
    return pd.DataFrame(ticks)


def download_hour(symbol, dt, max_retries=3):
    month_idx = dt.month - 1
    url = (f"{BASE_URL}/{symbol}/{dt.year}/{month_idx:02d}/"
           f"{dt.day:02d}/{dt.hour:02d}h_ticks.bi5")
    for attempt in range(max_retries):
        try:
            r = session.get(url, timeout=20)
            if r.status_code == 404 or len(r.content) == 0:
                return pd.DataFrame()
            if r.status_code != 200:
                return pd.DataFrame()
            df = parse_bi5(r.content, symbol)
            if df.empty:
                return df
            hour_start = dt.replace(minute=0, second=0, microsecond=0)
            df['timestamp'] = hour_start + pd.to_timedelta(df['time_offset_ms'], unit='ms')
            return df[['timestamp', 'bid', 'ask', 'bid_vol', 'ask_vol']]
        except (requests.ConnectionError, requests.Timeout):
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def download_day(symbol, date):
    hours = [datetime(date.year, date.month, date.day, h) for h in range(24)]
    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(download_hour, symbol, dt): dt for dt in hours}
        for future in as_completed(futures):
            dt = futures[future]
            try:
                df = future.result()
                if not df.empty:
                    results[dt] = df
            except Exception:
                pass
    if results:
        return pd.concat([results[dt] for dt in sorted(results.keys())], ignore_index=True)
    return pd.DataFrame()


def download_range(symbol, start_date, end_date):
    all_data = []
    current = start_date
    days_ok = 0
    days_empty = 0
    total_days = sum(1 for d in range((end_date - start_date).days + 1)
                     if (start_date + timedelta(days=d)).weekday() < 5)
    done = 0
    while current <= end_date:
        if current.weekday() < 5:
            done += 1
            t0 = time.time()
            day_df = download_day(symbol, current)
            elapsed = time.time() - t0
            if not day_df.empty:
                all_data.append(day_df)
                days_ok += 1
                print(f"[{symbol}] {current}  OK {len(day_df):>7,} ticks  ({elapsed:.1f}s)  [{done}/{total_days}]")
            else:
                days_empty += 1
                print(f"[{symbol}] {current}  -- no data           ({elapsed:.1f}s)  [{done}/{total_days}]")
            time.sleep(0.3)
        current += timedelta(days=1)
    print(f"\n  Days with data: {days_ok}")
    print(f"  Days empty:     {days_empty}")
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


if __name__ == "__main__":
    print("=" * 60)
    print("DUKASCOPY DATA DOWNLOADER (FINAL)")
    print("=" * 60)
    print(f"Range:  {START_DATE} to {END_DATE}")
    print(f"Folder: {DATA_DIR}")
    print("=" * 60)
    for symbol in SYMBOLS:
        print(f"\n{'=' * 60}\nDOWNLOADING: {symbol}\n{'=' * 60}")
        df = download_range(symbol, START_DATE, END_DATE)
        if not df.empty:
            out_file = DATA_DIR / f"{symbol}_ticks.parquet"
            df.to_parquet(out_file, index=False)
            print(f"\nSaved: {out_file}")
            print(f"  Total ticks: {len(df):,}")
        else:
            print(f"\n{symbol}: no data")
