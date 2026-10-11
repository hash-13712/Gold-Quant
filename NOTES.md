# Gold Quant Project — Notes

## Status (Oct 2026)
- 3 mahine ka data: Oct-Dec 2024
- Q1 2024: downloaded
- Strategy: Liquidity sweep (NY 07-15 → Late 21-22 target)
- Backtest WR: ~50% (LONG), PF ~2.9

## Kya Kaam Karta Hai
- Dukascopy data download
- Multi-timeframe candles (7 TFs)
- Streamlit Plotly chart app

## Kya Nahi Hua
- Full automated logic capture
- Conditional reasoning (pending liquidity, context)
- Human hindsight AI mein nahi aa sakta

## Next Steps (Jab Wapas Aao)
1. 10 saal ka data download (2015-2024)
2. Quarter-wise download (Q1, Q2, Q3, Q4)
3. Har quarter ~1 ghanta
4. Total: ~4-5 din ka kaam (raat bhar chalao)
5. Storage: ~30-40 GB

## Files on Laptop
- ~/gold_quant/data/*.csv           — raw ticks
- ~/gold_quant/data/candles/*.parquet — candles
- ~/gold_quant/build_candles.py     — candle builder
- ~/gold_quant/sweep_v8.py          — backtest
- ~/gold_quant/app_tv.py            — chart app
- ~/gold_quant/app_verify.py        — trade verification

## Commands (Cheat Sheet)
```bash
# Activate
cd ~/gold_quant && source venv/bin/activate

# Data download (per quarter)
npx dukascopy-node -i xauusd -from 2024-01-01 -to 2024-03-31 -t tick -f csv --batch-size 2 --batch-pause 3000

# Build candles
python build_candles.py

# Chart app
streamlit run app_tv.py

# Backtest
python sweep_v8.pyV
