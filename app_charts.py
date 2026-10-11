# ============================================================
# GOLD/SILVER CANDLE CHART APP
# Streamlit + Plotly — multi-timeframe, interactive
# ============================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

DATA_DIR = Path.home() / "gold_quant" / "data"
CANDLE_DIR = DATA_DIR / "candles"

st.set_page_config(
    page_title="Gold/Silver Charts",
    page_icon="📈",
    layout="wide",
)


@st.cache_data(ttl=600)
def discover():
    symbols = set()
    tfs = set()
    for f in CANDLE_DIR.glob("*.parquet"):
        stem = f.stem  # XAUUSD_5m
        if "_" in stem:
            s, t = stem.rsplit("_", 1)
            symbols.add(s)
            tfs.add(t)
    return sorted(symbols), sorted(tfs)


@st.cache_data(ttl=600)
def load(symbol, tf):
    f = CANDLE_DIR / f"{symbol}_{tf}.parquet"
    if not f.exists():
        return None
    df = pd.read_parquet(f)
    df.index = pd.to_datetime(df.index)
    return df


# ---------------- UI ----------------
st.title("📈 Gold & Silver Candle Charts")

symbols, timeframes = discover()

if not symbols:
    st.error("Koi candle data nahi mila. Pehle `python build_candles.py` chalao.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    symbol = st.selectbox("Symbol", symbols)

    # Timeframes ordered
    tf_order = ["1m", "5m", "15m", "30m", "1h", "4h", "1D"]
    available_tfs = [t for t in tf_order if t in timeframes]
    tf = st.selectbox("Timeframe", available_tfs)

    st.markdown("---")
    st.caption(f"Symbols: {', '.join(symbols)}")
    st.caption(f"Timeframes: {', '.join(available_tfs)}")

# Load data
df = load(symbol, tf)
if df is None:
    st.error(f"Data nahi mila: {symbol}_{tf}.parquet")
    st.stop()

# Date range picker
min_date = df.index.min().date()
max_date = df.index.max().date()

with st.sidebar:
    st.markdown("### 📅 Date Range")
    start_date = st.date_input(
        "Start", min_date, min_value=min_date, max_value=max_date
    )
    end_date = st.date_input(
        "End", max_date, min_value=min_date, max_value=max_date
    )

# Filter
mask = (df.index.date >= start_date) & (df.index.date <= end_date)
view = df[mask]

if len(view) == 0:
    st.warning("Selected range mein koi candle nahi.")
    st.stop()

# ---------------- Stats Row ----------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Candles", f"{len(view):,}")
c2.metric("High", f"{view['high'].max():,.2f}")
c3.metric("Low", f"{view['low'].min():,.2f}")
c4.metric("Close", f"{view['close'].iloc[-1]:,.2f}")
change = view['close'].iloc[-1] - view['open'].iloc[0]
change_pct = (change / view['open'].iloc[0]) * 100
c5.metric("Change", f"{change:+,.2f}", f"{change_pct:+.2f}%")

# ---------------- Chart ----------------
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.8, 0.2],
)

# Candles
fig.add_trace(
    go.Candlestick(
        x=view.index,
        open=view['open'],
        high=view['high'],
        low=view['low'],
        close=view['close'],
        name=symbol,
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350',
        increasing_fillcolor='#26a69a',
        decreasing_fillcolor='#ef5350',
    ),
    row=1, col=1,
)

# Ticks as volume proxy
if 'ticks' in view.columns:
    fig.add_trace(
        go.Bar(
            x=view.index, y=view['ticks'],
            name='Ticks',
            marker_color='#4a90e2',
            opacity=0.6,
        ),
        row=2, col=1,
    )

fig.update_layout(
    title=f"{symbol} — {tf}",
    template='plotly_dark',
    height=750,
    xaxis_rangeslider_visible=False,
    hovermode='x unified',
    showlegend=False,
    margin=dict(l=40, r=40, t=60, b=40),
)

fig.update_xaxes(title_text="Time", row=2, col=1)
fig.update_yaxes(title_text="Price", row=1, col=1)
fig.update_yaxes(title_text="Ticks", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# ---------------- Raw Data ----------------
with st.expander("🔍 Last 20 Candles"):
    st.dataframe(
        view.tail(20).style.format({
            'open': '{:.2f}', 'high': '{:.2f}',
            'low': '{:.2f}', 'close': '{:.2f}',
        }),
        use_container_width=True,
    )

# ---------------- Download ----------------
csv = view.to_csv().encode('utf-8')
st.download_button(
    "⬇️ Download candles (CSV)",
    data=csv,
    file_name=f"{symbol}_{tf}_{start_date}_{end_date}.csv",
    mime='text/csv',
)

st.caption(
    f"Data: {view.index.min()} → {view.index.max()} | "
    f"Timeframe: {tf} | Bars: {len(view):,}"
)
