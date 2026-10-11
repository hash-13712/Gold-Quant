# ============================================================
# CHART APP — Plotly (Fast + Range Slider)
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
    symbols, tfs = set(), set()
    for f in CANDLE_DIR.glob("*.parquet"):
        s, t = f.stem.rsplit("_", 1)
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
st.title("📈 Gold & Silver Charts")

symbols, timeframes = discover()
if not symbols:
    st.error("Candle data nahi mila. `python build_candles.py` chalao.")
    st.stop()

with st.sidebar:
    st.header("⚙️ Settings")
    symbol = st.selectbox("Symbol", symbols)

    # Timezone selector
    TIMEZONES = {
        "UTC": "UTC",
        "New York (EST)": "US/Eastern",
        "London (GMT)": "Europe/London",
        "Karachi (PKT)": "Asia/Karachi",
        "Dubai (GST)": "Asia/Dubai",
        "Tokyo (JST)": "Asia/Tokyo",
        "Sydney (AEDT)": "Australia/Sydney",
    }
    tz_label = st.selectbox("Timezone", list(TIMEZONES.keys()), index=1)
    tz = TIMEZONES[tz_label]

    tf_order = ["1m", "5m", "15m", "30m", "1h", "4h", "1D"]
    available_tfs = [t for t in tf_order if t in timeframes]
    tf = st.selectbox("Timeframe", available_tfs)

    st.markdown("---")
    st.markdown("### 🎨 Chart Type")
    chart_type = st.radio(
        "Type", ["Candles", "Bars", "Line"], horizontal=True
    )

    st.markdown("---")
    st.markdown("### 📊 Candles to show")
    max_candles = st.slider(
        "Last N candles",
        min_value=200, max_value=5000, value=1000, step=100,
    )

# Load data
df_full = load(symbol, tf)
if df_full is None:
    st.error(f"Data nahi mila: {symbol}_{tf}.parquet")
    st.stop()

# Timezone convert karo
df_full = df_full.copy()
if df_full.index.tz is None:
    df_full.index = df_full.index.tz_localize('UTC')
df_full.index = df_full.index.tz_convert(tz)

# Show only last N candles
df = df_full.tail(max_candles)

# ---------------- Stats Row ----------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Candles", f"{len(df):,}")
c2.metric("High", f"{df['high'].max():,.2f}")
c3.metric("Low", f"{df['low'].min():,.2f}")
c4.metric("Close", f"{df['close'].iloc[-1]:,.2f}")
chg = df['close'].iloc[-1] - df['open'].iloc[0]
chg_pct = (chg / df['open'].iloc[0]) * 100
c5.metric("Change", f"{chg:+,.2f}", f"{chg_pct:+.2f}%")

# ---------------- Chart ----------------
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    vertical_spacing=0.03, row_heights=[0.75, 0.25],
)

# Trace by type
if chart_type == "Candles":
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df['open'], high=df['high'],
            low=df['low'], close=df['close'], name=symbol,
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
        ),
        row=1, col=1,
    )
elif chart_type == "Bars":
    fig.add_trace(
        go.Ohlc(
            x=df.index, open=df['open'], high=df['high'],
            low=df['low'], close=df['close'], name=symbol,
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
        ),
        row=1, col=1,
    )
else:
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['close'], name=symbol,
            line=dict(color='#4a90e2', width=2),
        ),
        row=1, col=1,
    )

# Ticks volume
if 'ticks' in df.columns:
    fig.add_trace(
        go.Bar(
            x=df.index, y=df['ticks'], name='Ticks',
            marker_color='#4a90e2', opacity=0.6,
        ),
        row=2, col=1,
    )

fig.update_layout(
    title=f"{symbol} — {tf} (last {len(df):,} candles)",
    template='plotly_dark',
    height=700,
    hovermode='x unified',
    showlegend=False,
    dragmode='pan',
    margin=dict(l=40, r=40, t=50, b=40),
)

fig.update_xaxes(
    rangeslider_visible=False,
    rangebreaks=[dict(bounds=["sat", "mon"])],
)
fig.update_yaxes(title_text="Price", row=1, col=1)
if 'ticks' in df.columns:
    fig.update_yaxes(title_text="Ticks", row=2, col=1)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        'scrollZoom': True,
        'displayModeBar': True,
        'displaylogo': False,
    },
)

# ---------------- Raw Data ----------------
with st.expander("🔍 Last 20 Candles"):
    st.dataframe(
        df.tail(20).style.format({
            'open': '{:.2f}', 'high': '{:.2f}',
            'low': '{:.2f}', 'close': '{:.2f}',
        }),
        use_container_width=True,
    )

st.caption(
    f"Full data: {df_full.index.min()} → {df_full.index.max()} | "
    f"Showing last {len(df):,} of {len(df_full):,} candles"
)
