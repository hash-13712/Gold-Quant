# ============================================================
# CHART WITH INSTITUTIONAL INDICATORS
# VWAP + Session Levels (Asia/London/NY) + ORB
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

DATA_DIR = Path.home() / "gold_quant" / "data"
CANDLE_DIR = DATA_DIR / "candles"

st.set_page_config(page_title="Institutional Chart", layout="wide")
st.title("📊 Gold — Institutional Indicators")

# ---------------- CONFIG ----------------
TZ = 'US/Eastern'

# Session times (EST)
SESSIONS = {
    'Asia':    (19, 2),    # 19:00 to 02:00 next day
    'London':  (2, 7),     # 02:00 to 07:00
    'NY':      (7, 16),    # 07:00 to 16:00
}

# ---------------- LOADERS ----------------
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
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert(TZ)
    return df


# ---------------- INDICATORS ----------------
def add_vwap(df):
    """Session VWAP with SD bands (daily reset)."""
    df = df.copy()
    df['date'] = df.index.date

    # Typical price
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3

    # Weighted (use ticks as volume proxy)
    if 'ticks' in df.columns:
        df['vol'] = df['ticks']
    else:
        df['vol'] = 1

    df['cum_vol'] = df.groupby('date')['vol'].cumsum()
    df['cum_tpv'] = df.groupby('date').apply(
        lambda x: (x['tp'] * x['vol']).cumsum()
    ).reset_index(level=0, drop=True)

    df['vwap'] = df['cum_tpv'] / df['cum_vol']

    # SD bands (VWAP deviation)
    df['dev_sq'] = (df['tp'] - df['vwap']) ** 2
    df['cum_dev_sq'] = df.groupby('date').apply(
        lambda x: (x['dev_sq'] * x['vol']).cumsum()
    ).reset_index(level=0, drop=True)
    df['variance'] = df['cum_dev_sq'] / df['cum_vol']
    df['sd'] = np.sqrt(df['variance'])

    df['vwap_upper_1'] = df['vwap'] + df['sd']
    df['vwap_lower_1'] = df['vwap'] - df['sd']
    df['vwap_upper_2'] = df['vwap'] + 2 * df['sd']
    df['vwap_lower_2'] = df['vwap'] - 2 * df['sd']

    return df


def add_session_levels(df):
    """Asia, London, NY High/Low per day."""
    levels = {}
    for date, group in df.groupby(df.index.date):
        day_l = {'date': date}

        for sess_name, (start_h, end_h) in SESSIONS.items():
            if start_h < end_h:
                sess = group[(group.index.hour >= start_h) & (group.index.hour < end_h)]
            else:
                # Cross midnight (Asia 19-02)
                sess = group[(group.index.hour >= start_h) | (group.index.hour < end_h)]

            if len(sess) > 3:
                day_l[f'{sess_name}_high'] = sess['high'].max()
                day_l[f'{sess_name}_low'] = sess['low'].min()

        levels[date] = day_l
    return levels


def add_orb(df):
    """NY Opening Range (first 15 min at 9:30 EST)."""
    df = df.copy()
    df['orb_high'] = np.nan
    df['orb_low'] = np.nan

    for date, group in df.groupby(df.index.date):
        # 9:30 to 9:45 EST
        orb = group[(group.index.hour == 9) & (group.index.minute >= 30) & (group.index.minute < 45)]
        if len(orb) >= 2:
            orb_h = orb['high'].max()
            orb_l = orb['low'].min()
            # Apply to whole day
            mask = df.index.date == date
            df.loc[mask, 'orb_high'] = orb_h
            df.loc[mask, 'orb_low'] = orb_l

    return df


# ---------------- UI ----------------
symbols, timeframes = discover()
if not symbols:
    st.error("Data nahi mila")
    st.stop()

with st.sidebar:
    st.header("⚙️ Settings")
    symbol = st.selectbox("Symbol", symbols)

    tf_order = ["1m", "5m", "15m", "30m", "1h"]
    available_tfs = [t for t in tf_order if t in timeframes]
    tf = st.selectbox("Timeframe", available_tfs)

    st.markdown("---")
    st.markdown("### 📊 Indicators")

    show_vwap = st.checkbox("VWAP + SD Bands", value=True)
    show_sessions = st.checkbox("Session Levels (Asia/London/NY)", value=True)
    show_orb = st.checkbox("Opening Range (ORB)", value=True)

    st.markdown("---")
    st.markdown("### 📅 Range")
    max_candles = st.slider("Last N candles", 200, 5000, 500, 100)

# Load
df_full = load(symbol, tf)
if df_full is None:
    st.error(f"Data nahi: {symbol}_{tf}")
    st.stop()

# Compute indicators
with st.spinner("Computing indicators..."):
    df_full = add_vwap(df_full)
    df_full = add_orb(df_full)
    session_levels = add_session_levels(df_full)

df = df_full.tail(max_candles)

# ---------------- CHART ----------------
fig = go.Figure()

# Candles
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df['open'], high=df['high'],
    low=df['low'], close=df['close'],
    name=symbol,
    increasing_line_color='#26a69a',
    decreasing_line_color='#ef5350',
))

# VWAP
if show_vwap:
    fig.add_trace(go.Scatter(
        x=df.index, y=df['vwap'],
        name='VWAP',
        line=dict(color='#ffeb3b', width=2),
    ))
    # SD bands
    fig.add_trace(go.Scatter(
        x=df.index, y=df['vwap_upper_1'],
        name='+1 SD', line=dict(color='rgba(255,235,59,0.3)', width=1, dash='dot'),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['vwap_lower_1'],
        name='-1 SD', line=dict(color='rgba(255,235,59,0.3)', width=1, dash='dot'),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['vwap_upper_2'],
        name='+2 SD', line=dict(color='rgba(255,152,0,0.5)', width=1, dash='dash'),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['vwap_lower_2'],
        name='-2 SD', line=dict(color='rgba(255,152,0,0.5)', width=1, dash='dash'),
        showlegend=False,
    ))

# Session levels (only for selected date range)
if show_sessions:
    visible_dates = sorted(set(df.index.date))
    for d in visible_dates:
        if d in session_levels:
            lv = session_levels[d]
            # Asia
            if 'Asia_high' in lv:
                fig.add_hline(y=lv['Asia_high'], line_dash='dot',
                              line_color='rgba(156,39,176,0.4)', opacity=0.5,
                              annotation_text=f"Asia H" if d == visible_dates[0] else "",
                              annotation_position="right")
                fig.add_hline(y=lv['Asia_low'], line_dash='dot',
                              line_color='rgba(156,39,176,0.4)', opacity=0.5)
            # London
            if 'London_high' in lv:
                fig.add_hline(y=lv['London_high'], line_dash='dot',
                              line_color='rgba(33,150,243,0.4)', opacity=0.5)
                fig.add_hline(y=lv['London_low'], line_dash='dot',
                              line_color='rgba(33,150,243,0.4)', opacity=0.5)
            # NY
            if 'NY_high' in lv:
                fig.add_hline(y=lv['NY_high'], line_dash='dot',
                              line_color='rgba(0,188,212,0.4)', opacity=0.5)
                fig.add_hline(y=lv['NY_low'], line_dash='dot',
                              line_color='rgba(0,188,212,0.4)', opacity=0.5)

# ORB
if show_orb and 'orb_high' in df.columns:
    orb_data = df[df['orb_high'].notna()]
    if len(orb_data) > 0:
        fig.add_trace(go.Scatter(
            x=orb_data.index, y=orb_data['orb_high'],
            name='ORB High',
            line=dict(color='#00e676', width=1.5, dash='dash'),
        ))
        fig.add_trace(go.Scatter(
            x=orb_data.index, y=orb_data['orb_low'],
            name='ORB Low',
            line=dict(color='#ff1744', width=1.5, dash='dash'),
        ))

fig.update_layout(
    title=f"{symbol} — {tf} — Last {len(df)} candles",
    template='plotly_dark',
    height=750,
    xaxis_rangeslider_visible=False,
    hovermode='x unified',
    dragmode='pan',
    legend=dict(orientation="h", y=1.02, x=0),
    margin=dict(l=40, r=40, t=80, b=40),
)

fig.update_xaxes(type='category', nticks=15)
fig.update_yaxes(title_text="Price")

st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

# ---------------- Legend Guide ----------------
with st.expander("📖 Indicator Guide"):
    st.markdown("""
    **VWAP (Yellow):** Session volume-weighted average. Institutional benchmark.
    - Price above VWAP = bullish bias
    - Price below VWAP = bearish bias
    - VWAP magnet — price returns here often

    **SD Bands (Orange dashed):**
    - +1/-1 SD = normal range
    - +2/-2 SD = extreme — fade opportunity

    **Session Levels:**
    - **Asia (Purple):** 19:00-02:00 EST — builds liquidity
    - **London (Blue):** 02:00-07:00 EST — sweeps Asia
    - **NY (Cyan):** 07:00-16:00 EST — real move

    **ORB (Green/Red dash):** First 15 min at 9:30 EST — breakout levels
    """)

st.caption(f"Data: {df.index.min()} → {df.index.max()}")
