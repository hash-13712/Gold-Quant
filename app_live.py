# ============================================================
# LIVE TRADING APP — Real-time data + indicators + alerts
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta
import time

from streamlit_autorefresh import st_autorefresh

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Live Gold Trader",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Auto-refresh every 30 seconds
st_autorefresh(interval=30_000, key="autorefresh")

# Session times (EST)
SESSIONS = {
    'Asia':    (19, 2),
    'London':  (2, 7),
    'NY':      (7, 16),
}

SYMBOL = "GC=F"  # Gold futures (yfinance)
TF_LIVE = "1m"    # 1-minute candles


# ---------------- LIVE DATA FETCHER ----------------
@st.cache_data(ttl=30)  # 30 sec cache
def fetch_live_data(period="1d", interval="1m"):
    """Yahoo Finance se latest data fetch karo."""
    try:
        df = yf.download(
            SYMBOL, period=period, interval=interval,
            progress=False, auto_adjust=False,
        )
        if df.empty:
            return None
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.rename(columns={
            'Open': 'open', 'High': 'high',
            'Low': 'low', 'Close': 'close', 'Volume': 'volume'
        })
        df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('US/Eastern')
        return df
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return None


# ---------------- INDICATORS ----------------
def add_vwap(df):
    """Session VWAP (daily reset)."""
    df = df.copy()
    df['date'] = df.index.date
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['vol'] = df['volume'].replace(0, 1)

    df['cum_vol'] = df.groupby('date')['vol'].cumsum()
    df['cum_tpv'] = (df['tp'] * df['vol']).groupby(df['date']).cumsum()
    df['vwap'] = df['cum_tpv'] / df['cum_vol']

    # SD bands
    df['dev_sq'] = (df['tp'] - df['vwap']) ** 2
    df['cum_dev_sq'] = (df['dev_sq'] * df['vol']).groupby(df['date']).cumsum()
    df['sd'] = np.sqrt(df['cum_dev_sq'] / df['cum_vol'])

    df['vwap_up_1'] = df['vwap'] + df['sd']
    df['vwap_dn_1'] = df['vwap'] - df['sd']
    df['vwap_up_2'] = df['vwap'] + 2 * df['sd']
    df['vwap_dn_2'] = df['vwap'] - 2 * df['sd']
    return df


def add_session_levels(df):
    """Session high/low levels."""
    levels = {}
    for date, group in df.groupby(df.index.date):
        day_l = {'date': date}
        for sess, (sh, eh) in SESSIONS.items():
            if sh < eh:
                s = group[(group.index.hour >= sh) & (group.index.hour < eh)]
            else:
                s = group[(group.index.hour >= sh) | (group.index.hour < eh)]
            if len(s) > 3:
                day_l[f'{sess}_high'] = s['high'].max()
                day_l[f'{sess}_low'] = s['low'].min()
        levels[date] = day_l
    return levels


def detect_sweeps(df, lookback=10):
    """Recent swing high/low sweep."""
    df = df.copy()
    df['swing_high'] = df['high'].rolling(lookback).max().shift(1)
    df['swing_low'] = df['low'].rolling(lookback).min().shift(1)

    # Bullish sweep: low breaks swing low, close above
    df['bull_sweep'] = (
        (df['low'] < df['swing_low']) &
        (df['close'] > df['swing_low'])
    )
    # Bearish sweep
    df['bear_sweep'] = (
        (df['high'] > df['swing_high']) &
        (df['close'] < df['swing_high'])
    )
    return df


# ---------------- ALERT SYSTEM ----------------
def check_alerts(df, levels):
    """Latest candle pe alert check."""
    if len(df) < 20:
        return None

    latest = df.iloc[-1]
    ts = df.index[-1]
    today = ts.date()
    today_levels = levels.get(today, {})

    # Latest sweep?
    if latest['bull_sweep']:
        # Check VWAP confirmation
        vwap_ok = latest['close'] > latest['vwap']
        return {
            'type': 'LONG',
            'time': ts,
            'price': latest['close'],
            'reason': 'Bullish sweep',
            'vwap_ok': vwap_ok,
            'stop': latest['low'] - 0.5,
            'target': latest['close'] + (latest['close'] - latest['low']) * 2,
        }
    elif latest['bear_sweep']:
        vwap_ok = latest['close'] < latest['vwap']
        return {
            'type': 'SHORT',
            'time': ts,
            'price': latest['close'],
            'reason': 'Bearish sweep',
            'vwap_ok': vwap_ok,
            'stop': latest['high'] + 0.5,
            'target': latest['close'] - (latest['high'] - latest['close']) * 2,
        }
    return None


def play_alert():
    """Beep sound (base64 embedded)."""
    # Simple beep base64 (short)
    st.toast("🚨 ALERT! New signal detected!", icon="🚨")


# ---------------- UI ----------------
st.title("🥇 Live Gold Trader")
st.caption(f"Auto-refresh: every 30s | Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    period = st.selectbox("Data period", ["1d", "5d", "1mo"], index=0)
    interval = st.selectbox("Timeframe", ["1m", "5m", "15m"], index=1)

    st.markdown("---")
    st.markdown("### 📊 Indicators")
    show_vwap = st.checkbox("VWAP + SD Bands", value=True)
    show_sessions = st.checkbox("Session Levels", value=True)
    show_sweeps = st.checkbox("Sweep Markers", value=True)

    st.markdown("---")
    st.markdown("### 🚨 Alerts")
    alerts_enabled = st.checkbox("Enable alerts", value=True)
    sound_enabled = st.checkbox("Sound alert", value=True)

    st.markdown("---")
    st.caption(f"Last update: {datetime.now().strftime('%H:%M:%S')}")

# Fetch data
with st.spinner("Fetching live data..."):
    df = fetch_live_data(period=period, interval=interval)

if df is None or len(df) < 20:
    st.error("Live data nahi mila. Check internet connection.")
    st.stop()

# Compute indicators
with st.spinner("Computing indicators..."):
    df = add_vwap(df)
    df = detect_sweeps(df, lookback=10)
    levels = add_session_levels(df)

# Latest data info
latest = df.iloc[-1]
ts = df.index[-1]
age_sec = (pd.Timestamp.now(tz='US/Eastern') - ts).total_seconds()

# Top metrics
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Price", f"{latest['close']:.2f}",
          delta=f"{latest['close'] - df.iloc[-2]['close']:+.2f}")
c2.metric("High", f"{latest['high']:.2f}")
c3.metric("Low", f"{latest['low']:.2f}")
c4.metric("VWAP", f"{latest['vwap']:.2f}")
c5.metric("Data age", f"{int(age_sec)}s ago")

# Alert check
if alerts_enabled:
    alert = check_alerts(df, levels)
    if alert:
        # Check if we already showed this alert
        alert_key = f"alert_{alert['time']}"
        if alert_key not in st.session_state:
            st.session_state[alert_key] = True
            color = "🟢" if alert['type'] == 'LONG' else "🔴"
            st.success(
                f"{color} **{alert['type']} SIGNAL** @ {alert['price']:.2f} "
                f"| {alert['time'].strftime('%H:%M')} EST "
                f"| {alert['reason']} | VWAP {'✓' if alert['vwap_ok'] else '✗'}"
            )
            if sound_enabled:
                st.toast(f"🚨 {alert['type']} alert!", icon="🚨")

# Chart (last 200 candles)
chart_df = df.tail(200)

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    vertical_spacing=0.03, row_heights=[0.75, 0.25],
)

# Candles
fig.add_trace(go.Candlestick(
    x=chart_df.index,
    open=chart_df['open'], high=chart_df['high'],
    low=chart_df['low'], close=chart_df['close'],
    name='XAUUSD',
    increasing_line_color='#26a69a',
    decreasing_line_color='#ef5350',
), row=1, col=1)

# VWAP
if show_vwap:
    fig.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df['vwap'],
        name='VWAP', line=dict(color='#ffeb3b', width=2),
    ), row=1, col=1)
    for band, color in [('vwap_up_1', 'rgba(255,235,59,0.3)'),
                         ('vwap_dn_1', 'rgba(255,235,59,0.3)'),
                         ('vwap_up_2', 'rgba(255,152,0,0.5)'),
                         ('vwap_dn_2', 'rgba(255,152,0,0.5)')]:
        fig.add_trace(go.Scatter(
            x=chart_df.index, y=chart_df[band],
            name=band, line=dict(color=color, width=1, dash='dot'),
            showlegend=False,
        ), row=1, col=1)

# Sweep markers
if show_sweeps:
    bulls = chart_df[chart_df['bull_sweep']]
    bears = chart_df[chart_df['bear_sweep']]
    if len(bulls):
        fig.add_trace(go.Scatter(
            x=bulls.index, y=bulls['low'] - 1,
            mode='markers',
            marker=dict(symbol='triangle-up', size=12, color='#00e676'),
            name='Bull Sweep', showlegend=True,
        ), row=1, col=1)
    if len(bears):
        fig.add_trace(go.Scatter(
            x=bears.index, y=bears['high'] + 1,
            mode='markers',
            marker=dict(symbol='triangle-down', size=12, color='#ff1744'),
            name='Bear Sweep', showlegend=True,
        ), row=1, col=1)

# Volume
fig.add_trace(go.Bar(
    x=chart_df.index, y=chart_df['volume'],
    name='Volume', marker_color='#4a90e2', opacity=0.5,
), row=2, col=1)

fig.update_layout(
    template='plotly_dark',
    height=700,
    xaxis_rangeslider_visible=False,
    hovermode='x unified',
    dragmode='pan',
    margin=dict(l=40, r=40, t=40, b=40),
)

st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

# ---------------- Live Signals Table ----------------
st.subheader("🎯 Recent Sweeps")

sweeps = df[(df['bull_sweep']) | (df['bear_sweep'])].tail(10)
if len(sweeps) > 0:
    display = sweeps[['close', 'vwap', 'bull_sweep', 'bear_sweep']].copy()
    display['direction'] = display.apply(
        lambda r: 'LONG' if r['bull_sweep'] else 'SHORT', axis=1
    )
    display['vwap_ok'] = display.apply(
        lambda r: '✓' if (r['bull_sweep'] and r['close'] > r['vwap'])
                       or (r['bear_sweep'] and r['close'] < r['vwap']) else '✗',
        axis=1
    )
    display = display[['close', 'vwap', 'direction', 'vwap_ok']]
    display.columns = ['Price', 'VWAP', 'Direction', 'VWAP Confirm']
    st.dataframe(display, use_container_width=True, height=300)
else:
    st.info("Koi recent sweep nahi mila.")

# ---------------- Footer ----------------
st.caption(
    f"Data: {df.index.min()} → {df.index.max()} | "
    f"Bars: {len(df)} | "
    f"Auto-refresh 30s | "
    f"⚠️ Yahoo Finance se data (delayed ~15min on free tier)"
)
