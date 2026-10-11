# ============================================================
# TRADE VERIFICATION — Backtest Trades Ko Chart Pe Draw Karo
# Har trade ko visual proof ke saath dekho
# ============================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

DATA_DIR = Path.home() / "gold_quant" / "data"
CANDLE_DIR = DATA_DIR / "candles"
RESULTS_DIR = Path.home() / "gold_quant" / "results"

st.set_page_config(page_title="Verify Trades", layout="wide")
st.title("🔍 Trade Verification — Chart Pe Trades Dekho")


@st.cache_data(ttl=600)
def load_candles():
    f = CANDLE_DIR / "XAUUSD_5m.parquet"
    df = pd.read_parquet(f)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert('US/Eastern')
    return df


@st.cache_data(ttl=600)
def load_trades():
    f = RESULTS_DIR / "sweep_v8_trades.csv"
    if not f.exists():
        return None
    t = pd.read_csv(f)
    t['date'] = pd.to_datetime(t['date']).dt.date
    t['entry_time'] = pd.to_datetime(t['entry_time'], utc=True)
    t['exit_time'] = pd.to_datetime(t['exit_time'], utc=True)
    t['entry_time'] = t['entry_time'].dt.tz_convert('US/Eastern')
    t['exit_time'] = t['exit_time'].dt.tz_convert('US/Eastern')
    return t


candles = load_candles()
trades = load_trades()

if trades is None or len(trades) == 0:
    st.error("Trades CSV nahi mili. Pehle sweep_v8.py chalao (jo updated version hai).")
    st.stop()

# --------- Sidebar ---------
with st.sidebar:
    st.header("⚙️ Options")
    all_dates = sorted(trades['date'].unique())
    selected_date = st.selectbox(
        "Trade date select karo",
        all_dates,
        index=len(all_dates) - 1,
    )

    st.metric("Total trades", len(trades))
    st.metric("Wins", (trades['pnl'] > 0).sum())
    st.metric("Losses", (trades['pnl'] <= 0).sum())

# Filter trades for selected date
day_trades = trades[trades['date'] == selected_date]

# Chart window: prev day + current day
date_start = pd.Timestamp(selected_date) - pd.Timedelta(days=1)
date_end = pd.Timestamp(selected_date) + pd.Timedelta(days=1)

mask = (candles.index >= date_start.tz_localize('US/Eastern')) & \
       (candles.index <= date_end.tz_localize('US/Eastern'))
chart = candles[mask]

if len(chart) == 0:
    st.warning("Is date ka candle data nahi mila")
    st.stop()

# --------- Build Chart ---------
fig = go.Figure()

# Candles
fig.add_trace(go.Candlestick(
    x=chart.index,
    open=chart['open'], high=chart['high'],
    low=chart['low'], close=chart['close'],
    name='XAUUSD',
    increasing_line_color='#26a69a',
    decreasing_line_color='#ef5350',
))

# Levels from prev day
if len(day_trades) > 0:
    t = day_trades.iloc[0]

    # NY High / Low
    for name, col, color in [
        ('NY High', 'NY_high', '#00bcd4'),
        ('NY Low', 'NY_low', '#00bcd4'),
        ('Late High', 'Late_high', '#ff9800'),
        ('Late Low', 'Late_low', '#ff9800'),
    ]:
        if col in t.index and pd.notna(t[col]):
            fig.add_hline(
                y=t[col],
                line_dash='dash',
                line_color=color,
                annotation_text=f"{name}: {t[col]:.2f}",
                annotation_position="right",
                opacity=0.6,
            )

    # Sweep level (red dot)
    if 'level_price' in t.index and pd.notna(t['level_price']):
        fig.add_hline(
            y=t['level_price'],
            line_dash='dot',
            line_color='#ffeb3b',
            annotation_text=f"Swept {t['level']}: {t['level_price']:.2f}",
            annotation_position="left",
        )

    # Entry marker
    color = '#00e676' if t['direction'] == 'LONG' else '#ff1744'
    symbol = 'triangle-up' if t['direction'] == 'LONG' else 'triangle-down'
    fig.add_trace(go.Scatter(
        x=[t['entry_time']],
        y=[t['entry']],
        mode='markers+text',
        marker=dict(size=20, color=color, symbol=symbol,
                    line=dict(width=2, color='white')),
        text=[f"ENTRY {t['direction']}<br>{t['entry']:.2f}"],
        textposition='top center' if t['direction'] == 'LONG' else 'bottom center',
        name='Entry',
        showlegend=False,
    ))

    # SL line (red)
    fig.add_hline(
        y=t['stop'],
        line_dash='solid',
        line_color='#ef5350',
        annotation_text=f"SL: {t['stop']:.2f}",
        annotation_position="left",
        opacity=0.7,
    )

    # Target line (green)
    fig.add_hline(
        y=t['target'],
        line_dash='solid',
        line_color='#26a69a',
        annotation_text=f"TP ({t['target_level']}): {t['target']:.2f}",
        annotation_position="left",
        opacity=0.7,
    )

    # Exit marker
    exit_color = '#26a69a' if t['reason'] == 'TARGET' else '#ef5350'
    fig.add_trace(go.Scatter(
        x=[t['exit_time']],
        y=[t['exit']],
        mode='markers+text',
        marker=dict(size=16, color=exit_color, symbol='x',
                    line=dict(width=2, color='white')),
        text=[f"EXIT: {t['reason']}<br>{t['exit']:.2f} | P&L: {t['pnl']:+.2f}"],
        textposition='top center',
        name='Exit',
        showlegend=False,
    ))

fig.update_layout(
    title=f"Trade Verification — {selected_date}",
    template='plotly_dark',
    height=750,
    xaxis_rangeslider_visible=False,
    hovermode='x unified',
    dragmode='pan',
)

st.plotly_chart(fig, use_container_width=True)

# --------- Trade Details ---------
st.subheader("📋 Trade Details")
if len(day_trades) > 0:
    display = day_trades[[
        'direction', 'level', 'level_price',
        'entry', 'stop', 'target', 'target_level',
        'exit', 'pnl', 'reason'
    ]].copy()
    st.dataframe(display, use_container_width=True)

    # Logic Explanation
    st.subheader("🧠 Logic Explanation")
    t = day_trades.iloc[0]
    if t['direction'] == 'LONG':
        st.markdown(f"""
        **LONG Trade:**
        1. Prev day NY Low = `{t['NY_low']:.2f}`
        2. Aaj 01:00-04:00 EST mein price ne NY Low ko **sweep** kiya (wick neeche, close upar)
        3. Swept level: `{t['level_price']:.2f}`
        4. Entry: `{t['entry']:.2f}` (sweep candle ka close)
        5. SL: `{t['stop']:.2f}` (-$4)
        6. Target: `{t['target']:.2f}` = Late High (`{t['Late_high']:.2f}`)
        7. Exit: `{t['exit']:.2f}` ({t['reason']})
        8. P&L: `{t['pnl']:+.2f}`
        """)
    else:
        st.markdown(f"""
        **SHORT Trade:**
        1. Prev day NY High = `{t['NY_high']:.2f}`
        2. Aaj 01:00-04:00 EST mein price ne NY High ko **sweep** kiya
        3. Swept level: `{t['level_price']:.2f}`
        4. Entry: `{t['entry']:.2f}`
        5. SL: `{t['stop']:.2f}` (+$4)
        6. Target: `{t['target']:.2f}` = Late Low (`{t['Late_low']:.2f}`)
        7. Exit: `{t['exit']:.2f}` ({t['reason']})
        8. P&L: `{t['pnl']:+.2f}`
        """)
