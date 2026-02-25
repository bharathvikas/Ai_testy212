# ============================================================

# streamlit_app.py — Mobile-Optimized AI Trading Dashboard

# Designed for iPhone Safari / Streamlit Cloud

# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings(“ignore”)

# ─────────────────────────────────────────────

# PAGE CONFIG — must be first Streamlit call

# ─────────────────────────────────────────────

st.set_page_config(
page_title=“AI Trader”,
page_icon=“📈”,
layout=“centered”,       # Better for mobile
initial_sidebar_state=“collapsed”
)

# ─────────────────────────────────────────────

# MOBILE-FIRST CSS

# ─────────────────────────────────────────────

st.markdown(”””

<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

:root {
    --bg: #080c10;
    --surface: #0e1420;
    --border: #1e2d40;
    --accent: #00d4ff;
    --green: #00ff88;
    --red: #ff4466;
    --gold: #ffd700;
    --text: #e8f4fd;
    --muted: #6b8299;
}

html, body, [data-testid="stApp"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif;
}

/* Bigger tap targets for mobile */
.stButton > button {
    background: linear-gradient(135deg, #00d4ff22, #00d4ff11) !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    padding: 14px 24px !important;
    border-radius: 8px !important;
    width: 100% !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00d4ff44, #00d4ff22) !important;
    box-shadow: 0 0 20px #00d4ff33 !important;
}

/* Selectbox & inputs */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 16px !important; /* Prevents iOS zoom on focus */
}

/* Metric cards */
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    margin-bottom: 12px;
}
.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 24px;
    font-weight: 700;
}

/* Signal badge */
.signal-buy {
    background: linear-gradient(135deg, #00ff8822, #00ff8811);
    border: 1px solid var(--green);
    color: var(--green);
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 28px;
    font-weight: 700;
    letter-spacing: 4px;
}
.signal-sell {
    background: linear-gradient(135deg, #ff446622, #ff446611);
    border: 1px solid var(--red);
    color: var(--red);
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 28px;
    font-weight: 700;
    letter-spacing: 4px;
}
.signal-hold {
    background: linear-gradient(135deg, #ffd70022, #ffd70011);
    border: 1px solid var(--gold);
    color: var(--gold);
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 28px;
    font-weight: 700;
    letter-spacing: 4px;
}

/* Header */
.hero {
    text-align: center;
    padding: 24px 0 16px;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 28px;
    font-weight: 800;
    color: var(--accent);
    margin: 0;
    letter-spacing: -0.5px;
}
.hero p {
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    margin-top: 4px;
}

/* Section headers */
.section-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 3px;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin: 24px 0 16px;
}

/* Trade row */
.trade-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 0;
    border-bottom: 1px solid var(--border);
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
}
.trade-win { color: var(--green); }
.trade-loss { color: var(--red); }

/* Hide Streamlit branding on mobile */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 16px 80px !important; max-width: 480px !important; margin: auto !important; }
</style>

“””, unsafe_allow_html=True)

# ─────────────────────────────────────────────

# LIGHTWEIGHT FEATURE ENGINE (no TF/sklearn)

# ─────────────────────────────────────────────

def compute_rsi(series, period=14):
delta = series.diff()
gain = delta.clip(lower=0).rolling(period).mean()
loss = (-delta.clip(upper=0)).rolling(period).mean()
rs = gain / (loss + 1e-9)
return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
ema_fast = series.ewm(span=fast, adjust=False).mean()
ema_slow = series.ewm(span=slow, adjust=False).mean()
macd = ema_fast - ema_slow
sig = macd.ewm(span=signal, adjust=False).mean()
return macd, sig, macd - sig

def compute_bb(series, period=20, std=2.0):
mid = series.rolling(period).mean()
sigma = series.rolling(period).std()
return mid + std * sigma, mid, mid - std * sigma

def compute_vwap(df):
cum_vol = df[“Volume”].groupby(df.index.date).cumsum()
cum_pv = (df[“Close”] * df[“Volume”]).groupby(df.index.date).cumsum()
return cum_pv / (cum_vol + 1e-9)

def compute_atr(df, period=14):
hl = df[“High”] - df[“Low”]
hpc = (df[“High”] - df[“Close”].shift(1)).abs()
lpc = (df[“Low”] - df[“Close”].shift(1)).abs()
tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
return tr.ewm(span=period, adjust=False).mean()

# ─────────────────────────────────────────────

# VADER SENTIMENT (pure Python, no downloads)

# ─────────────────────────────────────────────

def vader_sentiment(ticker):
“””
Lightweight VADER-based sentiment on yfinance news headlines.
VADER is rule-based — no model download needed.
“””
try:
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
stock = yf.Ticker(ticker)
news = stock.news or []
if not news:
return 0.0, 0
scores = []
for item in news[:10]:
title = item.get(“title”, “”)
if title:
score = analyzer.polarity_scores(title)[“compound”]
scores.append(score)
return round(float(np.mean(scores)), 3), len(scores)
except Exception:
return 0.0, 0

# ─────────────────────────────────────────────

# RULE-BASED SIGNAL ENGINE

# Replaces LSTM with weighted scoring system

# Works entirely without TensorFlow

# ─────────────────────────────────────────────

def generate_signal_mobile(df, sentiment_score=0.0):
“””
Weighted multi-factor signal generator.
Returns: signal, score, breakdown dict
“””
latest = df.iloc[-1]
score = 0
breakdown = {}

```
# RSI
rsi = latest.get("RSI", 50)
if rsi < 35:
    s = 2; label = f"Oversold ({rsi:.0f})"
elif rsi < 45:
    s = 1; label = f"Mild oversold ({rsi:.0f})"
elif rsi > 65:
    s = -2; label = f"Overbought ({rsi:.0f})"
elif rsi > 55:
    s = -1; label = f"Mild overbought ({rsi:.0f})"
else:
    s = 0; label = f"Neutral ({rsi:.0f})"
score += s; breakdown["RSI"] = (s, label)

# MACD
macd_hist = latest.get("MACD_hist", 0)
macd_cross_up = latest.get("MACD_cross_up", 0)
macd_cross_dn = latest.get("MACD_cross_dn", 0)
if macd_cross_up:
    s = 2; label = "Bullish crossover ✓"
elif macd_hist > 0:
    s = 1; label = f"Positive hist ({macd_hist:.3f})"
elif macd_cross_dn:
    s = -2; label = "Bearish crossover ✗"
else:
    s = -1; label = f"Negative hist ({macd_hist:.3f})"
score += s; breakdown["MACD"] = (s, label)

# Bollinger Bands
bb_pct = latest.get("BB_pct_b", 0.5)
if bb_pct < 0.1:
    s = 2; label = f"Near lower band ({bb_pct:.2f})"
elif bb_pct < 0.3:
    s = 1; label = f"Lower zone ({bb_pct:.2f})"
elif bb_pct > 0.9:
    s = -2; label = f"Near upper band ({bb_pct:.2f})"
elif bb_pct > 0.7:
    s = -1; label = f"Upper zone ({bb_pct:.2f})"
else:
    s = 0; label = f"Mid band ({bb_pct:.2f})"
score += s; breakdown["Bollinger"] = (s, label)

# EMA trend
ema_bull = latest.get("EMA_bull", 0)
ema_bear = latest.get("EMA_bear", 0)
if ema_bull:
    s = 2; label = "Bullish alignment (9>21>50)"
elif ema_bear:
    s = -2; label = "Bearish alignment (9<21<50)"
else:
    s = 0; label = "Mixed / choppy"
score += s; breakdown["EMA Trend"] = (s, label)

# VWAP
price_vs_vwap = latest.get("Price_vs_VWAP", 0)
if price_vs_vwap > 0.005:
    s = 1; label = f"Above VWAP (+{price_vs_vwap*100:.2f}%)"
elif price_vs_vwap < -0.005:
    s = -1; label = f"Below VWAP ({price_vs_vwap*100:.2f}%)"
else:
    s = 0; label = "At VWAP"
score += s; breakdown["VWAP"] = (s, label)

# Volume
vol_ratio = latest.get("Vol_ratio", 1.0)
if vol_ratio > 2.0:
    s = 1 if score > 0 else -1; label = f"Spike {vol_ratio:.1f}x (confirms trend)"
elif vol_ratio > 1.3:
    s = 0; label = f"Above avg ({vol_ratio:.1f}x)"
else:
    s = 0; label = f"Low volume ({vol_ratio:.1f}x)"
score += s; breakdown["Volume"] = (s, label)

# Sentiment
if sentiment_score > 0.2:
    s = 1; label = f"Positive ({sentiment_score:+.2f})"
elif sentiment_score < -0.2:
    s = -1; label = f"Negative ({sentiment_score:+.2f})"
else:
    s = 0; label = f"Neutral ({sentiment_score:+.2f})"
score += s; breakdown["Sentiment"] = (s, label)

# Candlestick bonus
if latest.get("Bull_Engulf", 0):
    score += 1; breakdown["Candle"] = (1, "Bullish Engulfing")
elif latest.get("Bear_Engulf", 0):
    score -= 1; breakdown["Candle"] = (-1, "Bearish Engulfing")
elif latest.get("Hammer", 0):
    score += 1; breakdown["Candle"] = (1, "Hammer")
elif latest.get("Shooting_Star", 0):
    score -= 1; breakdown["Candle"] = (-1, "Shooting Star")
else:
    breakdown["Candle"] = (0, "No pattern")

# Convert score to signal
max_score = 13
norm = score / max_score  # -1 to +1

if norm > 0.25:
    signal = "BUY"
elif norm < -0.25:
    signal = "SELL"
else:
    signal = "HOLD"

confidence = min(abs(norm) * 100, 100)
return signal, round(confidence, 1), score, breakdown
```

# ─────────────────────────────────────────────

# DATA + FEATURE PIPELINE

# ─────────────────────────────────────────────

@st.cache_data(ttl=300)  # Cache 5 minutes
def load_and_process(ticker, interval):
df = yf.download(ticker, period=“5d”, interval=interval,
auto_adjust=True, progress=False)
if isinstance(df.columns, pd.MultiIndex):
df.columns = df.columns.get_level_values(0)
df = df[[“Open”, “High”, “Low”, “Close”, “Volume”]].dropna()
df = df.between_time(“09:30”, “16:00”)

```
if len(df) < 30:
    return None

# Technical indicators
df["RSI"] = compute_rsi(df["Close"])
df["MACD"], df["MACD_sig"], df["MACD_hist"] = compute_macd(df["Close"])
df["MACD_cross_up"] = ((df["MACD"] > df["MACD_sig"]) &
                        (df["MACD"].shift(1) <= df["MACD_sig"].shift(1))).astype(int)
df["MACD_cross_dn"] = ((df["MACD"] < df["MACD_sig"]) &
                        (df["MACD"].shift(1) >= df["MACD_sig"].shift(1))).astype(int)
df["BB_upper"], df["BB_mid"], df["BB_lower"] = compute_bb(df["Close"])
df["BB_pct_b"] = (df["Close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"] + 1e-9)
df["ATR"] = compute_atr(df)
df["VWAP"] = compute_vwap(df)
df["Price_vs_VWAP"] = (df["Close"] - df["VWAP"]) / (df["VWAP"] + 1e-9)

for s in [9, 21, 50]:
    df[f"EMA_{s}"] = df["Close"].ewm(span=s, adjust=False).mean()
df["EMA_bull"] = ((df["EMA_9"] > df["EMA_21"]) & (df["EMA_21"] > df["EMA_50"])).astype(int)
df["EMA_bear"] = ((df["EMA_9"] < df["EMA_21"]) & (df["EMA_21"] < df["EMA_50"])).astype(int)

df["Vol_ratio"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-9)

# Candlestick patterns
o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
body = (c - o).abs()
uw = h - pd.concat([c, o], axis=1).max(axis=1)
lw = pd.concat([c, o], axis=1).min(axis=1) - l
df["Bull_Engulf"] = ((c > o) & (c.shift(1) < o.shift(1)) &
                      (c > o.shift(1)) & (o < c.shift(1))).astype(int)
df["Bear_Engulf"] = ((c < o) & (c.shift(1) > o.shift(1)) &
                      (c < o.shift(1)) & (o > c.shift(1))).astype(int)
df["Hammer"] = ((lw >= 2 * body) & (uw <= 0.3 * body) & (c > o)).astype(int)
df["Shooting_Star"] = ((uw >= 2 * body) & (lw <= 0.3 * body) & (c < o)).astype(int)

df.dropna(inplace=True)
return df
```

def mini_backtest(df, sentiment=0.0):
“”“Quick 30-trade backtest on recent data.”””
trades = []
capital = 10000.0
equity = [capital]
position = None

```
for i in range(50, len(df)):
    window = df.iloc[:i]
    price = float(df["Close"].iloc[i])
    sig, conf, _, _ = generate_signal_mobile(window, sentiment)
    atr = float(df["ATR"].iloc[i])

    if position is None and sig == "BUY" and conf > 40:
        shares = int((capital * 0.95) / price)
        sl = price * 0.995
        tp = price * 1.015
        position = {"shares": shares, "entry": price, "sl": sl, "tp": tp, "type": "long"}

    elif position and position["type"] == "long":
        if price <= position["sl"]:
            pnl = (price - position["entry"]) * position["shares"]
            capital += pnl
            trades.append({"pnl": pnl, "reason": "SL", "entry": position["entry"], "exit": price})
            position = None
        elif price >= position["tp"]:
            pnl = (price - position["entry"]) * position["shares"]
            capital += pnl
            trades.append({"pnl": pnl, "reason": "TP", "entry": position["entry"], "exit": price})
            position = None

    equity.append(capital)

# Close any open position
if position:
    last_price = float(df["Close"].iloc[-1])
    pnl = (last_price - position["entry"]) * position["shares"]
    capital += pnl
    trades.append({"pnl": pnl, "reason": "EOD", "entry": position["entry"], "exit": last_price})

return trades, equity
```

# ─────────────────────────────────────────────

# UI RENDERING

# ─────────────────────────────────────────────

# Hero

st.markdown(”””

<div class="hero">
    <h1>⚡ AI TRADER</h1>
    <p>INTRADAY · LSTM-INSPIRED · MOBILE</p>
</div>
""", unsafe_allow_html=True)

# Controls

col1, col2 = st.columns(2)
with col1:
ticker = st.selectbox(“Ticker”, [“AAPL”, “MSFT”, “NVDA”, “TSLA”, “AMZN”,
“GOOG”, “META”, “SPY”, “QQQ”], index=0)
with col2:
interval = st.selectbox(“Interval”, [“5m”, “15m”, “1h”], index=0)

run = st.button(“🔍 ANALYSE NOW”)

if run or “df” in st.session_state:
if run:
with st.spinner(“Fetching data & computing signals…”):
df = load_and_process(ticker, interval)
sentiment_score, n_articles = vader_sentiment(ticker)
st.session_state[“df”] = df
st.session_state[“sentiment”] = sentiment_score
st.session_state[“n_articles”] = n_articles
st.session_state[“ticker”] = ticker
else:
df = st.session_state.get(“df”)
sentiment_score = st.session_state.get(“sentiment”, 0.0)
n_articles = st.session_state.get(“n_articles”, 0)

```
if df is None or len(df) < 30:
    st.error("Not enough intraday data. Try a different ticker or interval.")
    st.stop()

signal, confidence, raw_score, breakdown = generate_signal_mobile(df, sentiment_score)
latest = df.iloc[-1]
price = float(latest["Close"])
atr = float(latest["ATR"])
prev_close = float(df["Close"].iloc[-2])
change = (price - prev_close) / prev_close * 100

# ── SIGNAL BADGE
st.markdown('<div class="section-title">Current Signal</div>', unsafe_allow_html=True)
badge_class = f"signal-{'buy' if signal=='BUY' else 'sell' if signal=='SELL' else 'hold'}"
icon = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "🟡"
st.markdown(f'<div class="{badge_class}">{icon} {signal}</div>', unsafe_allow_html=True)

# ── PRICE METRICS
st.markdown('<div class="section-title">Market Snapshot</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    color = "#00ff88" if change >= 0 else "#ff4466"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Price</div>
        <div class="metric-value" style="color:{color}">${price:.2f}</div>
    </div>""", unsafe_allow_html=True)
with c2:
    color = "#00ff88" if change >= 0 else "#ff4466"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Change</div>
        <div class="metric-value" style="color:{color}">{change:+.2f}%</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Confidence</div>
        <div class="metric-value" style="color:#00d4ff">{confidence:.0f}%</div>
    </div>""", unsafe_allow_html=True)

c4, c5, c6 = st.columns(3)
rsi_val = float(latest.get("RSI", 50))
with c4:
    rsi_color = "#ff4466" if rsi_val > 70 else "#00ff88" if rsi_val < 30 else "#e8f4fd"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">RSI</div>
        <div class="metric-value" style="color:{rsi_color}">{rsi_val:.1f}</div>
    </div>""", unsafe_allow_html=True)
with c5:
    sent_color = "#00ff88" if sentiment_score > 0.1 else "#ff4466" if sentiment_score < -0.1 else "#ffd700"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Sentiment</div>
        <div class="metric-value" style="color:{sent_color}">{sentiment_score:+.2f}</div>
    </div>""", unsafe_allow_html=True)
with c6:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">ATR</div>
        <div class="metric-value" style="color:#e8f4fd">${atr:.2f}</div>
    </div>""", unsafe_allow_html=True)

# ── RISK LEVELS (if BUY/SELL)
if signal in ("BUY", "SELL"):
    st.markdown('<div class="section-title">Risk Levels</div>', unsafe_allow_html=True)
    if signal == "BUY":
        sl = price * 0.995
        tp = price * 1.015
    else:
        sl = price * 1.005
        tp = price * 0.985
    r1, r2 = st.columns(2)
    with r1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Stop Loss</div>
            <div class="metric-value" style="color:#ff4466">${sl:.2f}</div>
        </div>""", unsafe_allow_html=True)
    with r2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Take Profit</div>
            <div class="metric-value" style="color:#00ff88">${tp:.2f}</div>
        </div>""", unsafe_allow_html=True)

# ── SIGNAL BREAKDOWN
st.markdown('<div class="section-title">Signal Breakdown</div>', unsafe_allow_html=True)
for factor, (s, label) in breakdown.items():
    color = "#00ff88" if s > 0 else "#ff4466" if s < 0 else "#6b8299"
    arrow = "▲" if s > 0 else "▼" if s < 0 else "●"
    st.markdown(f"""
    <div class="trade-row">
        <span style="color:#e8f4fd;font-family:'JetBrains Mono',monospace">{factor}</span>
        <span style="color:{color};font-family:'JetBrains Mono',monospace;font-size:11px">{arrow} {label}</span>
    </div>""", unsafe_allow_html=True)

# ── PRICE CHART
st.markdown('<div class="section-title">Price Chart</div>', unsafe_allow_html=True)
chart_df = df[["Close", "BB_upper", "BB_lower", "EMA_9", "EMA_21"]].tail(60).copy()
chart_df.columns = ["Close", "BB Upper", "BB Lower", "EMA 9", "EMA 21"]
st.line_chart(chart_df, use_container_width=True, height=220)

# ── QUICK BACKTEST
st.markdown('<div class="section-title">Quick Backtest (5-day)</div>', unsafe_allow_html=True)
with st.spinner("Running backtest..."):
    trades, equity_curve = mini_backtest(df, sentiment_score)

if trades:
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    total_pnl = sum(t["pnl"] for t in trades)
    win_rate = len(wins) / len(trades) * 100

    b1, b2, b3 = st.columns(3)
    with b1:
        pnl_color = "#00ff88" if total_pnl >= 0 else "#ff4466"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Total P&L</div>
            <div class="metric-value" style="color:{pnl_color}">${total_pnl:+.0f}</div>
        </div>""", unsafe_allow_html=True)
    with b2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Win Rate</div>
            <div class="metric-value" style="color:#00d4ff">{win_rate:.0f}%</div>
        </div>""", unsafe_allow_html=True)
    with b3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Trades</div>
            <div class="metric-value" style="color:#e8f4fd">{len(trades)}</div>
        </div>""", unsafe_allow_html=True)

    # Equity curve chart
    eq_series = pd.Series(equity_curve, name="Equity ($10k)")
    st.line_chart(eq_series, use_container_width=True, height=180)

    # Trade log
    st.markdown('<div class="section-title">Trade Log</div>', unsafe_allow_html=True)
    for i, t in enumerate(trades[-8:]):
        color = "#00ff88" if t["pnl"] > 0 else "#ff4466"
        icon = "✅" if t["pnl"] > 0 else "❌"
        st.markdown(f"""
        <div class="trade-row">
            <span style="color:#6b8299;font-family:'JetBrains Mono',monospace">#{i+1} {t['reason']}</span>
            <span style="color:{color};font-family:'JetBrains Mono',monospace">{icon} ${t['pnl']:+.2f}</span>
        </div>""", unsafe_allow_html=True)
else:
    st.info("No trades generated in backtest window — try a different interval.")

# ── NEWS SENTIMENT NOTE
st.markdown(f"""
<div style="margin-top:24px;padding:12px;background:#0e1420;border:1px solid #1e2d40;
border-radius:8px;font-family:'JetBrains Mono',monospace;font-size:11px;color:#6b8299;text-align:center">
    📰 Sentiment from {n_articles} headlines · Updated {datetime.now().strftime('%H:%M')}
</div>
""", unsafe_allow_html=True)
```

else:
st.markdown(”””
<div style="text-align:center;padding:48px 24px;color:#6b8299;
font-family:'JetBrains Mono',monospace;font-size:13px;line-height:2">
Select a ticker and interval<br>then tap <b style="color:#00d4ff">ANALYSE NOW</b><br><br>
Signals powered by:<br>
RSI · MACD · Bollinger · EMA<br>
VWAP · Volume · Sentiment · Candlesticks
</div>
“””, unsafe_allow_html=True)
