import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands
from datetime import datetime

st.set_page_config(page_title="Market Summary", layout="wide")
st.title("📈 Market Summary")

# =========================
# Session checks
# =========================
if "ticker_symbol" not in st.session_state:
    st.warning("Please select an asset from the main page.")
    st.stop()

ticker = st.session_state["ticker_symbol"]
company_name = st.session_state.get("selected_company", ticker)

st.subheader(f"{company_name} ({ticker})")

# =========================
# Sidebar controls
# =========================
st.sidebar.header("Market Summary Settings")

period = st.sidebar.selectbox(
    "Select Period",
    ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
    index=3
)

interval = st.sidebar.selectbox(
    "Select Interval",
    ["1d", "1wk", "1mo"],
    index=0
)

# =========================
# Helper functions
# =========================
@st.cache_data(ttl=900)
def load_market_data(ticker_symbol, period_value, interval_value):
    """
    Historical OHLCV data for charts + indicators
    """
    df = yf.download(
        ticker_symbol,
        period=period_value,
        interval=interval_value,
        auto_adjust=False,
        progress=False,
        threads=False
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # Flatten columns if MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep only needed columns if present
    cols_to_keep = [col for col in ["Open", "High", "Low", "Close", "Volume"] if col in df.columns]
    df = df[cols_to_keep].copy()

    # Forward fill and drop remaining NaN
    df.ffill(inplace=True)
    df.dropna(inplace=True)

    return df


@st.cache_data(ttl=300)
def load_live_snapshot(ticker_symbol):
    """
    Fetch live/latest price snapshot separately from historical data.
    """
    try:
        tk = yf.Ticker(ticker_symbol)

        current_price = None
        prev_close = None
        day_high = None
        day_low = None
        currency = "₹"

        # Try fast_info first
        try:
            fi = tk.fast_info
            if fi:
                current_price = fi.get("lastPrice", None)
                prev_close = fi.get("previousClose", None)
                day_high = fi.get("dayHigh", None)
                day_low = fi.get("dayLow", None)
        except:
            pass

        # Fallback to info if needed
        if current_price is None or prev_close is None:
            try:
                info = tk.info
                current_price = current_price if current_price is not None else info.get("currentPrice", None)
                prev_close = prev_close if prev_close is not None else info.get("previousClose", None)
                day_high = day_high if day_high is not None else info.get("dayHigh", None)
                day_low = day_low if day_low is not None else info.get("dayLow", None)

                if "currency" in info and info["currency"]:
                    if info["currency"] == "INR":
                        currency = "₹"
                    else:
                        currency = info["currency"]
            except:
                pass

        return {
            "current_price": current_price,
            "prev_close": prev_close,
            "day_high": day_high,
            "day_low": day_low,
            "currency": currency
        }

    except Exception:
        return {
            "current_price": None,
            "prev_close": None,
            "day_high": None,
            "day_low": None,
            "currency": "₹"
        }


# =========================
# Load data
# =========================
df = load_market_data(ticker, period, interval)
snapshot = load_live_snapshot(ticker)

if df.empty:
    st.error("No market data found for this asset from Yahoo Finance.")
    st.stop()

# Required columns check
needed_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
for col in needed_cols:
    if col not in df.columns:
        st.error(f"Missing required column: {col}")
        st.stop()

# =========================
# Historical latest date info
# =========================
latest_hist_date = pd.to_datetime(df.index[-1]).date()
today_date = datetime.now().date()

if latest_hist_date < today_date:
    st.info(
        f"Latest historical candle available from Yahoo Finance is only up to **{latest_hist_date}**. "
        f"Top metrics may still show the latest/live price separately if available."
    )

# =========================
# Technical Indicators
# =========================
# Use a copy so we don't mutate the original unexpectedly
ind_df = df.copy()

# RSI
rsi = RSIIndicator(close=ind_df['Close'], window=14)
ind_df['RSI'] = rsi.rsi()

# MACD
macd_indicator = MACD(close=ind_df['Close'])
ind_df['MACD'] = macd_indicator.macd()
ind_df['MACD_Signal'] = macd_indicator.macd_signal()
ind_df['MACD_Hist'] = macd_indicator.macd_diff()

# SMA / EMA
sma20 = SMAIndicator(close=ind_df['Close'], window=20)
ema20 = EMAIndicator(close=ind_df['Close'], window=20)
ind_df['SMA20'] = sma20.sma_indicator()
ind_df['EMA20'] = ema20.ema_indicator()

# Bollinger Bands
bb = BollingerBands(close=ind_df['Close'], window=20, window_dev=2)
ind_df['BB_High'] = bb.bollinger_hband()
ind_df['BB_Low'] = bb.bollinger_lband()
ind_df['BB_Mid'] = bb.bollinger_mavg()

# Drop NaNs created by indicators
ind_df.dropna(inplace=True)

if ind_df.empty:
    st.warning("Not enough data to calculate technical indicators for the selected period/interval.")
    st.stop()

# =========================
# Latest values
# =========================
hist_latest_close = float(ind_df['Close'].iloc[-1])
hist_prev_close = float(ind_df['Close'].iloc[-2]) if len(ind_df) > 1 else hist_latest_close

live_price = snapshot["current_price"]
live_prev_close = snapshot["prev_close"]
day_high = snapshot["day_high"]
day_low = snapshot["day_low"]
currency = snapshot["currency"] if snapshot["currency"] else "₹"

# If live price available use that for top metric; otherwise use historical close
display_price = float(live_price) if live_price is not None else hist_latest_close
reference_prev_close = float(live_prev_close) if live_prev_close is not None else hist_prev_close

change = display_price - reference_prev_close
pct_change = (change / reference_prev_close * 100) if reference_prev_close not in [0, None] else 0

# =========================
# Top Summary Metrics
# =========================
st.subheader("📊 Quick Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Latest / Live Price",
    f"{currency}{display_price:,.2f}",
    f"{change:,.2f} ({pct_change:.2f}%)"
)

col2.metric(
    "Historical Close",
    f"{currency}{hist_latest_close:,.2f}"
)

col3.metric(
    "RSI (14)",
    f"{ind_df['RSI'].iloc[-1]:.2f}"
)

latest_volume = int(ind_df['Volume'].iloc[-1]) if not pd.isna(ind_df['Volume'].iloc[-1]) else 0
col4.metric(
    "Volume",
    f"{latest_volume:,}"
)

# Optional extra row
col5, col6, col7 = st.columns(3)
col5.metric("Day High", f"{currency}{float(day_high):,.2f}" if day_high is not None else "N/A")
col6.metric("Day Low", f"{currency}{float(day_low):,.2f}" if day_low is not None else "N/A")
col7.metric("Latest Historical Date", str(latest_hist_date))

# =========================
# Price Chart with SMA/EMA/Bollinger
# =========================
st.subheader("📉 Price Chart with Indicators")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=ind_df.index,
    y=ind_df['Close'],
    mode='lines',
    name='Close',
    line=dict(width=2)
))

fig.add_trace(go.Scatter(
    x=ind_df.index,
    y=ind_df['SMA20'],
    mode='lines',
    name='SMA 20',
    line=dict(width=2)
))

fig.add_trace(go.Scatter(
    x=ind_df.index,
    y=ind_df['EMA20'],
    mode='lines',
    name='EMA 20',
    line=dict(width=2)
))

fig.add_trace(go.Scatter(
    x=ind_df.index,
    y=ind_df['BB_High'],
    mode='lines',
    name='BB Upper',
    line=dict(dash='dot', width=2)
))

fig.add_trace(go.Scatter(
    x=ind_df.index,
    y=ind_df['BB_Low'],
    mode='lines',
    name='BB Lower',
    line=dict(dash='dot', width=2)
))

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Price",
    template="plotly_dark",
    height=550,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# RSI Chart
# =========================
st.subheader("📍 RSI (14)")

fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(
    x=ind_df.index,
    y=ind_df['RSI'],
    mode='lines',
    name='RSI',
    line=dict(width=2)
))

fig_rsi.add_hline(y=70, line_dash="dash", annotation_text="Overbought (70)")
fig_rsi.add_hline(y=30, line_dash="dash", annotation_text="Oversold (30)")

fig_rsi.update_layout(
    xaxis_title="Date",
    yaxis_title="RSI",
    template="plotly_dark",
    height=320
)

st.plotly_chart(fig_rsi, use_container_width=True)

# =========================
# MACD Chart
# =========================
st.subheader("📌 MACD")

fig_macd = go.Figure()

fig_macd.add_trace(go.Scatter(
    x=ind_df.index,
    y=ind_df['MACD'],
    mode='lines',
    name='MACD',
    line=dict(width=2)
))

fig_macd.add_trace(go.Scatter(
    x=ind_df.index,
    y=ind_df['MACD_Signal'],
    mode='lines',
    name='Signal',
    line=dict(width=2)
))

fig_macd.add_trace(go.Bar(
    x=ind_df.index,
    y=ind_df['MACD_Hist'],
    name='Histogram'
))

fig_macd.update_layout(
    xaxis_title="Date",
    yaxis_title="MACD",
    template="plotly_dark",
    height=360
)

st.plotly_chart(fig_macd, use_container_width=True)

# =========================
# Raw Data
# =========================
with st.expander("📄 Show Processed Data"):
    st.dataframe(ind_df.tail(100), use_container_width=True)