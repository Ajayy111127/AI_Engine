import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import RSIIndicator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam

st.set_page_config(page_title="30-Day AI Engine", layout="wide")

currency = "₹"
ticker_symbol = st.session_state.get('ticker_symbol', 'RELIANCE.NS')
selected_company = st.session_state.get('selected_company', 'Reliance Industries')

st.title("🤖 Deep Learning: 30-Day Forecasting")
st.write(f"Training Sequence-to-Sequence LSTM to project the next month of price action for **{selected_company}**.")

@st.cache_data(ttl=3600)
def fetch_ai_data(ticker):
    end_date = date.today()
    start_date = end_date - timedelta(days=730) # Need 2 years for 30 day predictions
    df = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.ffill(inplace=True)
    return df

if st.button("Initialize & Train 30-Day AI Model", type="primary"):
    with st.spinner("Compiling Tensors & Training Sequence Model (This will take a minute)..."):
        df = fetch_ai_data(ticker_symbol)
        
        # Preprocess
        rsi = RSIIndicator(close=df['Close'].squeeze(), window=14)
        df['RSI'] = rsi.rsi()
        df.dropna(inplace=True)
        data = df.filter(['Close', 'Volume', 'RSI']).values
        
        close_scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_close = close_scaler.fit_transform(data[:, 0].reshape(-1, 1))
        
        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = feature_scaler.fit_transform(data)
        scaled_features[:, 0] = scaled_close[:, 0]
        
        look_back = 60
        future_days = 30
        
        X, y = [], []
        for i in range(look_back, len(scaled_features) - future_days):
            X.append(scaled_features[i-look_back:i, :])
            y.append(scaled_features[i:i+future_days, 0]) # Target is the next 30 days
            
        X, y = np.array(X), np.array(y)
        
        # Build Multi-Step Model
        model = Sequential()
        model.add(LSTM(units=128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(units=96, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=future_days)) # Output 30 days
        model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
        
        # Train
        model.fit(X, y, batch_size=32, epochs=5, verbose=0)
        
        # Predict the Future
        last_60_data = scaled_features[-look_back:]
        X_future = np.array([last_60_data])
        future_pred_scaled = model.predict(X_future)
        future_prices = close_scaler.inverse_transform(future_pred_scaled).flatten()
        
        # Generate Future Dates (Skipping weekends)
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='B')

        # Charting
        st.subheader("🔮 30-Day Forward Projection")
        fig = go.Figure()
        
        # Plot last 60 days of actual data for context
        historical_context = df['Close'].tail(60)
        fig.add_trace(go.Scatter(x=historical_context.index, y=historical_context.squeeze(), mode='lines', name='Historical', line=dict(color='gray')))
        
        # Plot 30 day AI forecast
        fig.add_trace(go.Scatter(x=future_dates, y=future_prices, mode='lines+markers', name='AI Forecast', line=dict(color='#F23645', dash='dot')))
        
        fig.update_layout(height=500, template="plotly_dark", yaxis_title="Price Forecast")
        st.plotly_chart(fig, use_container_width=True)
        
        # Output specific targets
        c1, c2 = st.columns(2)
        c1.metric("Current Price", f"{currency}{historical_context.iloc[-1]:.2f}")
        c2.metric("AI Target (Day 30)", f"{currency}{future_prices[-1]:.2f}", f"{(future_prices[-1]-historical_context.iloc[-1]):.2f}")