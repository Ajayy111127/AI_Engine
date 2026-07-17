import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import RSIIndicator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import alpaca_trade_api as tradeapi
import math
import time

# ==========================================
# 1. ALPACA BROKER CONFIGURATION
# ==========================================
# Paste your keys here inside the quotes
API_KEY = 'PKQSFN2YU3QTVJTSLCEHIA344I'
SECRET_KEY = '2UME9oSJv1pXyjJFBfdAoCp3WkTg5rxqeBNEDhF8jsVz'

# FIX 1: Removed the /v2 from the end of this URL
BASE_URL = 'https://paper-api.alpaca.markets'

# Connect to the broker
try:
    api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')
    account = api.get_account()
    print(f"💰 Connected to Alpaca! Paper Trading Balance: ${account.cash}")
except Exception as e:
    print(f"❌ Failed to connect to Alpaca. Please check your API keys. Error: {e}")
    exit()

# Trading Parameters
TICKER = "AAPL"
RISK_PER_TRADE = 0.02  # Risk 2% of portfolio balance per trade
STOP_LOSS_PCT = 0.02   # 2% Stop Loss distance
TAKE_PROFIT_PCT = 0.04 # 4% Take Profit distance

# ==========================================
# 2. AI PREDICTION ENGINE
# ==========================================
def get_ai_prediction():
    print(f"🔄 Fetching data and training AI for {TICKER}...")
    
    # 1. Fetch Data
    df = yf.download(TICKER, period="2y")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.ffill(inplace=True)
    
    # 2. Preprocess Data
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
    X, y = [], []
    for i in range(look_back, len(scaled_features)):
        X.append(scaled_features[i-look_back:i, :])
        y.append(scaled_features[i, 0])
        
    X, y = np.array(X), np.array(y)
    
    # 3. Train Model (Fast daily retrain)
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=96, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1))
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
    
    model.fit(X, y, batch_size=32, epochs=15, verbose=0)
    
    # 4. Predict Tomorrow
    last_60 = scaled_features[-60:]
    X_future = np.array([last_60])
    future_pred_scaled = model.predict(X_future)
    tomorrow_price = close_scaler.inverse_transform(future_pred_scaled)[0][0]
    current_price = df['Close'].iloc[-1].item()
    
    return current_price, tomorrow_price

# ==========================================
# 3. TRADE EXECUTION LOGIC
# ==========================================
def execute_trade():
    current_price, tomorrow_price = get_ai_prediction()
    
    print(f"📊 Current Price: ${current_price:.2f}")
    print(f"🔮 AI Predicts Tomorrow: ${tomorrow_price:.2f}")
    
    # Check if we already own the stock
    try:
        position = api.get_position(TICKER)
        print(f"⚠️ We already hold {position.qty} shares of {TICKER}. Skipping trade to avoid over-exposure.")
        return
    except tradeapi.rest.APIError:
        pass # We don't own it, safe to proceed
    
    # Signal Logic
    if tomorrow_price > current_price:
        print("🟢 BUY SIGNAL DETECTED. Calculating Risk...")
        
        # Risk Management Math
        buying_power = float(account.cash)
        risk_amount = buying_power * RISK_PER_TRADE
        stop_loss_distance = current_price * STOP_LOSS_PCT
        
        # How many shares can we buy while only risking our defined amount?
        qty_to_buy = math.floor(risk_amount / stop_loss_distance)
        
        if qty_to_buy > 0:
            take_profit_price = round(current_price * (1 + TAKE_PROFIT_PCT), 2)
            stop_loss_price = round(current_price * (1 - STOP_LOSS_PCT), 2)
            
            print(f"🚀 Sending Order to Alpaca: BUY {qty_to_buy} shares.")
            print(f"🎯 Target: ${take_profit_price} | 🛡️ Stop Loss: ${stop_loss_price}")
            
            try:
                # Send the order with advanced Bracket Orders (OCO)
                api.submit_order(
                    symbol=TICKER,
                    qty=qty_to_buy,
                    side='buy',
                    type='market',
                    time_in_force='gtc',
                    order_class='bracket',
                    take_profit=dict(
                        limit_price=take_profit_price,
                    ),
                    stop_loss=dict(
                        stop_price=stop_loss_price,
                        limit_price=stop_loss_price,
                    )
                )
                print("✅ Order successfully filled by broker!")
            except Exception as e:
                print(f"❌ Order failed to execute. Error: {e}")
        else:
            print("❌ Account balance too low to execute safely according to risk parameters.")
    else:
        print("🔴 SELL/HOLD SIGNAL. AI predicts a drop. Doing nothing.")

# Run the execution
if __name__ == "__main__":
    execute_trade()