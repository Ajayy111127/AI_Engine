import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import RSIIndicator

def preprocess_multivariate_data(df, look_back=60):
    """
    Creates multivariate time-series sequences for LSTM.
    Uses Close + Volume + RSI as features.
    
    Returns:
        X, y, scaler, feature_columns
    """
    df = df.copy()

    # Basic validation
    required_cols = ['Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe.")

    # --- Feature Engineering ---
    # RSI from ta library
    rsi_indicator = RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi_indicator.rsi()

    # Keep only needed columns
    feature_columns = ['Close', 'Volume', 'RSI']
    df = df[feature_columns].dropna()

    if len(df) <= look_back:
        raise ValueError("Not enough data after indicator creation. Increase dataset size or reduce look_back.")

    # Scale features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    X, y = [], []

    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i])   # all 3 features
        y.append(scaled_data[i, 0])              # predict Close only

    X = np.array(X)
    y = np.array(y)

    return X, y, scaler, feature_columns