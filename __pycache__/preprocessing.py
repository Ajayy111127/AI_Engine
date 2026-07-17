import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import RSIIndicator

def preprocess_multivariate_data(df, look_back=60):
    """Calculates indicators, scales data, and creates multivariate sequences."""
    
    # 1. Feature Engineering (Adding RSI)
    # Using a standard 14-day RSI
    rsi_indicator = RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi_indicator.rsi()
    
    # Drop NaN values created by the 14-day RSI window
    df.dropna(inplace=True)
    
    # 2. Select Features
    features = ['Close', 'Volume', 'RSI']
    data = df.filter(features).values
    
    # 3. Scaling
    # We need a dedicated scaler for 'Close' to inverse_transform predictions later
    close_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close = close_scaler.fit_transform(data[:, 0].reshape(-1, 1))
    
    # Scaler for all features combined
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = feature_scaler.fit_transform(data)
    
    # Replace the Close column in scaled_features with the independently scaled Close data 
    # to maintain consistency
    scaled_features[:, 0] = scaled_close[:, 0]
    
    # 4. Create Sequences (Windows)
    X, y = [], []
    for i in range(look_back, len(scaled_features)):
        # X gets all features for the look_back window
        X.append(scaled_features[i-look_back:i, :])
        # y gets only the 'Close' price for the target day
        y.append(scaled_features[i, 0])
        
    X, y = np.array(X), np.array(y)
    
    # 5. Train/Test Split (80/20)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, y_train, X_test, y_test, close_scaler, df