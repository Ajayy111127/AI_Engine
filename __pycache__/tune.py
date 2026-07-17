import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import RSIIndicator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt

print("Fetching data for tuning...")

# 1. Fetch & Preprocess Data
df = yf.download('AAPL', start='2018-01-01', end='2024-01-01')

# --- FIX 1: Flatten MultiIndex columns from yfinance ---
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# --- FIX 2: Updated Pandas ffill syntax ---
df.ffill(inplace=True) 

# --- FIX 3: Added .squeeze() to flatten the 2D array into 1D for the TA library ---
rsi_indicator = RSIIndicator(close=df['Close'].squeeze(), window=14)
df['RSI'] = rsi_indicator.rsi()
df.dropna(inplace=True)

# Select features and scale
data = df.filter(['Close', 'Volume', 'RSI']).values
feature_scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = feature_scaler.fit_transform(data)

# Create sequences
look_back = 60
X, y = [], []
for i in range(look_back, len(scaled_features)):
    X.append(scaled_features[i-look_back:i, :])
    y.append(scaled_features[i, 0])

X, y = np.array(X), np.array(y)

# Train/Test Split
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# 2. Define the Tunable Model
def build_tunable_model(hp):
    """Builds an LSTM where KerasTuner decides the exact architecture."""
    model = Sequential()
    
    hp_units_1 = hp.Int('units_1', min_value=32, max_value=128, step=32)
    model.add(LSTM(units=hp_units_1, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    
    hp_dropout_1 = hp.Float('dropout_1', min_value=0.1, max_value=0.4, step=0.1)
    model.add(Dropout(hp_dropout_1))
    
    hp_units_2 = hp.Int('units_2', min_value=32, max_value=128, step=32)
    model.add(LSTM(units=hp_units_2, return_sequences=False))
    model.add(Dropout(hp_dropout_1)) 
    
    hp_dense = hp.Int('dense_units', min_value=16, max_value=64, step=16)
    model.add(Dense(units=hp_dense, activation='relu'))
    model.add(Dense(units=1))
    
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=Adam(learning_rate=hp_learning_rate), loss='mean_squared_error')
    
    return model

# 3. Initialize the Keras Tuner
print("Starting Hyperparameter Search...")
tuner = kt.RandomSearch(
    build_tunable_model,
    objective='val_loss', 
    max_trials=10,        
    executions_per_trial=1,
    directory='stock_tuning',
    project_name='lstm_optimization'
)

# 4. Run the Search
tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

# 5. Output the Winner
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("\n" + "="*50)
print("🏆 OPTIMAL HYPERPARAMETERS FOUND 🏆")
print(f"- First LSTM Layer Units: {best_hps.get('units_1')}")
print(f"- Second LSTM Layer Units: {best_hps.get('units_2')}")
print(f"- Dense Layer Units: {best_hps.get('dense_units')}")
print(f"- Dropout Rate: {best_hps.get('dropout_1')}")
print(f"- Learning Rate: {best_hps.get('learning_rate')}")
print("="*50)