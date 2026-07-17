from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

def build_multivariate_lstm(input_shape):
    """Builds an LSTM architecture for multidimensional inputs."""
    model = Sequential()
    
    # input_shape will now be (look_back, 3) instead of (look_back, 1)
    model.add(LSTM(units=64, return_sequences=True, input_shape=(input_shape[1], input_shape[2])))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.2))
    
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1)) # We still only output a single value (the predicted Close price)
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model