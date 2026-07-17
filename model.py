from sklearn.ensemble import RandomForestRegressor

def build_prediction_model():
    """
    Builds a simple prediction model using RandomForestRegressor.
    This replaces TensorFlow LSTM for compatibility.
    """
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=10
    )
    return model