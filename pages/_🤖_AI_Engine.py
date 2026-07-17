import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="AI Engine", layout="wide")

# =========================
# Session state values
# =========================
currency = "₹"
ticker_symbol = st.session_state.get("ticker_symbol", "RELIANCE.NS")
selected_company = st.session_state.get("selected_company", "Reliance Industries")

st.title("🤖 AI Engine - 1 Month Future Forecast")
st.write(f"Future 30-business-day forecast for **{selected_company}**")

# =========================
# Settings
# =========================
look_back = 30
forecast_days = 30

# =========================
# Data Fetch
# =========================
@st.cache_data(ttl=1800)
def fetch_ai_data(ticker):
    end_date = date.today() + timedelta(days=1)
    start_date = end_date - timedelta(days=3 * 365)

    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    if "Date" not in df.columns:
        if "Datetime" in df.columns:
            df.rename(columns={"Datetime": "Date"}, inplace=True)

    needed = ["Date", "Open", "High", "Low", "Close", "Volume"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df[needed].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna().sort_values("Date").reset_index(drop=True)

    return df


# =========================
# Feature Engineering
# =========================
def add_features(df):
    df = df.copy()

    # Basic lag features
    df["Close_lag1"] = df["Close"].shift(1)
    df["Close_lag2"] = df["Close"].shift(2)
    df["Close_lag3"] = df["Close"].shift(3)
    df["Close_lag5"] = df["Close"].shift(5)
    df["Close_lag10"] = df["Close"].shift(10)

    # Returns / momentum
    df["Return_1d"] = df["Close"].pct_change(1)
    df["Return_5d"] = df["Close"].pct_change(5)
    df["Return_10d"] = df["Close"].pct_change(10)

    # Rolling stats
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()

    df["STD5"] = df["Close"].rolling(5).std()
    df["STD10"] = df["Close"].rolling(10).std()
    df["STD20"] = df["Close"].rolling(20).std()

    # Volume features
    df["Vol_MA5"] = df["Volume"].rolling(5).mean()
    df["Vol_MA10"] = df["Volume"].rolling(10).mean()

    # High/Low spread
    df["HL_Spread"] = df["High"] - df["Low"]
    df["OC_Spread"] = df["Open"] - df["Close"]

    # Target = next day close
    df["Target"] = df["Close"].shift(-1)

    df = df.dropna().reset_index(drop=True)
    return df


# =========================
# Train model
# =========================
def train_model(feature_df):
    feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "Close_lag1", "Close_lag2", "Close_lag3", "Close_lag5", "Close_lag10",
        "Return_1d", "Return_5d", "Return_10d",
        "MA5", "MA10", "MA20",
        "STD5", "STD10", "STD20",
        "Vol_MA5", "Vol_MA10",
        "HL_Spread", "OC_Spread"
    ]

    X = feature_df[feature_cols]
    y = feature_df["Target"]

    split_idx = int(len(feature_df) * 0.8)
    if split_idx < 50:
        raise ValueError("Not enough data to train the model.")

    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]

    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    test_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, test_pred)

    return model, feature_cols, mae, X_test, y_test, test_pred


# =========================
# Recursive future forecast
# =========================
def recursive_forecast(raw_df, model, feature_cols, days=30):
    temp = raw_df.copy().reset_index(drop=True)
    future_dates = []
    future_prices = []

    last_date = temp["Date"].iloc[-1]

    for _ in range(days):
        feat_df = add_features(temp.copy())

        if feat_df.empty:
            break

        latest_row = feat_df.iloc[-1:].copy()
        X_latest = latest_row[feature_cols]

        pred_close = float(model.predict(X_latest)[0])

        next_date = last_date + pd.offsets.BDay(1)
        last_date = next_date

        # Approximate next row using predicted close
        prev_close = temp["Close"].iloc[-1]
        prev_volume = temp["Volume"].iloc[-1]

        new_row = {
            "Date": next_date,
            "Open": prev_close,
            "High": max(prev_close, pred_close),
            "Low": min(prev_close, pred_close),
            "Close": pred_close,
            "Volume": prev_volume
        }

        temp = pd.concat([temp, pd.DataFrame([new_row])], ignore_index=True)

        future_dates.append(next_date)
        future_prices.append(pred_close)

    return future_dates, future_prices


# =========================
# Main
# =========================
try:
    raw_df = fetch_ai_data(ticker_symbol)

    if raw_df.empty:
        st.error("No data available for this stock.")
        st.stop()

    st.subheader("Latest Market Data")
    st.dataframe(raw_df.tail(10), use_container_width=True)

    if st.button("🚀 Generate 1-Month Future Forecast", type="primary"):
        with st.spinner("Training AI model and generating next 30 business-day forecast..."):
            feature_df = add_features(raw_df)

            if len(feature_df) < 80:
                st.error("Not enough historical data after feature creation.")
                st.stop()

            model, feature_cols, mae, X_test, y_test, test_pred = train_model(feature_df)

            future_dates, future_prices = recursive_forecast(
                raw_df=raw_df,
                model=model,
                feature_cols=feature_cols,
                days=forecast_days
            )

            if len(future_prices) == 0:
                st.error("Could not generate future prediction.")
                st.stop()

            future_prices = np.array(future_prices, dtype=float)

            # Confidence bands
            upper_band = future_prices + mae
            lower_band = np.maximum(future_prices - mae, 0)

            current_price = float(raw_df["Close"].iloc[-1])
            last_actual_date = pd.to_datetime(raw_df["Date"].iloc[-1])
            predicted_month_price = float(future_prices[-1])

            change = predicted_month_price - current_price
            pct_change = (change / current_price) * 100 if current_price != 0 else 0

            st.success("Future forecast generated successfully.")

            # =========================
            # Metrics
            # =========================
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current Price", f"{currency}{current_price:,.2f}")
            c2.metric("Current Date", str(last_actual_date.date()))
            c3.metric(
                f"Predicted Price After {forecast_days} Business Days",
                f"{currency}{predicted_month_price:,.2f}",
                f"{change:,.2f} ({pct_change:.2f}%)"
            )
            c4.metric("Forecast End Date", str(pd.to_datetime(future_dates[-1]).date()))

            # =========================
            # Forecast chart
            # =========================
            st.subheader(f"📈 Next {forecast_days} Business-Day Future Forecast")

            historical_context = raw_df.tail(90).copy()

            fig = go.Figure()

            # Historical
            fig.add_trace(go.Scatter(
                x=historical_context["Date"],
                y=historical_context["Close"],
                mode="lines",
                name="Actual Historical Price",
                line=dict(color="#4C78A8", width=3)
            ))

            # Future forecast
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=future_prices,
                mode="lines+markers",
                name="Future Predicted Price",
                line=dict(color="#F58518", width=4, dash="dot"),
                marker=dict(size=6)
            ))

            # Confidence range
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=upper_band,
                mode="lines",
                line=dict(color="rgba(245,133,24,0.20)", dash="dash"),
                name="Upper Range"
            ))

            fig.add_trace(go.Scatter(
                x=future_dates,
                y=lower_band,
                mode="lines",
                fill="tonexty",
                line=dict(color="rgba(245,133,24,0.20)", dash="dash"),
                name="Lower Range"
            ))

            # Join last actual to first forecast
            fig.add_trace(go.Scatter(
                x=[historical_context["Date"].iloc[-1], future_dates[0]],
                y=[historical_context["Close"].iloc[-1], future_prices[0]],
                mode="lines",
                line=dict(color="white", width=2, dash="dash"),
                name="Forecast Connection"
            ))

            # Forecast start marker
            fig.add_vline(
                x=future_dates[0],
                line_width=2,
                line_dash="dash",
                line_color="red"
            )

            fig.add_annotation(
                x=future_dates[0],
                y=max(max(historical_context["Close"]), max(future_prices)),
                text="Forecast Starts",
                showarrow=True,
                arrowhead=2,
                yshift=20
            )

            fig.update_layout(
                template="plotly_dark",
                height=650,
                title=f"{selected_company}: Actual Price vs Next {forecast_days} Business-Day Forecast",
                xaxis_title="Date",
                yaxis_title="Close Price",
                hovermode="x unified"
            )

            st.plotly_chart(fig, use_container_width=True)

            # =========================
            # Forecast table
            # =========================
            st.subheader("📋 1-Month Future Prediction Table")

            forecast_df = pd.DataFrame({
                "Future Date": pd.to_datetime(future_dates),
                "Predicted Close": np.round(future_prices, 2),
                "Lower Range": np.round(lower_band, 2),
                "Upper Range": np.round(upper_band, 2)
            })

            st.dataframe(forecast_df, use_container_width=True)

            csv = forecast_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇ Download Future Forecast CSV",
                data=csv,
                file_name=f"{ticker_symbol.replace('.NS','')}_future_1month_forecast.csv",
                mime="text/csv"
            )

            # =========================
            # Summary
            # =========================
            st.subheader("🎯 1-Month Future Target")
            st.write(
                f"**Last actual market close:** {currency}{current_price:,.2f} on **{last_actual_date.date()}**  \n"
                f"**Predicted close after {forecast_days} business days:** {currency}{predicted_month_price:,.2f} on **{pd.to_datetime(future_dates[-1]).date()}**"
            )

            # =========================
            # Backtest chart
            # =========================
            st.subheader("🧪 Model Backtest")

            backtest_dates = feature_df.iloc[int(len(feature_df)*0.8):]["Date"].iloc[:len(y_test)].values

            backtest_df = pd.DataFrame({
                "Date": backtest_dates,
                "Actual": y_test.values,
                "Predicted": test_pred
            }).set_index("Date")

            st.line_chart(backtest_df)

            # =========================
            # Interpretation
            # =========================
            st.subheader("🧠 AI Summary")

            if pct_change > 5:
                st.success(
                    f"The model suggests a **bullish 1-month outlook** for **{selected_company}**. "
                    f"Projected move: **{currency}{current_price:,.2f} → {currency}{predicted_month_price:,.2f}**"
                )
            elif pct_change < -5:
                st.warning(
                    f"The model suggests a **bearish 1-month outlook** for **{selected_company}**. "
                    f"Projected move: **{currency}{current_price:,.2f} → {currency}{predicted_month_price:,.2f}**"
                )
            else:
                st.info(
                    f"The model suggests a **sideways / moderate movement** for **{selected_company}** "
                    f"over the next {forecast_days} business days."
                )

except Exception as e:
    st.error(f"Error in AI Engine: {e}")