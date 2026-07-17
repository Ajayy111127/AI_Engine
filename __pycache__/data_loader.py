import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, start_date, end_date):
    """Fetches historical stock data from Yahoo Finance."""
    print(f"Fetching data for {ticker}...")
    df = yf.download(ticker, start=start_date, end=end_date)
    # Forward fill missing values
    df.fillna(method='ffill', inplace=True)
    return df

# Example Usage:
# df = fetch_stock_data('RELIANCE.NS', '2020-01-01', '2026-04-11')