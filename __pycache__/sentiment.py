import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import pandas as pd

# Load the FinBERT financial sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")

def get_stock_news(ticker):
    """Scrapes recent news headlines for the given ticker."""
    # Using a basic Google Finance search URL for the ticker
    url = f'https://www.google.com/finance/quote/{ticker}:NASDAQ'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extracting headline elements (Note: HTML classes change, this is a general structure)
    headlines = []
    for item in soup.find_all('div', class_='Yfwt5'): 
        headlines.append(item.text)
        
    return headlines[:10] # Return top 10 recent headlines

def analyze_sentiment(ticker):
    """Calculates an average sentiment score (-1 to 1) for the stock."""
    headlines = get_stock_news(ticker)
    
    if not headlines:
        return 0.0 # Neutral if no news found
        
    results = sentiment_analyzer(headlines)
    
    # Convert labels to numerical scores
    score_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    total_score = 0
    
    for result in results:
        label = result['label']
        confidence = result['score']
        total_score += score_map[label] * confidence
        
    average_score = total_score / len(headlines)
    return average_score