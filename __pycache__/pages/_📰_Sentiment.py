import streamlit as st
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from transformers import pipeline

st.set_page_config(page_title="Sentiment Analysis", layout="wide")

ticker_symbol = st.session_state.get('ticker_symbol', 'RELIANCE.NS')
selected_company = st.session_state.get('selected_company', 'RELIANCE')

st.title("📰 News & AI Sentiment")
st.write(f"Running FinBERT NLP analysis on the latest headlines for **{selected_company}**.")

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

analyzer = load_sentiment_model()

with st.spinner("Scraping financial news and running NLP..."):
    clean_ticker = ticker_symbol.replace(".NS", "").replace("^", "")
    
    try:
        url = f'https://www.google.com/finance/quote/{clean_ticker}:NSE'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = [item.text for item in soup.find_all('div', class_='Yfwt5')][:10]
        
        if headlines:
            results = analyzer(headlines)
            score_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
            total_score = sum(score_map[res['label']] * res['score'] for res in results)
            score = total_score / len(headlines)
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "AI Sentiment Score"},
                gauge = {
                    'axis': {'range': [-1, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [-1, -0.2], 'color': "#F23645"},
                        {'range': [-0.2, 0.2], 'color': "gray"},
                        {'range': [0.2, 1], 'color': "#089981"}],
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("**Latest Headlines Analyzed:**")
            for i, news in enumerate(headlines):
                st.markdown(f"> {i+1}. {news}")
        else:
            st.warning("Could not fetch recent headlines right now.")
    except Exception as e:
        st.error(f"Sentiment Analysis failed: {e}")