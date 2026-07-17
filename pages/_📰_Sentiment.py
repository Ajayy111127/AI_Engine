import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Sentiment Analysis", layout="wide")
st.title("📰 Sentiment Analysis")

if "ticker_symbol" not in st.session_state:
    st.warning("Please select an asset from the main page.")
    st.stop()

ticker = st.session_state["ticker_symbol"]
company_name = st.session_state.get("selected_company", ticker)

st.subheader(f"News Sentiment for {company_name} ({ticker})")

st.info(
    "This lightweight version shows recent company news from Yahoo Finance.\n\n"
    "You can later upgrade this page to full NLP sentiment scoring using VADER / FinBERT."
)

try:
    stock = yf.Ticker(ticker)
    news = stock.news
except Exception as e:
    st.error(f"Unable to fetch news: {e}")
    st.stop()

if not news:
    st.warning("No recent news found for this asset.")
    st.stop()

rows = []
for item in news[:15]:
    title = item.get("title", "No title")
    publisher = item.get("publisher", "Unknown")
    link = item.get("link", "")
    provider_time = item.get("providerPublishTime", None)

    rows.append({
        "Title": title,
        "Publisher": publisher,
        "Link": link,
        "Published": provider_time
    })

news_df = pd.DataFrame(rows)

st.dataframe(news_df, use_container_width=True)

st.markdown("### Recent Headlines")
for idx, row in news_df.iterrows():
    st.markdown(f"**{idx+1}. {row['Title']}**")
    st.caption(f"Publisher: {row['Publisher']}")
    if row["Link"]:
        st.markdown(f"[Read article]({row['Link']})")
    st.markdown("---")