import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(page_title="Command Center", layout="wide")

# Injecting Custom CSS for a Premium, Clean UI
st.markdown("""
    <style>
    /* Remove top padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    /* Style the metric containers */
    div[data-testid="metric-container"] {
        background-color: #1E1E1E;
        border: 1px solid #333;
        padding: 5% 5% 5% 10%;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    /* Hide the default Streamlit header and footer */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

currency = "₹"

# Watchlist for the Dashboard
# Watchlist for the Dashboard (MAXIMUM 6 STOCKS!)
DASHBOARD_TICKERS = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "RELIANCE": "RELIANCE.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS"

}

# ==========================================
# 2. FAST DATA FETCHING (CACHED)
# ==========================================
@st.cache_data(ttl=300)
def fetch_dashboard_data():
    """Fetches 7 days of data to calculate live changes and draw sparklines."""
    end_date = date.today()
    start_date = end_date - timedelta(days=10)
    
    data_dict = {}
    for name, ticker in DASHBOARD_TICKERS.items():
        try:
            df = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) >= 2:
                data_dict[name] = df['Close'].dropna()
        except:
            pass
    return data_dict

# Helper function to draw clean, minimalist sparklines
def draw_sparkline(series, color):
    fig = go.Figure(go.Scatter(
        x=series.index, y=series.values, mode='lines',
        line=dict(color=color, width=3),
        fill='tozeroy', fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.1,)}"
    ))
    fig.update_layout(
        height=80, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False, hovermode=False
    )
    return fig

# ==========================================
# 3. BUILD THE UI LAYOUT
# ==========================================
st.title("🏦 QuantTrade Command Center")
st.markdown("Global Market Overview & Portfolio Health")
st.markdown("---")

with st.spinner("Syncing live market data..."):
    market_data = fetch_dashboard_data()

    if market_data:
        # ROW 1: Top Level Metrics (Glassmorphism Cards)
        st.subheader("Market Pulse")
        metric_cols = st.columns(4)
        
        # We will display the first 4 assets in our metrics row
        metrics_to_show = list(DASHBOARD_TICKERS.keys())[:4]
        
        for i, name in enumerate(metrics_to_show):
            if name in market_data:
                series = market_data[name]
                current_price = series.iloc[-1].item()
                prev_price = series.iloc[-2].item()
                change = current_price - prev_price
                pct_change = (change / prev_price) * 100
                
                with metric_cols[i]:
                    st.metric(
                        label=name, 
                        value=f"{currency if 'NIFTY' not in name else ''}{current_price:,.2f}", 
                        delta=f"{change:+.2f} ({pct_change:+.2f}%)"
                    )

        st.markdown("<br>", unsafe_allow_html=True) # Spacer

        # ROW 2: Charts Grid (Sparklines + Allocation)
        col_sparklines, col_allocation = st.columns([6, 4])
        
        with col_sparklines:
            st.subheader("7-Day Trendlines")
            # Build a mini-grid for sparklines
            spark_cols = st.columns(3)
            
            for i, (name, series) in enumerate(market_data.items()):
                if i >= 6: break # Only show up to 6 trendlines
                
                current = series.iloc[-1].item()
                prev = series.iloc[-2].item()
                color = '#089981' if current >= prev else '#F23645'
                
                with spark_cols[i % 3]:
                    st.markdown(f"**{name}**")
                    st.plotly_chart(draw_sparkline(series, color), use_container_width=True, config={'displayModeBar': False})
                    
        with col_allocation:
            st.subheader("Simulated Portfolio Allocation")
            # Creating a beautiful dark-mode Donut Chart
            labels = ['RELIANCE', 'HDFCBANK', 'TCS', 'INFY', 'CASH']
            values = [35000, 25000, 20000, 10000, 10000]
            colors = ['#2962FF', '#FF6D00', '#00C853', '#FFD600', '#333333']
            
            fig_donut = go.Figure(data=[go.Pie(
                labels=labels, values=values, hole=.6, 
                marker=dict(colors=colors, line=dict(color='#0e1117', width=2))
            )])
            fig_donut.update_layout(
                height=300, margin=dict(l=0, r=0, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="v", yanchor="top", y=0.9, xanchor="left", x=1.05)
            )
            fig_donut.add_annotation(text="₹100K", x=0.5, y=0.5, font_size=24, showarrow=False, font_color="white")
            st.plotly_chart(fig_donut, use_container_width=True)

        st.markdown("---")

        # ROW 3: Detailed Data Table
        st.subheader("Watchlist Data Matrix")
        table_data = []
        for name, series in market_data.items():
            current = series.iloc[-1].item()
            prev = series.iloc[-2].item()
            high_7d = series.max().item()
            low_7d = series.min().item()
            change = current - prev
            pct = (change / prev) * 100
            
            table_data.append({
                "Asset": name,
                "Last Price": f"{current:,.2f}",
                "24h Change": f"{change:+.2f}",
                "24h %": f"{pct:+.2f}%",
                "7D High": f"{high_7d:,.2f}",
                "7D Low": f"{low_7d:,.2f}"
            })
            
        df_display = pd.DataFrame(table_data)
        
        # Use Streamlit's new highly styled dataframe integration
        st.dataframe(
            df_display, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "24h %": st.column_config.TextColumn(
                    "24h %", 
                    help="Percentage change over the last trading day"
                )
            }
        )
    else:
        st.error("Failed to load market data. Check your network connection.")