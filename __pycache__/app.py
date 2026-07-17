import streamlit as st

# Global App Configuration
st.set_page_config(page_title="QuantTrade Pro Terminal", layout="wide", initial_sidebar_state="expanded")

st.sidebar.title("🏦 QuantTrade Pro")
st.sidebar.markdown("Use the pages above to navigate between charting, AI predictions, and sentiment analysis.")
st.sidebar.markdown("---")

# Global Ticker Database (Indian Market Focus) 

TICKERS = {
    # --- INDICES ---
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    
    # --- TOP CAP & NIFTY 50 ---
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "State Bank of India": "SBIN.NS",
    "Infosys": "INFY.NS",
    "ITC Limited": "ITC.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "Larsen & Toubro": "LT.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "HCL Technologies": "HCLTECH.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "NTPC": "NTPC.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Titan Company": "TITAN.NS",
    "ONGC": "ONGC.NS",
    "Tata Steel": "TATASTEEL.NS",
    "Coal India": "COALINDIA.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "Power Grid Corp": "POWERGRID.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Wipro": "WIPRO.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Nestle India": "NESTLEIND.NS",
    "Grasim Industries": "GRASIM.NS",
    "Tech Mahindra": "TECHM.NS",
    "Hindalco Industries": "HINDALCO.NS",
    "Divi's Laboratories": "DIVISLAB.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Tata Consumer Products": "TATACONSUM.NS",
    "HDFC Life Insurance": "HDFCLIFE.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS",
    "SBI Life Insurance": "SBILIFE.NS",
    "Cipla": "CIPLA.NS",
    "Dr. Reddy's Laboratories": "DRREDDY.NS",
    "Britannia Industries": "BRITANNIA.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "BPCL": "BPCL.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    "Eicher Motors": "EICHERMOT.NS",
    
    # --- NIFTY NEXT 50 & HIGH GROWTH ---
    "Zomato": "ZOMATO.NS",
    "Jio Financial Services": "JIOFIN.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "Trent": "TRENT.NS",
    "Hindustan Aeronautics (HAL)": "HAL.NS",
    "Bharat Electronics (BEL)": "BEL.NS",
    "Siemens": "SIEMENS.NS",
    "DLF": "DLF.NS",
    "InterGlobe Aviation (Indigo)": "INDIGO.NS",
    "Varun Beverages": "VBL.NS",
    "Godrej Consumer Products": "GODREJCP.NS",
    "Pidilite Industries": "PIDILITIND.NS",
    "Cholamandalam Investment": "CHOLAFIN.NS",
    "Bank of Baroda": "BANKBARODA.NS",
    "Punjab National Bank": "PNB.NS",
    "Vedanta": "VEDL.NS",
    "Havells India": "HAVELLS.NS",
    "TVS Motor": "TVSMOTOR.NS",
    "Zydus Lifesciences": "ZYDUSLIFE.NS",
    "Shriram Finance": "SHRIRAMFIN.NS",
    "SRF Limited": "SRF.NS",
    "Torrent Pharma": "TORNTPHARM.NS",
    "Cummins India": "CUMMINSIND.NS",
    "Ashok Leyland": "ASHOKLEY.NS",
    "Max Healthcare": "MAXHEALTH.NS",
    "ICICI Prudential Life": "ICICIPRULI.NS",
    "CG Power": "CGPOWER.NS",
    "PI Industries": "PIIND.NS",
    "Polycab India": "POLYCAB.NS",
    "Muthoot Finance": "MUTHOOTFIN.NS",
    "IDFC First Bank": "IDFCFIRSTB.NS",
    "Bosch Limited": "BOSCHLTD.NS",
    "LIC Housing Finance": "LICHSGFIN.NS",
    "Canara Bank": "CANBK.NS",
    "Indian Bank": "INDIANB.NS",
    "Yes Bank": "YESBANK.NS",
    
    # --- INFRA, ENERGY, RAILWAYS & TECH ---
    "Suzlon Energy": "SUZLON.NS",
    "NHPC": "NHPC.NS",
    "SJVN": "SJVN.NS",
    "IRFC": "IRFC.NS",
    "Rail Vikas Nigam (RVNL)": "RVNL.NS",
    "IREDA": "IREDA.NS",
    "Paytm (One97)": "PAYTM.NS",
    "Nykaa (FSN)": "NYKAA.NS",
    "PB Fintech (PolicyBazaar)": "POLICYBZR.NS",
    "Delhivery": "DELHIVERY.NS",
    "Info Edge (Naukri)": "NAUKRI.NS",
    "MRF Limited": "MRF.NS",
    "Page Industries": "PAGEIND.NS",
    "Berger Paints": "BERGEPAINT.NS",
    "Marico": "MARICO.NS",
    "Dabur India": "DABUR.NS",
    "ICICI Lombard": "ICICIGI.NS",
    "SBI Cards": "SBICARD.NS",
    "HDFC AMC": "HDFCAMC.NS",
    "Nippon Life India AMC": "NAM-INDIA.NS"
}

# Initialize Session State for Global Variables (With Safety Fallback)
if 'selected_company' not in st.session_state or st.session_state['selected_company'] not in TICKERS:
    st.session_state['selected_company'] = "Reliance Industries" # Updated name
if 'ticker_symbol' not in st.session_state:
    st.session_state['ticker_symbol'] = "RELIANCE.NS"

# Global Sidebar Asset Selector
st.sidebar.header("Global Asset Selector")
new_company = st.sidebar.selectbox("Select Asset", list(TICKERS.keys()), index=list(TICKERS.keys()).index(st.session_state['selected_company']))

# Update session state if changed
if new_company != st.session_state['selected_company']:
    st.session_state['selected_company'] = new_company
    st.session_state['ticker_symbol'] = TICKERS[new_company]
    st.rerun()

st.title("Welcome to QuantTrade Pro Terminal")
st.write("👈 **Please select a module from the sidebar to begin your analysis.**")