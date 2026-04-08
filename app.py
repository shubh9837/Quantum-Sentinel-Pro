import streamlit as st
import pandas as pd
import numpy as np
import os
import yfinance as yf

st.set_page_config(page_title="Quantum-Sentinel Pro", layout="wide")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    try:
        if not os.path.exists("market_data.parquet") or not os.path.exists("tickers_enriched.csv"):
            return None, None
        data = pd.read_parquet("market_data.parquet")
        meta = pd.read_csv("tickers_enriched.csv")
        return data, meta
    except: return None, None

market_raw, meta = load_data()

# --- THE LOGIC ENGINE ---
def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

@st.cache_data(ttl=3600)
def get_sentiment_score(symbol):
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        news = ticker.news[:3]
        score = 0
        positive_words = ['profit', 'growth', 'order', 'deal', 'expansion', 'buy', 'upgrade']
        for n in news:
            title = n.get('title', '').lower()
            if any(w in title for w in positive_words): score += 1
        return score
    except: return 0

# --- UI INTERFACE ---
st.title("🛡️ Quantum-Sentinel Pro (High-Logic)")

if market_raw is None:
    st.error("⚠️ Data missing! Please run your GitHub Action first.")
    st.stop()

# Get Prices and Index
prices = market_raw['Close']
available_tickers = prices.columns.tolist()
nifty_sym = "^NSEI" if "^NSEI" in available_tickers else available_tickers[0]

# Sidebar Market Status
nifty_close = prices[nifty_sym].dropna()
nifty_ema = nifty_close.ewm(span=50).mean()
market_status = "BULLISH" if nifty_close.iloc[-1] > nifty_ema.iloc[-1] else "BEARISH"
st.sidebar.metric("Market Trend", market_status, delta="Nifty 50 Context")

tab1, tab2 = st.tabs(["🔍 Intelligence Screener", "💼 Portfolio Analytics"])

with tab1:
    min_score = st.slider("Minimum Rating for Buy Signal", 1, 10, 7)
    
    if st.button("🚀 Run Multi-Factor Deep Scan"):
        results = []
        # Filter out the Index for scanning
        scan_list = [t for t in available_tickers if t != nifty_sym]
        
        for t in scan_list:
            try:
                # 1. Technical Indicators
                df_stock = market_raw.xs(t, axis=1, level=1) if isinstance(market_raw.columns, pd.MultiIndex) else market_raw[t]
                df_stock = df_stock.dropna()
                if len(df_stock) < 50: continue
                
                curr_price = df_stock['Close'].iloc[-1]
                ema20 = df_stock['Close'].ewm(span=20).mean().iloc[-1]
                atr = calculate_atr(df_stock).iloc[-1]
                
                # 2. Scoring Logic
                score = 5
                if curr_price > ema20: score += 1
                if market_status == "BULLISH": score += 2
                
                # 3. Sentiment Integration
                clean_sym = t.replace(".NS", "")
                sentiment = get_sentiment_score(clean_sym)
                score += sentiment
                
                # 4. Target Calculation (ATR based)
                target = curr_price + (atr * 2)
                
                results.append({
                    "Stock": clean_sym,
                    "Rating": int(score),
                    "Price": round(curr_price, 2),
                    "Target (ATR)": round(target, 2),
                    "Sentiment": "Positive" if sentiment > 0 else "Neutral"
                })
            except: continue
        
        # --- THE SAFETY SHIELD ---
        if results:
            full_df = pd.DataFrame(results)
            # We check if the 'Rating' column exists before filtering
            if 'Rating' in full_df.columns:
                filtered_df = full_df[full_df['Rating'] >= min_score].sort_values(by="Rating", ascending=False)
                st.dataframe(filtered_df, use_container_width=True, hide_index=True)
            else:
                st.warning("Scan completed but data format was incorrect.")
        else:
            st.info("No stocks met the criteria in this scan.")

with tab2:
    st.info("Portfolio logic is active. Update your holdings in the sidebar or portfolio.json to see live ATR targets.")
