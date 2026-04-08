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
        # Validate data is not empty
        if data.empty or meta.empty:
            return None, None
        return data, meta
    except: return None, None

market_raw, meta = load_data()

# --- THE LOGIC ENGINE ---
def calculate_atr(df, period=14):
    if len(df) < period: return pd.Series([0] * len(df))
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
        pos = ['profit', 'growth', 'order', 'deal', 'expansion', 'buy', 'upgrade']
        for n in news:
            title = n.get('title', '').lower()
            if any(w in title for w in pos): score += 1
        return score
    except: return 0

# --- UI INTERFACE ---
st.title("🛡️ Quantum-Sentinel Pro")

if market_raw is None:
    st.error("⚠️ Data files are empty or missing. Please trigger 'Run workflow' in GitHub Actions and wait 5 minutes.")
    st.stop()

# --- ROBUST MARKET STATUS ---
try:
    prices = market_raw['Close']
    available_tickers = prices.columns.tolist()
    nifty_sym = "^NSEI" if "^NSEI" in available_tickers else available_tickers[0]
    nifty_close = prices[nifty_sym].dropna()
    
    if not nifty_close.empty:
        nifty_ema = nifty_close.ewm(span=50).mean()
        market_status = "BULLISH" if nifty_close.iloc[-1] > nifty_ema.iloc[-1] else "BEARISH"
        st.sidebar.metric("Market Trend", market_status)
    else:
        market_status = "NEUTRAL (No Data)"
        st.sidebar.warning("Nifty data missing")
except:
    market_status = "NEUTRAL"
    st.sidebar.info("Market Trend Unavailable")

tab1, tab2 = st.tabs(["🔍 Intelligence Screener", "💼 Portfolio Analytics"])

with tab1:
    min_score = st.slider("Minimum Rating", 1, 10, 7)
    if st.button("🚀 Run Multi-Factor Deep Scan"):
        results = []
        scan_list = [t for t in market_raw['Close'].columns if t != "^NSEI"]
        
        for t in scan_list:
            try:
                # Handle MultiIndex if present
                if isinstance(market_raw.columns, pd.MultiIndex):
                    df_stock = market_raw.xs(t, axis=1, level=1).dropna()
                else:
                    # Fallback for simpler structures
                    df_stock = pd.DataFrame({
                        'Close': market_raw['Close'][t],
                        'High': market_raw['High'][t],
                        'Low': market_raw['Low'][t]
                    }).dropna()
                
                if len(df_stock) < 20: continue
                
                curr = df_stock['Close'].iloc[-1]
                ema20 = df_stock['Close'].ewm(span=20).mean().iloc[-1]
                atr = calculate_atr(df_stock).iloc[-1]
                
                score = 5
                if curr > ema20: score += 1
                if market_status == "BULLISH": score += 2
                
                clean_sym = t.replace(".NS", "")
                sentiment = get_sentiment_score(clean_sym)
                score += sentiment
                
                results.append({
                    "Stock": clean_sym, "Rating": int(score),
                    "Price": round(curr, 2), "Target": round(curr + (atr*2), 2),
                    "Sentiment": "Positive" if sentiment > 0 else "Neutral"
                })
            except: continue
        
        if results:
            st.dataframe(pd.DataFrame(results).sort_values("Rating", ascending=False), hide_index=True)
        else:
            st.info("No stocks matched the criteria.")
            
