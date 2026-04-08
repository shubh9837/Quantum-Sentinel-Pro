import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import yfinance as yf

st.set_page_config(page_title="Quantum-Sentinel Pro", layout="wide")

# --- DATA & PORTFOLIO PERSISTENCE ---
PORTFOLIO_FILE = "portfolio.json"

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r") as f: return json.load(f)
        except: return {}
    return {}

def save_portfolio(p):
    with open(PORTFOLIO_FILE, "w") as f: json.dump(p, f)

@st.cache_data
def load_data():
    try:
        if not os.path.exists("market_data.parquet") or not os.path.exists("tickers_enriched.csv"):
            return None, None
        return pd.read_parquet("market_data.parquet"), pd.read_csv("tickers_enriched.csv")
    except: return None, None

market_raw, meta = load_data()

# --- BACKEND LOGIC UPGRADES ---
def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

@st.cache_data(ttl=3600)
def get_sentiment(sym):
    try:
        t = yf.Ticker(f"{sym}.NS")
        news = t.news[:3]
        score = sum(1 for n in news if any(w in n.get('title','').lower() for w in ['profit','growth','order','win','buy']))
        return score
    except: return 0

# --- INTERFACE RESTORATION ---
st.title("🎯 Quantum-Sentinel Pro")

if market_raw is None:
    st.error("⚠️ Data files not found. Please run the GitHub Action.")
    st.stop()

# SIDEBAR: Nifty Context (Fixed)
try:
    prices = market_raw['Close']
    nifty_sym = "^NSEI" if "^NSEI" in prices.columns else prices.columns[0]
    n_close = prices[nifty_sym].dropna()
    m_trend = "BULLISH" if n_close.iloc[-1] > n_close.ewm(span=50).mean().iloc[-1] else "BEARISH"
    st.sidebar.metric("Market Status", m_trend, delta="Nifty 50 Trend")
except:
    m_trend = "NEUTRAL"
    st.sidebar.warning("Nifty Data Syncing...")

tab1, tab2 = st.tabs(["📊 Screener", "💼 Portfolio Analytics"])

with tab1:
    min_rating = st.sidebar.slider("Min Buy Rating", 1, 10, 7)
    if st.button("🔍 Run Full Market Scan"):
        results = []
        prog = st.progress(0)
        ticks = [t for t in prices.columns if t != "^NSEI"]
        
        for i, t in enumerate(ticks):
            try:
                df = market_raw.xs(t, axis=1, level=1).dropna() if isinstance(market_raw.columns, pd.MultiIndex) else market_raw[t].dropna()
                if len(df) < 50: continue
                
                curr = df['Close'].iloc[-1]
                ema20 = df['Close'].ewm(span=20).mean().iloc[-1]
                atr = calculate_atr(df).iloc[-1]
                
                # STRENGTHENED SCORING
                score = 5
                if curr > ema20: score += 1
                if m_trend == "BULLISH": score += 2
                
                clean_s = t.replace(".NS","")
                sent = get_sentiment(clean_s)
                score += sent
                
                results.append({
                    "Stock": clean_s, "Rating": int(score), "Price": round(curr,1),
                    "Target": round(curr + (atr*2),1), "Sentiment": "Positive" if sent > 0 else "Neutral"
                })
            except: continue
            if i % 50 == 0: prog.progress(i/len(ticks))
        
        st.session_state['scan'] = pd.DataFrame(results)
        prog.empty()

    if 'scan' in st.session_state:
        df_res = st.session_state['scan']
        # Filter Logic Fix: Rating must be >= slider
        filtered = df_res[df_res['Rating'] >= min_rating].sort_values("Rating", ascending=False)
        st.dataframe(filtered, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Your Holdings")
    if 'portfolio' not in st.session_state: st.session_state['portfolio'] = load_portfolio()
    
    # Portfolio Input Area
    with st.expander("➕ Add/Update Stock"):
        c1, c2, c3 = st.columns(3)
        p_sym = c1.text_input("Symbol (e.g. RELIANCE)").upper()
        p_prc = c2.number_input("Avg Price")
        p_qty = c3.number_input("Qty", step=1)
        if st.button("Update Portfolio"):
            if p_qty <= 0: st.session_state['portfolio'].pop(p_sym, None)
            else: st.session_state['portfolio'][p_sym] = {"price": p_prc, "qty": p_qty}
            save_portfolio(st.session_state['portfolio'])
            st.rerun()

    if st.session_state['portfolio']:
        p_df = pd.DataFrame.from_dict(st.session_state['portfolio'], orient='index').reset_index()
        p_df.columns = ['Stock', 'Avg Price', 'Qty']
        st.table(p_df)
    else:
        st.info("Portfolio is empty. Add stocks above.")
        
