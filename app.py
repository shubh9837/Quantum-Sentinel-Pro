import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import requests
import base64
import yfinance as yf # New import for News fetching

st.set_page_config(page_title="Quantum-Sentinel Pro", layout="wide")

PORTFOLIO_FILE = "portfolio.json"

# --- GITHUB PERSISTENCE ---
def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except: return {}
    return {}

def save_portfolio(portfolio):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f)
    try:
        if "GITHUB_TOKEN" in st.secrets and "REPO_NAME" in st.secrets:
            token = st.secrets["GITHUB_TOKEN"]
            repo = st.secrets["REPO_NAME"]
            url = f"https://api.github.com/repos/{repo}/contents/{PORTFOLIO_FILE}"
            headers = {"Authorization": f"token {token}"}
            res = requests.get(url, headers=headers)
            sha = res.json().get("sha") if res.status_code == 200 else None
            content = base64.b64encode(json.dumps(portfolio).encode()).decode()
            payload = {"message": "Update Portfolio", "content": content, "branch": "main"}
            if sha: payload["sha"] = sha
            requests.put(url, headers=headers, json=payload)
    except Exception as e: st.error(f"Sync Failed: {e}")

@st.cache_data
def load_all_data():
    try:
        data = pd.read_parquet("market_data.parquet", engine='pyarrow')
        meta = pd.read_csv("tickers_enriched.csv")
        return data, meta
    except: return None, None

data, meta = load_all_data()
if data is None:
    st.error("Market data missing.")
    st.stop()

def clean_sym(s): return s.replace(".NS", "").replace("_", "&")

# --- NEWS SENTIMENT ENGINE ---
def get_sentiment_score(symbol):
    """Simple keyword-based sentiment analyzer for stock news."""
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        news = ticker.news[:5] # Get last 5 headlines
        score = 0
        pos_words = ['growth', 'profit', 'order', 'win', 'buy', 'upgrade', 'expansion', 'dividend']
        neg_words = ['loss', 'debt', 'fraud', 'penalty', 'sell', 'downgrade', 'slump', 'investigation']
        
        for n in news:
            title = n['title'].lower()
            for w in pos_words: 
                if w in title: score += 1
            for w in neg_words: 
                if w in title: score -= 1
        return 1 if score > 1 else -1 if score < -1 else 0
    except: return 0

# --- SIDEBAR ---
st.sidebar.title("🛡️ Sentinel Controls")
n_sym = "^NSEI" if "^NSEI" in data.columns else data.columns[0]
n_close = data['Close'][n_sym].dropna()
n_curr = n_close.iloc[-1]
market_trend = "BULLISH" if n_curr > n_close.ewm(span=50).mean().iloc[-1] else "BEARISH"

min_rating = st.sidebar.slider("Min Rating", 0, 10, 7)
mcap_filter = st.sidebar.selectbox("Market Cap Focus", ["All", "Large Cap Only", "Small/Mid Cap Only"])

st.title("🎯 Quantum-Sentinel: Intelligence-Driven Trading")
tab1, tab2 = st.tabs(["📊 Smart Screener", "💼 Portfolio Analytics"])

with tab1:
    if st.button("🔍 Run Multi-Factor Scan"):
        results = []
        prog = st.progress(0)
        ticks = [t for t in data['Close'].columns if t != n_sym]
        
        for i, t in enumerate(ticks):
            try:
                c_data = data['Close'][t].dropna()
                if len(c_data) < 200: continue
                
                curr = c_data.iloc[-1]
                e20, e50, e200 = c_data.ewm(span=20).mean().iloc[-1], c_data.ewm(span=50).mean().iloc[-1], c_data.ewm(span=200).mean().iloc[-1]
                
                # 1. Technical Score
                score = 3 + (2 if market_trend == "BULLISH" else -2)
                if curr > e200: score += 2
                if e20 > e50: score += 1
                
                # RSI Logic
                delta = c_data.diff()
                up, down = delta.clip(lower=0).ewm(com=13).mean(), -1*delta.clip(upper=0).ewm(com=13).mean()
                rsi = 100 - (100 / (1 + (up/down))).iloc[-1]
                if 45 <= rsi <= 68: score += 1
                elif rsi > 75: score -= 2 # Overbought filter
                
                # 2. Fundamental & M-Cap Filter
                clean_t = clean_sym(t)
                stock_meta = meta[meta['SYMBOL'] == clean_t]
                m_cap = stock_meta['MARKET_CAP'].values[0] if not stock_meta.empty else 0
                
                # Logical Strengthening for Small Caps
                is_small_cap = m_cap < 5000 # Example: < 5000 Cr
                if is_small_cap:
                    # Small caps MUST have volume breakout to be "Genuine"
                    v_data = data['Volume'][t].dropna()
                    if v_data.iloc[-1] < v_data.rolling(20).mean().iloc[-1] * 1.5:
                        score -= 2 # Penalty for low-volume small cap signal
                
                # 3. Sentiment Integration (Only for top candidates to save time)
                sentiment = 0
                if score >= 6:
                    sentiment = get_sentiment_score(clean_t)
                    score += sentiment
                
                # 4. ATR Target
                h, l = data['High'][t].dropna(), data['Low'][t].dropna()
                tr = pd.concat([h-l, (h-c_data.shift()).abs(), (l-c_data.shift()).abs()], axis=1).max(axis=1)
                atr = tr.rolling(14).mean().iloc[-1]
                target = round(float(curr + (2.5 * atr)), 1)
                
                rating = max(0, min(10, int(score)))
                results.append({
                    "Stock": clean_t, "Rating": rating, "M-Cap": round(m_cap, 0),
                    "Price": round(curr, 1), "Target": target, "Upside %": round(((target-curr)/curr)*100, 1),
                    "Sentiment": "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral",
                    "Verdict": "🔥 Strong Buy" if rating >= 8 else "✅ Accumulate" if rating >= 6 else "↔️ Avoid"
                })
            except: continue
            if i % 50 == 0: prog.progress(i/len(ticks))
            
        st.session_state['raw_results'] = pd.DataFrame(results)
        prog.empty()

    if 'raw_results' in st.session_state:
        df = st.session_state['raw_results'].copy()
        if mcap_filter == "Large Cap Only": df = df[df['M-Cap'] > 20000]
        elif mcap_filter == "Small/Mid Cap Only": df = df[df['M-Cap'] < 20000]
        st.dataframe(df[df['Rating'] >= min_rating].sort_values("Rating", ascending=False), use_container_width=True, hide_index=True)

with tab2:
    # Portfolio logic remains same but now displays the "Sentinel Rating" for your holdings
    if 'portfolio' not in st.session_state: st.session_state['portfolio'] = load_portfolio()
    # ... (Rest of portfolio code from previous version)
