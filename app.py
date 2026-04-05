import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Quantum-Sentinel Pro", layout="wide")

@st.cache_data
def load_all():
    try:
        data = pd.read_parquet("market_data.parquet", engine='pyarrow')
        meta = pd.read_csv("tickers_enriched.csv")
        return data, meta
    except: return None, None

data, meta = load_all()

if data is None:
    st.error("Data missing. Please run Downloader in GitHub.")
    st.stop()

# Helper to fix symbols for mapping
def clean_sym(s): return s.replace(".NS", "").replace("_", "&")

close_prices = data['Close']
volumes = data['Volume']

# --- SIDEBAR: RISK & SECTOR ---
ema50_market = close_prices.ewm(span=50).mean()
breadth = (close_prices.iloc[-1] > ema50_market.iloc[-1]).mean() * 100
market_bonus = 1 if breadth > 50 else (-2 if breadth < 30 else 0)

st.sidebar.title("🛡️ Market Risk")
if breadth > 50: st.sidebar.success(f"🟢 BULLISH: {breadth:.1f}%")
else: st.sidebar.error(f"🔴 BEARISH: {breadth:.1f}%")

# --- ENGINE ---
st.title("🎯 Quantum-Sentinel: Pro Scanner")

if st.button("🚀 Run Full Market Scan"):
    results = []
    nifty = close_prices['^NSEI'].dropna()
    nifty_ret = (nifty.iloc[-1] - nifty.iloc[-60])/nifty.iloc[-60] if len(nifty)>60 else 0

    progress = st.progress(0)
    cols = close_prices.columns
    for i, ticker in enumerate(cols):
        if ticker == "^NSEI": continue
        try:
            s_data = close_prices[ticker].dropna()
            if s_data.empty: continue
            
            curr = s_data.iloc[-1]
            ema20 = s_data.ewm(span=20).mean().iloc[-1]
            ema50_v = s_data.ewm(span=50).mean().iloc[-1]
            
            # Simple RSI
            delta = s_data.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
            loss = -delta.where(delta < 0, 0).rolling(14).mean().iloc[-1]
            rsi = 100 - (100/(1 + gain/loss)) if loss > 0 else 50
            
            # Scoring (0-10)
            score = 4 + market_bonus
            if curr > ema20: score += 1
            if ema20 > ema50_v: score += 2
            if rsi > 50: score += 1
            if (curr - s_data.iloc[-20])/s_data.iloc[-20] > nifty_ret: score += 2
            
            score = max(0, min(10, int(score)))
            
            results.append({
                "Stock": clean_sym(ticker),
                "Score": score,
                "Price": round(float(curr), 1),
                "Target": round(float(curr * 1.15), 1),
                "StopLoss": round(float(curr * 0.93), 1),
                "Upside%": "15.0%",
                "RSI": round(float(rsi), 1),
                "Verdict": "🔥 BUY" if score >= 8 else ("✅ HOLD" if score >= 5 else "❌ AVOID")
            })
        except: continue
        if i % 100 == 0: progress.progress(i/len(cols))
    
    st.session_state['results'] = pd.DataFrame(results).sort_values(by="Score", ascending=False)
    progress.empty()

# --- DISPLAY & SEARCH ---
if 'results' in st.session_state:
    df = st.session_state['results']
    
    search = st.text_input("🔍 Search Stock Symbol (e.g. RELIANCE)").upper()
    if search:
        res = df[df['Stock'].str.contains(search)]
        st.table(res)
    
    st.subheader("📊 All Ranked Stocks")
    st.dataframe(df, use_container_width=True, hide_index=True)
