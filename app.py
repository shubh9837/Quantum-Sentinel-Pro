import streamlit as st
import pandas as pd

st.set_page_config(page_title="Quantum-Sentinel 10.0", layout="wide")

@st.cache_data
def load_all():
    try:
        data = pd.read_parquet("market_data.parquet")
        meta = pd.read_csv("tickers_enriched.csv")
        return data, meta
    except: return None, None

data, meta = load_all()

if data is None:
    st.error("Data missing. Please run Downloader in GitHub Actions.")
    st.stop()

# --- INSTITUTIONAL LOGIC: MARKET BREADTH ---
# Checking % of stocks above 50-day Moving Average
close_prices = data['Close']
ema50 = close_prices.ewm(span=50).mean()
breadth = (close_prices.iloc[-1] > ema50.iloc[-1]).mean() * 100

st.sidebar.title("🛡️ Safety Switch")
if breadth > 50:
    st.sidebar.success(f"BULLISH BREADTH: {breadth:.1f}%")
    market_modifier = 1
else:
    st.sidebar.error(f"BEARISH BREADTH: {breadth:.1f}%")
    market_modifier = -2

# --- VERDICT ENGINE ---
st.title("🎯 Quantum-Sentinel: Institutional Verdict")

if st.button("🔍 Run Full Market Scan"):
    results = []
    # Nifty benchmark for Relative Strength
    nifty = close_prices['^NSEI'].dropna()
    nifty_ret = (nifty.iloc[-1] - nifty.iloc[-60]) / nifty.iloc[-60]

    for ticker in close_prices.columns:
        if ticker == "^NSEI": continue
        
        try:
            s_price = close_prices[ticker].dropna()
            curr = s_price.iloc[-1]
            
            # Technical Indicators
            ema20 = s_price.ewm(span=20).mean().iloc[-1]
            rsi = 50 # Placeholder for simplicity, you can add full RSI logic
            
            # Relative Strength (Beating the Nifty?)
            s_ret = (curr - s_price.iloc[-60]) / s_price.iloc[-60]
            rs_score = 2 if s_ret > nifty_ret else 0
            
            # Final Scoring
            score = 0
            if curr > ema20: score += 3
            if rs_score > 0: score += 3
            score += market_modifier
            
            # Verdict Mapping
            if score >= 6: v, color = "🔥 STRONG BUY", "green"
            elif score >= 4: v, color = "✅ WATCH", "blue"
            else: v, color = "❌ AVOID", "red"
            
            sector = meta[meta['SYMBOL'] == ticker.replace(".NS","")]['SECTOR'].values[0]
            
            results.append({"Stock": ticker, "Sector": sector, "Verdict": v, "Score": score})
        except: continue

    res_df = pd.DataFrame(results).sort_values("Score", ascending=False)
    st.table(res_df.head(20))
