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
    st.error("Data missing. Please run the Downloader in GitHub Actions.")
    st.stop()

try:
    close_prices = data['Close']
except KeyError:
    st.error("Data format error. Please force a fresh download via GitHub Actions.")
    st.stop()

# --- INSTITUTIONAL LOGIC: MARKET BREADTH ---
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
    
    # Safe Nifty Index Loading
    if "^NSEI" in close_prices.columns:
        nifty = close_prices['^NSEI'].dropna()
        if len(nifty) > 60:
            nifty_ret = (nifty.iloc[-1] - nifty.iloc[-60]) / nifty.iloc[-60]
        else: nifty_ret = 0
    else:
        nifty_ret = 0
        st.sidebar.warning("Nifty benchmark missing. Relative Strength disabled.")

    progress_text = "Scanning 2600+ stocks..."
    my_bar = st.progress(0, text=progress_text)
    
    total_tickers = len(close_prices.columns)
    
    for i, ticker in enumerate(close_prices.columns):
        if i % 50 == 0: my_bar.progress(i / total_tickers, text=progress_text)
        if ticker == "^NSEI": continue
        
        try:
            s_price = close_prices[ticker].dropna()
            if len(s_price) < 60: continue # Skip new IPOs with not enough data
            
            curr = s_price.iloc[-1]
            
            ema20 = s_price.ewm(span=20).mean().iloc[-1]
            s_ret = (curr - s_price.iloc[-60]) / s_price.iloc[-60]
            rs_score = 2 if s_ret > nifty_ret else 0
            
            score = 0
            if curr > ema20: score += 3
            if rs_score > 0: score += 3
            score += market_modifier
            
            if score >= 6: v, action = "🔥 STRONG BUY", "Institutional Entry"
            elif score >= 4: v, action = "✅ WATCH", "Bullish Trend"
            else: v, action = "❌ AVOID", "Weakness"
            
            # --- THE SAFETY CATCH FOR SECTORS ---
            clean_ticker = ticker.replace(".NS", "")
            sector_match = meta.loc[meta['SYMBOL'] == clean_ticker, 'SECTOR'].values
            sector = sector_match[0] if len(sector_match) > 0 else "Unmapped"
            
            results.append({"Stock": clean_ticker, "Sector": sector, "Verdict": v, "Score": score})
        except: continue
    
    my_bar.empty()
    res_df = pd.DataFrame(results).sort_values("Score", ascending=False)
    st.dataframe(res_df, use_container_width=True)
    
