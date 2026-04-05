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
    except Exception as e: 
        return None, None

data, meta = load_all()

if data is None:
    st.error("Data missing or corrupted. Please run Downloader in GitHub Actions.")
    st.stop()

try:
    close_prices = data['Close']
    volumes = data['Volume']
except KeyError:
    st.error("Data format error. Please ensure downloader.py used group_by='column'.")
    st.stop()

# --- 1. MARKET BREADTH (The Safety Switch) ---
ema50_market = close_prices.ewm(span=50).mean()
breadth = (close_prices.iloc[-1] > ema50_market.iloc[-1]).mean() * 100

st.sidebar.title("🛡️ Market Environment")
if breadth > 50:
    st.sidebar.success(f"🟢 BULLISH: {breadth:.1f}%")
    market_bonus = 1
elif breadth > 30:
    st.sidebar.warning(f"🟡 NEUTRAL: {breadth:.1f}%")
    market_bonus = 0
else:
    st.sidebar.error(f"🔴 BEARISH: {breadth:.1f}%")
    market_bonus = -2

# --- 2. SECTOR SENTIMENT ---
st.sidebar.markdown("---")
st.sidebar.subheader("🔥 Top 3 Trending Sectors")
sector_returns = {}

for ticker in close_prices.columns:
    if ticker == "^NSEI" or len(close_prices[ticker].dropna()) < 25: continue
    clean_ticker = ticker.replace(".NS", "")
    try:
        sector = meta.loc[meta['SYMBOL'] == clean_ticker, 'SECTOR'].values[0]
        ret_1m = (close_prices[ticker].dropna().iloc[-1] - close_prices[ticker].dropna().iloc[-20]) / close_prices[ticker].dropna().iloc[-20]
        if sector not in sector_returns: sector_returns[sector] = []
        sector_returns[sector].append(ret_1m)
    except: continue

sector_avg = {sec: np.mean(rets) for sec, rets in sector_returns.items()}
top_sectors = sorted(sector_avg, key=sector_avg.get, reverse=True)[:3]

for sec in top_sectors:
    st.sidebar.markdown(f"**- {sec}**")

# --- 3. SIGNAL ENGINE ---
st.title("🎯 Quantum-Sentinel: Institutional Scanner")
st.caption("Scoring 0-10 based on Trend, Momentum, Volume, and Sector Sentiment.")

if st.button("🚀 Run Full Market Analysis"):
    results = []
    
    if "^NSEI" in close_prices.columns:
        nifty = close_prices['^NSEI'].dropna()
        nifty_ret = (nifty.iloc[-1] - nifty.iloc[-60]) / nifty.iloc[-60] if len(nifty) > 60 else 0
    else: nifty_ret = 0

    progress_bar = st.progress(0, text="Scoring all stocks...")
    total = len(close_prices.columns)

    for i, ticker in enumerate(close_prices.columns):
        if i % 50 == 0: progress_bar.progress(i / total)
        if ticker == "^NSEI": continue
        
        try:
            s_data = close_prices[ticker].dropna()
            v_data = volumes[ticker].dropna()
            if len(s_data) < 60: continue
            
            curr_price = round(s_data.iloc[-1], 2)
            
            ema20 = s_data.ewm(span=20).mean().iloc[-1]
            ema50_val = s_data.ewm(span=50).mean().iloc[-1]
            
            delta = s_data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean().iloc[-1]
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().iloc[-1]
            rs = gain / loss if loss != 0 else 0
            rsi = 100 - (100 / (1 + rs))
            
            avg_vol_20 = v_data.rolling(window=20).mean().iloc[-1]
            curr_vol = v_data.iloc[-1]
            vol_surge = curr_vol > (1.5 * avg_vol_20)
            
            stock_ret = (curr_price - s_data.iloc[-60]) / s_data.iloc[-60]
            is_beating_nifty = stock_ret > nifty_ret
            
            clean_ticker = ticker.replace(".NS", "")
            sector_match = meta.loc[meta['SYMBOL'] == clean_ticker, 'SECTOR'].values
            sector = sector_match[0] if len(sector_match) > 0 else "Unmapped"

            # 0 to 10 SCORING
            score = 3
            if curr_price > ema20 and ema20 > ema50_val: score += 2 
            if is_beating_nifty: score += 1                         
            if 55 <= rsi <= 75: score += 1                          
            if vol_surge: score += 1                                
            if sector in top_sectors: score += 1                    
            score += market_bonus                                   
            
            score = max(0, min(10, int(score))) # Lock strictly between 0 and 10
            
            # Universal Trade Math
            target = round(curr_price * 1.15, 2)
            stop_loss = round(min(curr_price * 0.93, ema50_val), 2)
            
            if score >= 8: verdict = "🔥 STRONG BUY"
            elif score >= 6: verdict = "✅ ACCUMULATE"
            elif score >= 4: verdict = "🟡 HOLD / WATCH"
            else: verdict = "❌ AVOID"
                
            results.append({
                "Stock": clean_ticker,
                "Score": score,
                "Sector": sector,
                "Price": curr_price,
                "Target": target,
                "Stop Loss": stop_loss,
                "RSI": round(rsi, 1),
                "Vol Surge": "Yes" if vol_surge else "No",
                "Verdict": verdict
            })
        except: continue

    progress_bar.empty()
    
    if results:
        df_res = pd.DataFrame(results)
        # Sort descending by Score, then RSI
        df_res = df_res.sort_values(by=["Score", "RSI"], ascending=[False, False])
        
        # Save dataframe to session state so search works without re-running
        st.session_state['full_results'] = df_res

# --- 4. SEARCH & DISPLAY UI ---
if 'full_results' in st.session_state:
    df_res = st.session_state['full_results']
    
    st.markdown("---")
    st.subheader("🔍 Search Specific Stock")
    search_query = st.text_input("Enter Stock Symbol (e.g., RELIANCE, INFY)").strip().upper()
    
    if search_query:
        search_result = df_res[df_res['Stock'] == search_query]
        if not search_result.empty:
            st.success(f"Found Data for {search_query}")
            st.dataframe(search_result, use_container_width=True, hide_index=True)
        else:
            st.error(f"Stock '{search_query}' not found. Check spelling or ensure it is in the NSE list.")
    
    st.markdown("---")
    st.subheader(f"📊 Full Market Rankings ({len(df_res)} Stocks)")
    
    def highlight_score(val):
        if val >= 8: return 'background-color: #004d00; color: white'
        elif val >= 6: return 'background-color: #556b2f; color: white'
        elif val <= 3: return 'background-color: #4d0000; color: white'
        return ''
        
    st.dataframe(
        df_res.style.map(highlight_score, subset=['Score']), 
        use_container_width=True, 
        hide_index=True
    )
