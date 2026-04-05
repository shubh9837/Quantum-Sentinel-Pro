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

# Extracting Close Prices and Volume
try:
    close_prices = data['Close']
    volumes = data['Volume']
except KeyError:
    st.error("Data format error. Please ensure downloader.py uses group_by='column'.")
    st.stop()

# --- 1. MARKET BREADTH (The Safety Switch) ---
ema50_market = close_prices.ewm(span=50).mean()
breadth = (close_prices.iloc[-1] > ema50_market.iloc[-1]).mean() * 100

st.sidebar.title("🛡️ Market Environment")
if breadth > 50:
    st.sidebar.success(f"🟢 BULLISH BREADTH: {breadth:.1f}%")
    market_bonus = 1
elif breadth > 30:
    st.sidebar.warning(f"🟡 NEUTRAL BREADTH: {breadth:.1f}%")
    market_bonus = 0
else:
    st.sidebar.error(f"🔴 BEARISH BREADTH: {breadth:.1f}%")
    market_bonus = -2

# --- 2. SECTOR SENTIMENT ENGINE ---
st.sidebar.markdown("---")
st.sidebar.subheader("🔥 Top 3 Trending Sectors")
sector_returns = {}

# Calculate 1-month returns for sector ranking
for ticker in close_prices.columns:
    if ticker == "^NSEI" or len(close_prices[ticker].dropna()) < 25: continue
    clean_ticker = ticker.replace(".NS", "")
    try:
        sector = meta.loc[meta['SYMBOL'] == clean_ticker, 'SECTOR'].values[0]
        ret_1m = (close_prices[ticker].dropna().iloc[-1] - close_prices[ticker].dropna().iloc[-20]) / close_prices[ticker].dropna().iloc[-20]
        if sector not in sector_returns: sector_returns[sector] = []
        sector_returns[sector].append(ret_1m)
    except: continue

# Average out the returns and rank
sector_avg = {sec: np.mean(rets) for sec, rets in sector_returns.items()}
top_sectors = sorted(sector_avg, key=sector_avg.get, reverse=True)[:3]

for sec in top_sectors:
    st.sidebar.markdown(f"**- {sec}**")

# --- 3. SIGNAL GENERATION ENGINE ---
st.title("🎯 Quantum-Sentinel: 15-30 Day Swing Targets")
st.caption("Scoring incorporates Trend, Relative Strength, RSI Momentum, Volume Surge, and Sector Rotation.")

if st.button("🚀 Run Institutional Scan"):
    results = []
    
    # Benchmark Nifty
    if "^NSEI" in close_prices.columns:
        nifty = close_prices['^NSEI'].dropna()
        nifty_ret = (nifty.iloc[-1] - nifty.iloc[-60]) / nifty.iloc[-60] if len(nifty) > 60 else 0
    else:
        nifty_ret = 0

    progress_bar = st.progress(0, text="Analyzing 2,600+ Stocks for optimal setups...")
    total = len(close_prices.columns)

    for i, ticker in enumerate(close_prices.columns):
        if i % 50 == 0: progress_bar.progress(i / total)
        if ticker == "^NSEI": continue
        
        try:
            s_data = close_prices[ticker].dropna()
            v_data = volumes[ticker].dropna()
            if len(s_data) < 60: continue
            
            curr_price = round(s_data.iloc[-1], 2)
            
            # --- Technical Indicators ---
            # 1. EMAs
            ema20 = s_data.ewm(span=20).mean().iloc[-1]
            ema50_val = s_data.ewm(span=50).mean().iloc[-1]
            
            # 2. RSI (14)
            delta = s_data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean().iloc[-1]
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().iloc[-1]
            rs = gain / loss if loss != 0 else 0
            rsi = 100 - (100 / (1 + rs))
            
            # 3. Volume Surge (Is today's volume > 1.5x the 20-day avg?)
            avg_vol_20 = v_data.rolling(window=20).mean().iloc[-1]
            curr_vol = v_data.iloc[-1]
            vol_surge = curr_vol > (1.5 * avg_vol_20)
            
            # 4. Relative Strength vs Nifty
            stock_ret = (curr_price - s_data.iloc[-60]) / s_data.iloc[-60]
            is_beating_nifty = stock_ret > nifty_ret
            
            # Get Sector
            clean_ticker = ticker.replace(".NS", "")
            sector_match = meta.loc[meta['SYMBOL'] == clean_ticker, 'SECTOR'].values
            sector = sector_match[0] if len(sector_match) > 0 else "Unmapped"

            # --- THE SCORING ALGORITHM (Out of 10) ---
            score = 3 # Base score
            if curr_price > ema20 and ema20 > ema50_val: score += 2 # Perfect Trend
            if is_beating_nifty: score += 1                         # Alpha
            if 55 <= rsi <= 75: score += 1                          # Ideal Momentum Zone
            if vol_surge: score += 1                                # Institutional Buying
            if sector in top_sectors: score += 1                    # Sector Tailwinds
            score += market_bonus                                   # Market Environment
            
            score = max(1, min(10, int(score))) # Lock between 1 and 10
            
            # Filter for high probability only
            if score >= 7:
                # Math for 15-30 day trade
                target = round(curr_price * 1.15, 2)
                stop_loss = round(min(curr_price * 0.93, ema50_val), 2) # Use 7% OR the 50EMA (whichever is lower)
                upside_pct = round(((target - curr_price) / curr_price) * 100, 2)
                
                results.append({
                    "Stock": clean_ticker,
                    "Score": score,
                    "Sector": sector,
                    "Entry": curr_price,
                    "Target (15%)": target,
                    "Stop Loss": stop_loss,
                    "RSI": round(rsi, 1),
                    "Vol Surge": "Yes" if vol_surge else "No",
                    "Verdict": "🔥 HIGH PROBABILITY" if score >= 9 else "✅ BUY"
                })
        except: continue

    progress_bar.empty()
    
    if results:
        df_res = pd.DataFrame(results)
        
        # Sort so the absolute best (Top 10-15) are clearly on top
        df_res = df_res.sort_values(by=["Score", "RSI"], ascending=[False, False])
        
        st.subheader(f"📊 Top Opportunities ({len(df_res)} found)")
        
        # Highlight logic for the dataframe
        def highlight_score(val):
            if val == 10: return 'background-color: #004d00; color: white'
            elif val >= 8: return 'background-color: #008000; color: white'
            elif val == 7: return 'background-color: #556b2f; color: white'
            return ''
            
        st.dataframe(
            df_res.style.map(highlight_score, subset=['Score']), 
            use_container_width=True, 
            hide_index=True
        )
        
        st.info("💡 **Trade Execution Rule:** Never risk more than 2% of your total capital on a single trade. If a stock drops below the listed Stop Loss, exit immediately without emotion.")
    else:
        st.warning("No stocks passed the strict 7/10 threshold today. Protect your capital and wait for the market to improve.")
