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
    st.error("Data missing. Please run Downloader in GitHub Actions.")
    st.stop()

close_prices = data['Close']

# --- THE SAFETY SWITCH (MARKET BREADTH) ---
ema50 = close_prices.ewm(span=50).mean()
breadth = (close_prices.iloc[-1] > ema50.iloc[-1]).mean() * 100

st.sidebar.title("🛡️ Institutional Risk")
if breadth > 50:
    st.sidebar.success(f"🟢 BULLISH BREADTH: {breadth:.1f}%")
    market_bonus = 2
else:
    st.sidebar.error(f"🔴 BEARISH BREADTH: {breadth:.1f}%")
    market_bonus = -2 # Penalizes stocks because the market is falling

# --- ACTIONABLE SIGNAL ENGINE ---
st.title("🎯 Quantum-Sentinel: Trading Signals")
st.caption(f"Analyzing {len(close_prices.columns)} NSE Stocks | Mode: {'Aggressive' if breadth > 40 else 'Defensive'}")

if st.button("🚀 Generate High-Probability Signals"):
    results = []
    
    # Benchmark Calculation (Nifty)
    nifty = close_prices['^NSEI'].dropna()
    nifty_ret = (nifty.iloc[-1] - nifty.iloc[-60]) / nifty.iloc[-60]

    progress_bar = st.progress(0)
    total = len(close_prices.columns)

    for i, ticker in enumerate(close_prices.columns):
        if i % 100 == 0: progress_bar.progress(i / total)
        if ticker == "^NSEI": continue
        
        try:
            s_data = close_prices[ticker].dropna()
            if len(s_data) < 100: continue
            
            curr_price = round(s_data.iloc[-1], 2)
            
            # 1. Trend Logic (EMA 20/50)
            ema20 = s_data.ewm(span=20).mean().iloc[-1]
            ema50_val = s_data.ewm(span=50).mean().iloc[-1]
            
            # 2. Relative Strength (RS)
            stock_ret = (curr_price - s_data.iloc[-60]) / s_data.iloc[-60]
            is_beating_nifty = stock_ret > nifty_ret
            
            # --- SCORING ENGINE (Out of 10) ---
            score = 5 # Base Score
            if curr_price > ema20: score += 1
            if ema20 > ema50_val: score += 1
            if is_beating_nifty: score += 2
            score += market_bonus
            
            # Clamp score between 1 and 10
            score = max(1, min(10, score))
            
            # --- ONLY SHOW HIGH PROBABILITY (Score 7+) ---
            if score >= 7 or (is_beating_nifty and breadth < 20):
                # Calculate Targets and SL (Standard Swing Math)
                # Entry = Current Price
                # Target = 15% (Standard Swing)
                # Stop Loss = 7% or the 50 EMA
                target = round(curr_price * 1.15, 2)
                stop_loss = round(min(curr_price * 0.93, ema50_val), 2)
                risk_reward = round((target - curr_price) / (curr_price - stop_loss), 1)

                clean_ticker = ticker.replace(".NS", "")
                sector_match = meta.loc[meta['SYMBOL'] == clean_ticker, 'SECTOR'].values
                sector = sector_match[0] if len(sector_match) > 0 else "Unmapped"

                results.append({
                    "Stock": clean_ticker,
                    "Rating": f"{score}/10",
                    "Sector": sector,
                    "Entry Point": curr_price,
                    "Exit (Target)": target,
                    "Stop Loss": stop_loss,
                    "R:R Ratio": risk_reward,
                    "Holding Period": "2-4 Weeks (Swing)",
                    "Verdict": "🔥 Institutional Buy" if score >= 8 else "✅ Momentum Play"
                })
        except: continue

    progress_bar.empty()
    
    if results:
        df_res = pd.DataFrame(results).sort_values(by="Rating", ascending=False)
        st.dataframe(df_res, use_container_width=True, hide_index=True)
        
        st.info("💡 **Strategy Note:** Focus on stocks where the R:R Ratio is above 2.0. If the Market Breadth is RED, keep your position sizes small.")
    else:
        st.warning("No high-probability opportunities found in this bearish environment. Cash is a position!")
