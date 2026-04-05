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
    st.error("Data missing. Please run 'download' in GitHub Actions.")
    st.stop()

def clean_sym(s): return s.replace(".NS", "").replace("_", "&")

close_prices = data['Close']
volumes = data['Volume']

# --- 1. BENCHMARK & SECTOR ANALYSIS ---
st.sidebar.title("🛡️ Institutional Intelligence")

# NIFTY 50 Trend Check
nifty_symbol = "^NSEI"
nifty_close = close_prices[nifty_symbol].dropna()
nifty_ema50 = nifty_close.ewm(span=50).mean().iloc[-1]
nifty_curr = nifty_close.iloc[-1]

if nifty_curr > nifty_ema50:
    st.sidebar.success(f"🟢 NIFTY 50: Above 50-EMA (Bullish)")
    nifty_bias = 1
else:
    st.sidebar.error(f"🔴 NIFTY 50: Below 50-EMA (Bearish)")
    nifty_bias = -2

# Sector Sentiment (Top 3)
sector_rets = {}
for ticker in close_prices.columns:
    if ticker == nifty_symbol: continue
    try:
        clean = clean_sym(ticker)
        sec = meta.loc[meta['SYMBOL'] == clean, 'SECTOR'].values[0]
        ret = (close_prices[ticker].iloc[-1] - close_prices[ticker].iloc[-20]) / close_prices[ticker].iloc[-20]
        if sec not in sector_rets: sector_rets[sec] = []
        sector_rets[sec].append(ret)
    except: continue

avg_sec_ret = {k: np.mean(v) for k, v in sector_rets.items()}
top_sectors = sorted(avg_sec_ret, key=avg_sec_ret.get, reverse=True)[:3]

st.sidebar.markdown("---")
st.sidebar.subheader("🔥 Top 3 Sectors")
for s in top_sectors: st.sidebar.markdown(f"**- {s}**")

# --- 2. PREDICTION ENGINE ---
st.title("🎯 Quantum-Sentinel: Pro Predictor")
st.caption("Benchmark-aware scoring with 15-30 day directional forecasting.")

if st.button("🚀 Run Institutional Prediction Scan"):
    results = []
    # Nifty 60-day return for Relative Strength comparison
    nifty_60d_ret = (nifty_curr - nifty_close.iloc[-60]) / nifty_close.iloc[-60]

    prog = st.progress(0)
    all_ticks = close_prices.columns
    
    for i, t in enumerate(all_ticks):
        if t == nifty_symbol: continue
        try:
            s_data = close_prices[t].dropna()
            if len(s_data) < 60: continue
            
            curr = s_data.iloc[-1]
            e20 = s_data.ewm(span=20).mean().iloc[-1]
            e50 = s_data.ewm(span=50).mean().iloc[-1]
            
            # Relative Strength vs NIFTY 50
            stock_60d_ret = (curr - s_data.iloc[-60]) / s_data.iloc[-60]
            outperforming_nifty = stock_60d_ret > nifty_60d_ret
            
            # Momentum (RSI)
            diff = s_data.diff()
            g = diff.where(diff > 0, 0).rolling(14).mean().iloc[-1]
            l = -diff.where(diff < 0, 0).rolling(14).mean().iloc[-1]
            rsi = 100 - (100/(1 + g/l)) if l > 0 else 50
            
            # Volatility-Based Expected Move (1-Month Std Dev)
            expected_vol = s_data.pct_change().std() * np.sqrt(20) * 100
            
            # 0-10 SCORING LOGIC
            score = 4 + nifty_bias
            if curr > e20: score += 1
            if e20 > e50: score += 1
            if outperforming_nifty: score += 2  # Big reward for beating the market
            if 50 < rsi < 70: score += 1
            if clean_sym(t) in [clean_sym(x) for x in top_sectors]: score += 1
            
            score = max(0, min(10, int(score)))

            results.append({
                "Stock": clean_sym(t),
                "Score": score,
                "Prediction": "⬆️ BULLISH" if score >= 7 else ("⬇️ BEARISH" if score <= 3 else "↔️ NEUTRAL"),
                "Price": round(float(curr), 1),
                "Exp. Move": f"±{round(float(expected_vol), 1)}%",
                "30D High Target": round(float(curr * (1 + expected_vol/100)), 1),
                "30D Low Target": round(float(curr * (1 - expected_vol/100)), 1),
                "RSI": round(float(rsi), 1)
            })
        except: continue
        if i % 100 == 0: prog.progress(i/len(all_ticks))
    
    st.session_state['res'] = pd.DataFrame(results).sort_values(by="Score", ascending=False)
    prog.empty()

# --- 3. SEARCH & TABLE ---
if 'res' in st.session_state:
    df = st.session_state['res']
    
    search = st.text_input("🔍 Search Stock Detail (e.g. INFOSYS)").upper()
    if search:
        st.table(df[df['Stock'].str.contains(search)])

    st.subheader("📊 Ranked Market Predictions")
    st.dataframe(df, use_container_width=True, hide_index=True)
