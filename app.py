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
    st.error("Data missing. Please run the 'download' task in GitHub Actions.")
    st.stop()

# Helper to fix symbols
def clean_sym(s): return s.replace(".NS", "").replace("_", "&")

close_prices = data['Close']
volumes = data['Volume']

# --- 1. SECTOR & MARKET ANALYSIS (Sidebar) ---
st.sidebar.title("🛡️ Market Intelligence")

# Market Breadth (Nifty 500 vs 50 EMA)
ema50_m = close_prices.ewm(span=50).mean()
breadth = (close_prices.iloc[-1] > ema50_m.iloc[-1]).mean() * 100
market_bias = 1 if breadth > 50 else (-2 if breadth < 30 else 0)

if breadth > 50: st.sidebar.success(f"🟢 MARKET: BULLISH ({breadth:.1f}%)")
else: st.sidebar.error(f"🔴 MARKET: BEARISH ({breadth:.1f}%)")

# Sector Sentiment Engine
st.sidebar.markdown("---")
st.sidebar.subheader("🔥 Top 3 Trending Sectors")
sector_rets = {}
for ticker in close_prices.columns:
    if ticker == "^NSEI": continue
    try:
        clean = clean_sym(ticker)
        sec = meta.loc[meta['SYMBOL'] == clean, 'SECTOR'].values[0]
        ret = (close_prices[ticker].iloc[-1] - close_prices[ticker].iloc[-20]) / close_prices[ticker].iloc[-20]
        if sec not in sector_rets: sector_rets[sec] = []
        sector_rets[sec].append(ret)
    except: continue

avg_sec_ret = {k: np.mean(v) for k, v in sector_rets.items()}
top_sectors = sorted(avg_sec_ret, key=avg_sec_ret.get, reverse=True)[:3]
for s in top_sectors: st.sidebar.markdown(f"**- {s}**")

# --- 2. PREDICTION ENGINE ---
st.title("🎯 Quantum-Sentinel: 30-Day Pro Predictor")
st.caption("Predicting Directional Movement and Expected Volatility for 2,200+ Stocks.")

if st.button("🚀 Run Institutional Prediction Scan"):
    results = []
    nifty = close_prices['^NSEI'].dropna()
    n_ret = (nifty.iloc[-1] - nifty.iloc[-60])/nifty.iloc[-60] if len(nifty)>60 else 0

    prog = st.progress(0)
    all_ticks = close_prices.columns
    
    for i, t in enumerate(all_ticks):
        if t == "^NSEI": continue
        try:
            s_data = close_prices[t].dropna()
            v_data = volumes[t].dropna()
            if len(s_data) < 40: continue
            
            curr = s_data.iloc[-1]
            e20 = s_data.ewm(span=20).mean().iloc[-1]
            e50 = s_data.ewm(span=50).mean().iloc[-1]
            
            # Momentum (RSI)
            diff = s_data.diff()
            g = diff.where(diff > 0, 0).rolling(14).mean().iloc[-1]
            l = -diff.where(diff < 0, 0).rolling(14).mean().iloc[-1]
            rsi = 100 - (100/(1 + g/l)) if l > 0 else 50
            
            # Volatility (Expected Move)
            # Use 20-day Std Dev to predict the 30-day "swing range"
            daily_std = s_data.pct_change().std()
            expected_move_pct = daily_std * np.sqrt(20) * 100 # 20 trading days in a month
            
            # Scoring Logic (0-10)
            score = 4 + market_bias
            if curr > e20: score += 1
            if e20 > e50: score += 1
            if rsi > 55: score += 1
            if rsi > 70: score -= 1 # Penalty for overbought
            if v_data.iloc[-1] > v_data.rolling(20).mean().iloc[-1] * 1.5: score += 2 # Volume Surge
            
            clean_t = clean_sym(t)
            sec = meta.loc[meta['SYMBOL'] == clean_t, 'SECTOR'].values[0] if clean_t in meta['SYMBOL'].values else "Other"
            if sec in top_sectors: score += 1
            
            score = max(0, min(10, int(score)))

            # Directional Prediction
            if score >= 7: direction = "⬆️ BULLISH"
            elif score <= 3: direction = "⬇️ BEARISH"
            else: direction = "↔️ SIDEWAYS"

            results.append({
                "Stock": clean_t,
                "Score": score,
                "Direction": direction,
                "Price": round(float(curr), 1),
                "Exp. Move (%)": f"±{round(float(expected_move_pct), 1)}%",
                "Target (High)": round(float(curr * (1 + expected_move_pct/100)), 1),
                "Target (Low)": round(float(curr * (1 - expected_move_pct/100)), 1),
                "RSI": round(float(rsi), 1),
                "Sector": sec
            })
        except: continue
        if i % 100 == 0: prog.progress(i/len(all_ticks))
    
    st.session_state['res'] = pd.DataFrame(results).sort_values(by="Score", ascending=False)
    prog.empty()

# --- 3. SEARCH & RESULTS ---
if 'res' in st.session_state:
    df = st.session_state['res']
    
    search = st.text_input("🔍 Check Specific Stock Prediction").upper()
    if search:
        s_res = df[df['Stock'].str.contains(search)]
        if not s_res.empty:
            st.write(f"### Prediction for {search}")
            st.dataframe(s_res, hide_index=True)
        else: st.warning("Stock not found.")

    st.subheader("📊 Full Market Prediction Rankings")
    
    # Apply styling
    def color_dir(val):
        if "BULLISH" in str(val): return 'color: #00ff00; font-weight: bold'
        if "BEARISH" in str(val): return 'color: #ff4b4b; font-weight: bold'
        return ''

    st.dataframe(df.style.map(color_dir, subset=['Direction']), use_container_width=True, hide_index=True)
