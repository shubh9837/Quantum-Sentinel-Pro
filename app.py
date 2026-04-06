import streamlit as st
import pandas as pd
import numpy as np
import json
import os

st.set_page_config(page_title="Quantum-Sentinel Pro", layout="wide")

# --- PORTFOLIO STORAGE FILE ---
PORTFOLIO_FILE = "portfolio.json"

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r") as f:
                return json.load(f)
        except: return []
    return []

def save_portfolio(portfolio):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f)

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

# --- 1. BENCHMARK & SECTOR ANALYSIS (Sidebar) ---
st.sidebar.title("🛡️ Institutional Intelligence")

nifty_symbol = "^NSEI"
if nifty_symbol in close_prices.columns:
    nifty_close = close_prices[nifty_symbol].dropna()
    nifty_ema50 = nifty_close.ewm(span=50).mean().iloc[-1]
    nifty_curr = nifty_close.iloc[-1]
    
    if nifty_curr > nifty_ema50:
        st.sidebar.success(f"🟢 NIFTY 50: Above 50-EMA (Bullish)")
        nifty_bias = 1
    else:
        st.sidebar.error(f"🔴 NIFTY 50: Below 50-EMA (Bearish)")
        nifty_bias = -2
else:
    nifty_bias = 0
    nifty_curr = 1
    nifty_close = pd.Series([1]*60)

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

# --- INITIALIZE TABS ---
st.title("🎯 Quantum-Sentinel: Pro Predictor")
tab1, tab2 = st.tabs(["📊 Market Screener", "💼 My Portfolio"])

# --- TAB 1: MARKET SCREENER ---
with tab1:
    st.caption("Benchmark-aware scoring with 15-30 day directional forecasting.")

    if st.button("🚀 Run Institutional Prediction Scan"):
        results = []
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
                
                stock_60d_ret = (curr - s_data.iloc[-60]) / s_data.iloc[-60]
                outperforming_nifty = stock_60d_ret > nifty_60d_ret
                
                diff = s_data.diff()
                g = diff.where(diff > 0, 0).rolling(14).mean().iloc[-1]
                l = -diff.where(diff < 0, 0).rolling(14).mean().iloc[-1]
                rsi = 100 - (100/(1 + g/l)) if l > 0 else 50
                
                expected_vol = s_data.pct_change().std() * np.sqrt(20) * 100
                
                # 0-10 SCORING LOGIC
                score = 4 + nifty_bias
                if curr > e20: score += 1
                if e20 > e50: score += 1
                if outperforming_nifty: score += 2 
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

    if 'res' in st.session_state:
        df = st.session_state['res']
        
        search = st.text_input("🔍 Search Stock Detail (e.g. INFOSYS)").upper()
        if search:
            st.table(df[df['Stock'].str.contains(search)])

        st.subheader("📊 Ranked Market Predictions")
        st.dataframe(df, use_container_width=True, hide_index=True)

# --- TAB 2: MY PORTFOLIO ---
with tab2:
    st.subheader("💼 Active Portfolio Monitor")
    
    # Load session state portfolio
    if 'portfolio' not in st.session_state:
        st.session_state['portfolio'] = load_portfolio()
        
    col1, col2 = st.columns(2)
    with col1:
        new_stock = st.text_input("Add Stock Symbol (e.g. RELIANCE)").upper().strip()
        if st.button("➕ Add to Portfolio"):
            if new_stock and new_stock not in st.session_state['portfolio']:
                st.session_state['portfolio'].append(new_stock)
                save_portfolio(st.session_state['portfolio'])
                st.success(f"Added {new_stock} to your portfolio.")
                st.rerun()
                
    with col2:
        remove_stock = st.selectbox("Remove Stock", [""] + st.session_state['portfolio'])
        if st.button("🗑️ Remove"):
            if remove_stock in st.session_state['portfolio']:
                st.session_state['portfolio'].remove(remove_stock)
                save_portfolio(st.session_state['portfolio'])
                st.warning(f"Removed {remove_stock} from your portfolio.")
                st.rerun()

    st.markdown("---")
    
    if st.session_state['portfolio']:
        if 'res' in st.session_state:
            df = st.session_state['res']
            # Filter main results for only the stocks in the portfolio
            port_df = df[df['Stock'].isin(st.session_state['portfolio'])].copy()
            
            def get_action(score):
                if score >= 7: return "🟢 ADD MORE (BUY)"
                elif score >= 4: return "🟡 HOLD TIGHT"
                else: return "🔴 SELL / TRIM RISK"
                
            if not port_df.empty:
                port_df.insert(2, 'Action Required', port_df['Score'].apply(get_action))
                st.dataframe(port_df, use_container_width=True, hide_index=True)
            else:
                st.warning("Ensure the symbols you added match the exact NSE tickers (e.g., INFY, TCS, HDFCBANK).")
        else:
            st.info("⚠️ Please click **'🚀 Run Institutional Prediction Scan'** in the Market Screener tab first to load the live analysis for your portfolio.")
    else:
        st.caption("Your portfolio is currently empty. Add stocks above to begin monitoring.")
        
