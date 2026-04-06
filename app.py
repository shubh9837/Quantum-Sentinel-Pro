import streamlit as st
import pandas as pd
import numpy as np
import json
import os

st.set_page_config(page_title="Quantum-Sentinel Pro", layout="wide")

PORTFOLIO_FILE = "portfolio.json"

# --- PERSISTENCE ---
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

@st.cache_data
def load_all():
    try:
        data = pd.read_parquet("market_data.parquet", engine='pyarrow')
        meta = pd.read_csv("tickers_enriched.csv")
        return data, meta
    except: return None, None

data, meta = load_all()
if data is None:
    st.error("Market data not found. Please trigger the 'download' action in GitHub.")
    st.stop()

def clean_sym(s): return s.replace(".NS", "").replace("_", "&")
close_prices = data['Close']

# --- SIDEBAR: SWING TRADING FILTERS & CONTEXT ---
st.sidebar.title("🛡️ Institutional Risk Guard")
n_sym = "^NSEI"
n_close = close_prices[n_sym].dropna()
n_curr = n_close.iloc[-1]
n_ema50 = n_close.ewm(span=50).mean().iloc[-1]
market_trend = "BULLISH" if n_curr > n_ema50 else "BEARISH"

if market_trend == "BULLISH":
    st.sidebar.success(f"🟢 MARKET: {market_trend} (Nifty > 50-EMA)")
    market_bonus = 2
else:
    st.sidebar.error(f"🔴 MARKET: {market_trend} (Defensive Mode)")
    market_bonus = -2

st.sidebar.markdown("---")
st.sidebar.subheader("🎚️ Screener Filters")
min_rating = st.sidebar.slider("Min. Rating (Score)", 0, 10, 6)
selected_sectors = st.sidebar.multiselect("Filter by Sector", options=sorted(meta['SECTOR'].unique().tolist()))

# --- TAB SETUP ---
st.title("🎯 Quantum-Sentinel: Swing Predictor")
tab1, tab2 = st.tabs(["📊 Market Screener", "💼 My Portfolio"])

# --- TAB 1: RESTORED & ENHANCED SCREENER ---
with tab1:
    st.markdown("### 🚀 High-Probability Swing Signals")
    
    if st.button("🔍 Run Full Market Scan"):
        results = []
        n_60d_ret = (n_curr - n_close.iloc[-60]) / n_close.iloc[-60]
        prog = st.progress(0)
        all_ticks = close_prices.columns
        
        for i, t in enumerate(all_ticks):
            if t == n_sym: continue
            try:
                s_data = close_prices[t].dropna()
                if len(s_data) < 60: continue
                
                curr = s_data.iloc[-1]
                e20 = s_data.ewm(span=20).mean().iloc[-1]
                e50 = s_data.ewm(span=50).mean().iloc[-1]
                
                # Technical Indicators
                rel_strength = ((curr - s_data.iloc[-60])/s_data.iloc[-60]) - n_60d_ret
                vol = s_data.pct_change().std() * np.sqrt(20) # Monthly volatility
                
                # Logic: Rating (Score)
                score = 4 + market_bonus
                if curr > e20: score += 1
                if e20 > e50: score += 1
                if rel_strength > 0: score += 2
                
                rating = max(0, min(10, int(score)))
                
                # Calculation of Swing Parameters
                target_val = round(float(curr * (1 + vol)), 2)
                upside_pct = round(((target_val - curr) / curr) * 100, 1)
                stop_loss = round(float(curr * (1 - (vol * 0.7))), 2)
                rr_ratio = round((target_val - curr) / (curr - stop_loss), 1) if (curr - stop_loss) > 0 else 0
                
                clean_t = clean_sym(t)
                sector_val = meta.loc[meta['SYMBOL'] == clean_t, 'SECTOR'].values[0] if clean_t in meta['SYMBOL'].values else "Other"

                results.append({
                    "Stock": clean_t,
                    "Rating": rating,
                    "Sector": sector_val,
                    "Current Price": round(float(curr), 2),
                    "Target": target_val,
                    "Upside %": upside_pct,
                    "Chances (%)": rating * 10,
                    "Stop Loss": stop_loss,
                    "R:R": rr_ratio,
                    "Verdict": "🔥 Institutional Buy" if rating >= 8 else "✅ Momentum Play" if rating >= 6 else "↔️ Neutral"
                })
            except: continue
            if i % 100 == 0: prog.progress(i/len(all_ticks))
        
        raw_df = pd.DataFrame(results)
        # Apply Sidebar Filters
        filtered_df = raw_df[raw_df['Rating'] >= min_rating]
        if selected_sectors:
            filtered_df = filtered_df[filtered_df['Sector'].isin(selected_sectors)]
            
        st.session_state['res'] = filtered_df.sort_values(by="Rating", ascending=False)
        prog.empty()

    if 'res' in st.session_state:
        # Search Box
        search_q = st.text_input("🔍 Search specific stock in scan results:").upper()
        df_to_show = st.session_state['res']
        if search_q:
            df_to_show = df_to_show[df_to_show['Stock'].str.contains(search_q)]
            
        st.dataframe(df_to_show, use_container_width=True, hide_index=True)

# --- TAB 2: ROBUST PORTFOLIO MANAGER ---
with tab2:
    if 'portfolio' not in st.session_state: st.session_state['portfolio'] = load_portfolio()
    
    st.subheader("💼 Portfolio & Position Tracker")
    
    # 1. Entry Interface
    with st.expander("➕ Add / Edit Position", expanded=False):
        with st.form("port_form"):
            col_s, col_p, col_q = st.columns([2, 1, 1])
            all_syms = sorted(meta['SYMBOL'].astype(str).unique().tolist())
            s_sym = col_s.selectbox("Stock Symbol", [""] + all_syms)
            
            existing = st.session_state['portfolio'].get(s_sym, {"price": 0.0, "qty": 0})
            s_price = col_p.number_input("Avg Buy Price (₹)", value=float(existing['price']))
            s_qty = col_q.number_input("Quantity", value=int(existing['qty']), min_value=0)
            
            if st.form_submit_button("💾 Save to Portfolio"):
                if s_sym:
                    if s_qty == 0:
                        if s_sym in st.session_state['portfolio']: del st.session_state['portfolio'][s_sym]
                    else:
                        st.session_state['portfolio'][s_sym] = {"price": s_price, "qty": s_qty}
                    save_portfolio(st.session_state['portfolio'])
                    st.rerun()

    st.markdown("---")
    
    if st.session_state['portfolio']:
        port_data = []
        action_sell, action_buy = [], []
        main_df = st.session_state.get('res', pd.DataFrame())

        for sym, d in st.session_state['portfolio'].items():
            invested = d['price'] * d['qty']
            row = {"Stock": sym, "Avg Price": round(d['price'], 2), "Qty": d['qty'], "Invested": round(invested, 2)}
            
            if not main_df.empty and sym in main_df['Stock'].values:
                live = main_df[main_df['Stock'] == sym].iloc[0]
                curr_p = float(live['Current Price'])
                score = int(live['Rating'])
                curr_val = curr_p * d['qty']
                pnl = curr_val - invested
                pnl_pct = (pnl / invested * 100) if invested > 0 else 0
                
                # Verdict Logic for Portfolio
                if pnl_pct <= -7.0 or score <= 3: 
                    verdict =
                    
