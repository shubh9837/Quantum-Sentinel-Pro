import streamlit as st
import pandas as pd
import numpy as np
import json
import os

st.set_page_config(page_title="Quantum-Sentinel Pro", layout="wide")

PORTFOLIO_FILE = "portfolio.json"

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
    st.error("Data missing. Please run 'download' in GitHub Actions.")
    st.stop()

def clean_sym(s): return s.replace(".NS", "").replace("_", "&")
close_prices = data['Close']

# --- SIDEBAR: MARKET CONTEXT ---
st.sidebar.title("🛡️ Institutional Intelligence")
nifty_symbol = "^NSEI"
if nifty_symbol in close_prices.columns:
    n_close = close_prices[nifty_symbol].dropna()
    n_curr = n_close.iloc[-1]
    n_ema50 = n_close.ewm(span=50).mean().iloc[-1]
    n_bias = 1 if n_curr > n_ema50 else -2
    st.sidebar.success("🟢 NIFTY: Bullish") if n_bias == 1 else st.sidebar.error("🔴 NIFTY: Bearish")
else:
    n_bias, n_curr, n_close = 0, 1, pd.Series([1]*60)

st.title("🎯 Quantum-Sentinel: Pro Predictor")
tab1, tab2 = st.tabs(["📊 Market Screener", "💼 My Portfolio"])

# --- TAB 1: SCREENER ---
with tab1:
    if st.button("🚀 Run Institutional Prediction Scan"):
        results = []
        nifty_60d_ret = (n_curr - n_close.iloc[-60]) / n_close.iloc[-60]
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
                rsi = 100 - (100/(1 + (s_data.diff().where(s_data.diff()>0,0).rolling(14).mean().iloc[-1] / -s_data.diff().where(s_data.diff()<0,0).rolling(14).mean().iloc[-1])))
                vol = s_data.pct_change().std() * np.sqrt(20) * 100
                score = max(0, min(10, int(4 + n_bias + (1 if curr > e20 else 0) + (2 if ((curr-s_data.iloc[-60])/s_data.iloc[-60]) > nifty_60d_ret else 0))))
                results.append({"Stock": clean_sym(t), "Score": score, "Price": round(float(curr), 1), "Exp. Move": f"±{round(float(vol), 1)}%", "Target": round(float(curr*(1+vol/100)), 1)})
            except: continue
            if i % 100 == 0: prog.progress(i/len(all_ticks))
        st.session_state['res'] = pd.DataFrame(results).sort_values(by="Score", ascending=False)
        prog.empty()
    if 'res' in st.session_state: st.dataframe(st.session_state['res'], use_container_width=True, hide_index=True)

# --- TAB 2: MY PORTFOLIO (WITH EDITING & SUGGESTIONS) ---
with tab2:
    if 'portfolio' not in st.session_state: st.session_state['portfolio'] = load_portfolio()
    
    st.subheader("💼 Portfolio Command Center")
    col_a, col_b = st.columns([2, 1])

    with col_a:
        with st.form("portfolio_form"):
            st.write("**Add or Edit Position**")
            all_symbols = sorted(meta['SYMBOL'].astype(str).unique().tolist())
            selected_sym = st.selectbox("Select/Search Stock", [""] + all_symbols, help="Start typing to see suggestions")
            
            # Auto-fill values if editing
            existing = st.session_state['portfolio'].get(selected_sym, {"price": 0.0, "qty": 0})
            
            c1, c2 = st.columns(2)
            b_price = c1.number_input("Average Buy Price (₹)", value=float(existing['price']), step=0.1, format="%.1f")
            b_qty = c2.number_input("Quantity", value=int(existing['qty']), min_value=0, step=1)
            
            if st.form_submit_button("💾 Save Record"):
                if selected_sym:
                    st.session_state['portfolio'][selected_sym] = {"price": b_price, "qty": b_qty}
                    save_portfolio(st.session_state['portfolio'])
                    st.success(f"Record for {selected_sym} updated!")
                    st.rerun()

    with col_b:
        st.write("**Quick Delete**")
        to_del = st.selectbox("Select to Remove", [""] + list(st.session_state['portfolio'].keys()))
        if st.button("🗑️ Delete Position") and to_del:
            del st.session_state['portfolio'][to_del]
            save_portfolio(st.session_state['portfolio'])
            st.rerun()

    st.markdown("---")
    if st.session_state['portfolio']:
        st.write("### Active Positions")
        port_list = []
        main_df = st.session_state.get('res', pd.DataFrame())

        for sym, d in st.session_state['portfolio'].items():
            row = {"Stock": sym, "Qty": d['qty'], "Avg Buy": round(d['price'], 1)}
            
            # Link live data if scan was run
            if not main_df.empty and sym in main_df['Stock'].values:
                live = main_df[main_df['Stock'] == sym].iloc[0]
                live_p = float(live['Price'])
                score = int(live['Score'])
                pnl = ((live_p - d['price']) / d['price'] * 100) if d['price'] > 0 else 0
                
                # Professional Advice Logic
                if pnl <= -7: advice, hold = "🚨 CUT LOSS", "Exit Now"
                elif score >= 8: advice, hold = "🔥 ACCUMULATE", "30 Days"
                elif score >= 5: advice, hold = "✅ HOLD", "15 Days"
                else: advice, hold = "⚠️ WEAK (Exit on bounce)", "Short Term"
                
                row.update({"Live": live_p, "P&L %": round(pnl, 1), "Score": f"{score}/10", "Advice": advice, "Duration": hold})
            else:
                row.update({"Live": "Scan Required", "P&L %": "-", "Score": "-", "Advice": "Click 'Run Scan' in Tab 1", "Duration": "-"})
            port_list.append(row)

        st.dataframe(pd.DataFrame(port_list), use_container_width=True, hide_index=True)
