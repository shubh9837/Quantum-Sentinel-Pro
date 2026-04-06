import streamlit as st
import pandas as pd
import numpy as np
import json
import os

st.set_page_config(page_title="Quantum-Sentinel Pro", layout="wide")

PORTFOLIO_FILE = "portfolio.json"

# --- PERSISTENCE: LOAD AT START ---
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

# Initialize portfolio in session state immediately
if 'portfolio' not in st.session_state:
    st.session_state['portfolio'] = load_portfolio()

@st.cache_data
def load_all():
    try:
        data = pd.read_parquet("market_data.parquet", engine='pyarrow')
        meta = pd.read_csv("tickers_enriched.csv")
        return data, meta
    except: return None, None

data, meta = load_all()
if data is None:
    st.warning("📡 Connecting to Market Data... Please run the 'download' task if this persists.")
    st.stop()

def clean_sym(s): return s.replace(".NS", "").replace("_", "&")
close_prices = data['Close']

# --- MARKET CONTEXT ---
n_symbol = "^NSEI"
n_close = close_prices[n_symbol].dropna()
n_curr = n_close.iloc[-1]
n_ema50 = n_close.ewm(span=50).mean().iloc[-1]
n_bias = 1 if n_curr > n_ema50 else -2

st.title("🎯 Quantum-Sentinel: Pro Predictor")
tab1, tab2 = st.tabs(["📊 Market Screener", "💼 My Portfolio"])

# --- TAB 1: SCREENER (NO CHANGES) ---
with tab1:
    if st.button("🚀 Run Institutional Prediction Scan"):
        results = []
        n_60d_ret = (n_curr - n_close.iloc[-60]) / n_close.iloc[-60]
        prog = st.progress(0)
        all_ticks = close_prices.columns
        for i, t in enumerate(all_ticks):
            if t == n_symbol: continue
            try:
                s_data = close_prices[t].dropna()
                if len(s_data) < 60: continue
                curr = s_data.iloc[-1]
                e20 = s_data.ewm(span=20).mean().iloc[-1]
                rsi = 100 - (100/(1 + (s_data.diff().where(s_data.diff()>0,0).rolling(14).mean().iloc[-1] / -s_data.diff().where(s_data.diff()<0,0).rolling(14).mean().iloc[-1])))
                vol = s_data.pct_change().std() * np.sqrt(20) * 100
                score = max(0, min(10, int(4 + n_bias + (1 if curr > e20 else 0) + (2 if ((curr - s_data.iloc[-60])/s_data.iloc[-60]) > n_60d_ret else 0))))
                results.append({"Stock": clean_sym(t), "Score": score, "Price": round(float(curr), 1), "Target": round(float(curr*(1+vol/100)), 1), "RSI": round(float(rsi), 1)})
            except: continue
            if i % 100 == 0: prog.progress(i/len(all_ticks))
        st.session_state['res'] = pd.DataFrame(results).sort_values(by="Score", ascending=False)
        prog.empty()
    if 'res' in st.session_state: st.dataframe(st.session_state['res'], use_container_width=True, hide_index=True)

# --- TAB 2: MY PORTFOLIO (FIXED ONE-CLICK ADD) ---
with tab2:
    st.subheader("💼 Position Management")
    
    # Input fields outside a form to prevent "two-click" lag
    col_s, col_p, col_q, col_btn = st.columns([2, 1, 1, 1])
    all_syms = sorted(meta['SYMBOL'].astype(str).unique().tolist())
    
    s_sym = col_s.selectbox("Stock", [""] + all_syms, key="sym_input")
    existing = st.session_state['portfolio'].get(s_sym, {"price": 0.0, "qty": 0})
    s_price = col_p.number_input("Avg Price", value=float(existing['price']), step=0.1, key="price_input")
    s_qty = col_q.number_input("Qty", value=int(existing['qty']), min_value=0, key="qty_input")
    
    if col_btn.button("💾 Save Record", use_container_width=True):
        if s_sym:
            if s_qty == 0:
                if s_sym in st.session_state['portfolio']: del st.session_state['portfolio'][s_sym]
            else:
                st.session_state['portfolio'][s_sym] = {"price": s_price, "qty": s_qty}
            save_portfolio(st.session_state['portfolio'])
            st.rerun()

    st.markdown("---")
    
    if st.session_state['portfolio']:
        port_list = []
        actionables = {"SELL": [], "ADD": []}
        main_df = st.session_state.get('res', pd.DataFrame())

        for sym, d in st.session_state['portfolio'].items():
            invested = d['price'] * d['qty']
            row = {"Stock": sym, "Qty": d['qty'], "Avg Price": round(d['price'], 2), "Invested": round(invested, 2)}
            
            if not main_df.empty and sym in main_df['Stock'].values:
                live = main_df[main_df['Stock'] == sym].iloc[0]
                live_p, score = float(live['Price']), int(live['Score'])
                current_val = live_p * d['qty']
                unrealized = current_val - invested
                pnl_pct = (unrealized / invested * 100) if invested > 0 else 0
                
                if pnl_pct <= -7.0 or score <= 3: 
                    verdict, color = "🔴 SELL", actionables["SELL"].append(sym)
                elif score >= 8: 
                    verdict, color = "🔵 BUY/ADD", actionables["ADD"].append(sym)
                else: verdict = "🟡 HOLD"
                
                row.update({"Current": round(current_val, 2), "Profit/Loss": round(unrealized, 2), "P&L %": round(pnl_pct, 2), "Target (30D)": live['Target'], "Verdict": verdict})
            else:
                row.update({"Current": 0.0, "Profit/Loss": 0.0, "P&L %": 0.0, "Target (30D)": "Run Scan", "Verdict": "Scan Required"})
            port_list.append(row)

        df_p = pd.DataFrame(port_list)
        if not df_p.empty and "Current" in df_p.columns:
            total_inv = df_p["Invested"].sum()
            total_cur = df_p["Current"].sum()
            total_pnl = total_cur - total_inv
            total_pct = (total_pnl / total_inv * 100) if total_inv > 0 else 0
            
            # Add Total Row
            total_row = pd.DataFrame([{"Stock": "TOTAL", "Invested": total_inv, "Current": total_cur, "Profit/Loss": total_pnl, "P&L %": round(total_pct, 2)}])
            df_display = pd.concat([df_p, total_row], ignore_index=True)
            st.dataframe(df_display, use_container_width=True, hide_index=True)

        # --- ACTIONABLE SUMMARY ---
        st.subheader("📝 Tomorrow's Game Plan")
        c1, c2 = st.columns(2)
        with c1:
            st.info("**⚠️ Portfolio Adjustments**")
            if actionables["SELL"]: st.error(f"Exit immediately: {', '.join(actionables['SELL'])}")
            if actionables["ADD"]: st.success(f"Consider adding: {', '.join(actionables['ADD'])}")
            if not actionables["SELL"] and not actionables["ADD"]: st.write("No major adjustments needed for tomorrow.")
        with c2:
            st.success("**🚀 Top Swing Candidates**")
            if not main_df.empty:
                candidates = main_df[~main_df['Stock'].isin(st.session_state['portfolio'].keys())].head(3)
                for _, r in candidates.iterrows(): st.write(f"🌟 **{r['Stock']}** (Target: ₹{r['Target']})")
                    
