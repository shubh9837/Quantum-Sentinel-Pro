import streamlit as st
import pandas as pd
import numpy as np
import json
import os

st.set_page_config(page_title="Quantum-Sentinel Pro", layout="wide")

PORTFOLIO_FILE = "portfolio.json"

# --- PERSISTENCE HELPERS ---
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
    st.error("Data missing. Please run 'download' task.")
    st.stop()

def clean_sym(s): return s.replace(".NS", "").replace("_", "&")
close_prices = data['Close']

# --- SIDEBAR: BENCHMARK CONTEXT ---
st.sidebar.title("🛡️ Institutional Intelligence")
n_symbol = "^NSEI"
n_close = close_prices[n_symbol].dropna()
n_curr = n_close.iloc[-1]
n_ema50 = n_close.ewm(span=50).mean().iloc[-1]
n_bias = 1 if n_curr > n_ema50 else -2
st.sidebar.success(f"🟢 NIFTY: Bullish") if n_bias == 1 else st.sidebar.error(f"🔴 NIFTY: Bearish")

# --- TAB SETUP ---
st.title("🎯 Quantum-Sentinel: Pro Predictor")
tab1, tab2 = st.tabs(["📊 Market Screener", "💼 My Portfolio"])

# --- TAB 1: RESTORED 10/10 SCREENER ---
with tab1:
    st.caption("Standard Deviation Volatility & Nifty-Relative Momentum")
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
                
                # Scoring Logic
                score = 4 + n_bias
                if curr > e20: score += 1
                if ((curr - s_data.iloc[-60])/s_data.iloc[-60]) > n_60d_ret: score += 2
                if 50 < rsi < 70: score += 1
                
                score = max(0, min(10, int(score)))
                results.append({"Stock": clean_sym(t), "Score": score, "Price": round(float(curr), 1), "Exp. Move": f"±{round(float(vol), 1)}%", "Target": round(float(curr*(1+vol/100)), 1), "RSI": round(float(rsi), 1)})
            except: continue
            if i % 100 == 0: prog.progress(i/len(all_ticks))
        st.session_state['res'] = pd.DataFrame(results).sort_values(by="Score", ascending=False)
        prog.empty()
    if 'res' in st.session_state: st.dataframe(st.session_state['res'], use_container_width=True, hide_index=True)

# --- TAB 2: MY PORTFOLIO (NEW CALCULATIONS & TOTALS) ---
with tab2:
    if 'portfolio' not in st.session_state: st.session_state['portfolio'] = load_portfolio()
    
    st.subheader("💼 Position Management")
    with st.form("port_form"):
        col_s, col_p, col_q = st.columns([2, 1, 1])
        all_syms = sorted(meta['SYMBOL'].astype(str).unique().tolist())
        s_sym = col_s.selectbox("Stock", [""] + all_syms)
        
        # Auto-load existing details for editing
        existing = st.session_state['portfolio'].get(s_sym, {"price": 0.0, "qty": 0})
        s_price = col_p.number_input("Avg Price", value=float(existing['price']), step=0.1)
        s_qty = col_q.number_input("Qty", value=int(existing['qty']), min_value=0)
        
        if st.form_submit_button("💾 Save / Update Record"):
            if s_sym:
                if s_qty == 0: # Remove if qty set to 0
                    if s_sym in st.session_state['portfolio']: del st.session_state['portfolio'][s_sym]
                else:
                    st.session_state['portfolio'][s_sym] = {"price": s_price, "qty": s_qty}
                save_portfolio(st.session_state['portfolio'])
                st.rerun()

    st.markdown("---")
    
    if st.session_state['portfolio']:
        port_list = []
        actionables = {"SELL": [], "ADD": [], "NEW": []}
        main_df = st.session_state.get('res', pd.DataFrame())

        for sym, d in st.session_state['portfolio'].items():
            invested = d['price'] * d['qty']
            row = {"Stock": sym, "Qty": d['qty'], "Avg Price": round(d['price'], 1), "Invested": round(invested, 1)}
            
            if not main_df.empty and sym in main_df['Stock'].values:
                live = main_df[main_df['Stock'] == sym].iloc[0]
                live_p = float(live['Price'])
                score = int(live['Score'])
                current_val = live_p * d['qty']
                unrealized = current_val - invested
                pnl_pct = (unrealized / invested * 100) if invested > 0 else 0
                
                # Verdict Logic
                if pnl_pct <= -7.0 or score <= 3: 
                    verdict = "🔴 SELL"
                    actionables["SELL"].append(sym)
                elif score >= 8: 
                    verdict = "🔵 BUY/ADD"
                    actionables["ADD"].append(sym)
                else: verdict = "🟡 HOLD"
                
                row.update({
                    "Current": round(current_val, 1),
                    "Profit/Loss": round(unrealized, 1),
                    "P&L %": round(pnl_pct, 1),
                    "Target (30D)": live['Target'],
                    "Score": score,
                    "Verdict": verdict
                })
            else:
                row.update({"Current": 0.0, "Profit/Loss": 0.0, "P&L %": 0.0, "Target (30D)": "-", "Score": "-", "Verdict": "Scan Market First"})
            port_list.append(row)

        df_port = pd.DataFrame(port_list)
        
        # --- TOTAL ROW CALCULATION ---
        if not df_port.empty and "Current" in df_port.columns:
            totals = {
                "Stock": "TOTAL",
                "Invested": df_port["Invested"].sum(),
                "Current": df_port["Current"].sum(),
                "Profit/Loss": df_port["Profit/Loss"].sum(),
                "P&L %": round((df_port["Profit/Loss"].sum() / df_port["Invested"].sum() * 100), 1) if df_port["Invested"].sum() > 0 else 0
            }
            df_port = pd.concat([df_port, pd.DataFrame([totals])], ignore_index=True)

        st.write("### Active Portfolio Table")
        st.dataframe(df_port, use_container_width=True, hide_index=True)

        # --- NEXT TRADING SESSION SUMMARY ---
        st.markdown("---")
        st.subheader("📝 Next Trading Session: Actionable Summary")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("**⚠️ Portfolio Adjustments**")
            if actionables["SELL"]: st.write(f"❌ **SELL / CUT LOSS:** {', '.join(actionables['SELL'])} (Weak score or Stop-Loss)")
            if actionables["ADD"]: st.write(f"💎 **ADD MORE:** {', '.join(actionables['ADD'])} (Strong technical breakout)")
            if not actionables["SELL"] and not actionables["ADD"]: st.write("✅ Portfolio is stable. No immediate action required.")

        with col2:
            st.success("**🚀 New Opportunities (High Score)**")
            if not main_df.empty:
                top_new = main_df[~main_df['Stock'].isin(st.session_state['portfolio'].keys())].head(3)
                for _, r in top_new.iterrows():
                    st.write(f"✨ **{r['Stock']}** (Score: {r['Score']}/10) - Target: ₹{r['Target']}")
            else:
                st.write("Run Market Scan to see new opportunities.")
                
