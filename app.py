import streamlit as st
import pandas as pd
import numpy as np
import json
import os

st.set_page_config(page_title="Quantum-Sentinel Pro", layout="wide")

PORTFOLIO_FILE = "portfolio.json"

# --- DATA HELPERS ---
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
    st.error("Market data missing. Please trigger the 'download' task.")
    st.stop()

def clean_sym(s): return s.replace(".NS", "").replace("_", "&")
close_prices = data['Close']

# --- SIDEBAR: GLOBAL MARKET BREADTH (SWING ROBUSTNESS) ---
st.sidebar.title("🛡️ Risk Intelligence")
n_sym = "^NSEI"
n_close = close_prices[n_sym].dropna()
n_curr = n_close.iloc[-1]
n_ema50 = n_close.ewm(span=50).mean().iloc[-1]
market_trend = "BULLISH" if n_curr > n_ema50 else "BEARISH"

if market_trend == "BULLISH":
    st.sidebar.success(f"🟢 MARKET: {market_trend}")
    m_bonus = 2
else:
    st.sidebar.error(f"🔴 MARKET: {market_trend} (Defensive)")
    m_bonus = -2

st.sidebar.markdown("---")
st.sidebar.subheader("🎚️ Screener Filters")
min_rating = st.sidebar.slider("Min. Rating Score", 0, 10, 6)
all_sectors = sorted(meta['SECTOR'].unique().tolist())
sel_sectors = st.sidebar.multiselect("Filter by Sector", options=all_sectors)

st.title("🎯 Quantum-Sentinel: Swing Predictor")
tab1, tab2 = st.tabs(["📊 Market Screener", "💼 My Portfolio"])

# --- TAB 1: RESTORED 10/10 SCREENER ---
with tab1:
    if st.button("🔍 Run Full Market Scan"):
        results = []
        n_60d_ret = (n_curr - n_close.iloc[-60]) / n_close.iloc[-60]
        prog = st.progress(0)
        ticks = [t for t in close_prices.columns if t != n_sym]
        
        for i, t in enumerate(ticks):
            try:
                s_data = close_prices[t].dropna()
                if len(s_data) < 60: continue
                
                curr = s_data.iloc[-1]
                e20 = s_data.ewm(span=20).mean().iloc[-1]
                e50 = s_data.ewm(span=50).mean().iloc[-1]
                
                # Robust Swing Logic: Relative Strength & Volatility
                rel_str = ((curr - s_data.iloc[-60])/s_data.iloc[-60]) - n_60d_ret
                vol = s_data.pct_change().std() * np.sqrt(20) # Monthly Vol
                
                score = 4 + m_bonus
                if curr > e20: score += 1
                if e20 > e50: score += 1
                if rel_str > 0: score += 2
                
                rating = max(0, min(10, int(score)))
                target = round(float(curr * (1 + vol)), 1)
                upside = round(((target - curr) / curr) * 100, 1)
                sl = round(float(curr * (1 - (vol * 0.7))), 1)
                rr = round((target - curr) / (curr - sl), 1) if (curr - sl) > 0 else 0
                
                clean_t = clean_sym(t)
                sector = meta.loc[meta['SYMBOL'] == clean_t, 'SECTOR'].values[0] if clean_t in meta['SYMBOL'].values else "Other"

                results.append({
                    "Stock": clean_t, "Rating": rating, "Sector": sector,
                    "Current Price": round(float(curr), 1), "Target": target,
                    "Upside %": upside, "Chances (%)": rating * 10,
                    "Stop Loss": sl, "R:R": rr,
                    "Verdict": "🔥 Institutional Buy" if rating >= 8 else "✅ Momentum Play" if rating >= 6 else "↔️ Hold"
                })
            except: continue
            if i % 100 == 0: prog.progress(i/len(ticks))
        
        raw_df = pd.DataFrame(results)
        # Apply Filters
        filt_df = raw_df[raw_df['Rating'] >= min_rating]
        if sel_sectors: filt_df = filt_df[filt_df['Sector'].isin(sel_sectors)]
        st.session_state['res'] = filt_df.sort_values(by="Rating", ascending=False)
        prog.empty()

    if 'res' in st.session_state:
        search = st.text_input("🔍 Search Stock:").upper()
        df_show = st.session_state['res']
        if search: df_show = df_show[df_show['Stock'].str.contains(search)]
        st.dataframe(df_show, use_container_width=True, hide_index=True)

# --- TAB 2: PORTFOLIO & ACTIONABLE SUMMARY ---
with tab2:
    if 'portfolio' not in st.session_state: st.session_state['portfolio'] = load_portfolio()
    
    with st.expander("➕ Add / Edit Position", expanded=False):
        with st.form("p_form"):
            all_syms = sorted(meta['SYMBOL'].astype(str).unique().tolist())
            s_sym = st.selectbox("Stock", [""] + all_syms)
            ex = st.session_state['portfolio'].get(s_sym, {"price": 0.0, "qty": 0})
            c1, c2 = st.columns(2)
            s_p = c1.number_input("Avg Price", value=float(ex['price']))
            s_q = c2.number_input("Qty", value=int(ex['qty']), min_value=0)
            if st.form_submit_button("Save"):
                if s_sym:
                    if s_q == 0: st.session_state['portfolio'].pop(s_sym, None)
                    else: st.session_state['portfolio'][s_sym] = {"price": s_p, "qty": s_q}
                    save_portfolio(st.session_state['portfolio'])
                    st.rerun()

    if st.session_state['portfolio']:
        p_list, a_sell, a_add = [], [], []
        m_df = st.session_state.get('res', pd.DataFrame())

        for sym, d in st.session_state['portfolio'].items():
            inv = d['price'] * d['qty']
            row = {"Stock": sym, "Avg Price": round(d['price'], 1), "Qty": d['qty'], "Invested": round(inv, 1)}
            
            if not m_df.empty and sym in m_df['Stock'].values:
                live = m_df[m_df['Stock'] == sym].iloc[0]
                cp = float(live['Current Price'])
                val = cp * d['qty']
                pnl = val - inv
                p_pct = (pnl / inv * 100) if inv > 0 else 0
                
                # Verdict & Action logic
                if p_pct <= -7.0 or int(live['Rating']) <= 3: 
                    v = "🔴 SELL"
                    a_sell.append(sym)
                elif int(live['Rating']) >= 8: 
                    v = "🔵 BUY/ADD"
                    a_add.append(sym)
                else: v = "🟡 HOLD"

                row.update({"Current": round(val, 1), "Profit": round(pnl, 1), "%": round(p_pct, 1), "Target": live['Target'], "Verdict": v})
            else:
                row.update({k: 0.0 for k in ["Current", "Profit", "%"]})
                row.update({"Target": "-", "Verdict": "Scan Required"})
            p_list.append(row)

        df_p = pd.DataFrame(p_list)
        if not df_p.empty and "Current" in df_p.columns:
            total_row = pd.DataFrame([{"Stock": "TOTAL", "Invested": df_p["Invested"].sum(), "Current": df_p["Current"].sum(), "Profit": df_p["Profit"].sum(), "%": round((df_p["Profit"].sum()/df_p["Invested"].sum()*100),1) if df_p["Invested"].sum()>0 else 0}])
            df_p = pd.concat([df_p, total_row], ignore_index=True).fillna("-")

        st.dataframe(df_p, use_container_width=True, hide_index=True)

        st.subheader("📝 Next Session Actionable Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.info("**Portfolio Health**")
            if a_sell: st.error(f"⚠️ **EXIT:** {', '.join(a_sell)} (Technical weakness)")
            if a_add: st.success(f"💎 **ADD:** {', '.join(a_add)} (Strong momentum)")
            if not a_sell and not a_add: st.write("Positions are stable.")
        with col2:
            st.success("**Market Opportunities**")
            if not m_df.empty:
                top = m_df[~m_df['Stock'].isin(st.session_state['portfolio'].keys())].head(3)
                for _, r in top.iterrows(): st.write(f"🚀 **{r['Stock']}** (Target: {r['Target']} | {r['Upside %']}% Upside)")
