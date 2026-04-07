import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import requests
import base64
import yfinance as yf

st.set_page_config(page_title="Quantum-Sentinel Pro", layout="wide")

PORTFOLIO_FILE = "portfolio.json"

# --- GITHUB PERSISTENCE ENGINE ---
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
    try:
        if "GITHUB_TOKEN" in st.secrets and "REPO_NAME" in st.secrets:
            token = st.secrets["GITHUB_TOKEN"]
            repo = st.secrets["REPO_NAME"]
            url = f"https://api.github.com/repos/{repo}/contents/{PORTFOLIO_FILE}"
            headers = {"Authorization": f"token {token}"}
            res = requests.get(url, headers=headers)
            sha = res.json().get("sha") if res.status_code == 200 else None
            content = base64.b64encode(json.dumps(portfolio).encode()).decode()
            payload = {"message": "Update Portfolio", "content": content, "branch": "main"}
            if sha: payload["sha"] = sha
            requests.put(url, headers=headers, json=payload)
    except Exception as e:
        st.error(f"GitHub Sync Failed: {e}")

@st.cache_data
def load_all_data():
    try:
        data = pd.read_parquet("market_data.parquet", engine='pyarrow')
        meta = pd.read_csv("tickers_enriched.csv")
        return data, meta
    except: return None, None

data, meta = load_all_data()
if data is None:
    st.error("Market data missing. Please check your data pipeline.")
    st.stop()

def clean_sym(s): return str(s).replace(".NS", "").replace("_", "&")

# Safely identify Close prices
if isinstance(data.columns, pd.MultiIndex):
    close_prices = data['Close']
else:
    close_prices = data

# --- NEWS SENTIMENT ENGINE ---
@st.cache_data(ttl=3600)
def get_sentiment_score(symbol):
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        news = ticker.news[:5]
        score = 0
        pos_words = ['growth', 'profit', 'order', 'win', 'buy', 'upgrade', 'expansion', 'dividend', 'positive']
        neg_words = ['loss', 'debt', 'fraud', 'penalty', 'sell', 'downgrade', 'slump', 'investigation', 'negative']
        for n in news:
            title = n.get('title', '').lower()
            for w in pos_words: 
                if w in title: score += 1
            for w in neg_words: 
                if w in title: score -= 1
        return 1 if score >= 1 else -1 if score <= -1 else 0
    except: return 0

# --- SIDEBAR: STRATEGY & FILTERS ---
st.sidebar.title("🛡️ Sentinel Controls")

# FIXED: Safe Benchmark Selection
available_cols = close_prices.columns.tolist()
n_sym = "^NSEI" if "^NSEI" in available_cols else (available_cols[0] if available_cols else None)

if n_sym:
    n_close = close_prices[n_sym].dropna()
    n_curr = n_close.iloc[-1]
    n_ema50 = n_close.ewm(span=50).mean().iloc[-1]
    market_trend = "BULLISH" if n_curr > n_ema50 else "BEARISH"
else:
    market_trend = "NEUTRAL"

if market_trend == "BULLISH":
    st.sidebar.success(f"🟢 MARKET: {market_trend}")
    m_bonus = 2
else:
    st.sidebar.error(f"🔴 MARKET: {market_trend}")
    m_bonus = -2

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Intelligence Filters")
min_rating = st.sidebar.slider("Minimum Rating Score", 0, 10, 7)
mcap_focus = st.sidebar.selectbox("Market Cap Filter", ["All Stocks", "Large Cap (>20k Cr)", "Mid/Small Cap (<20k Cr)"])
all_sectors = sorted(meta['SECTOR'].unique().tolist()) if 'SECTOR' in meta.columns else []
sel_sectors = st.sidebar.multiselect("Filter by Sector", options=all_sectors)

st.title("🎯 Quantum-Sentinel: Personal Swing Predictor")
tab1, tab2 = st.tabs(["📊 Market Screener", "💼 My Portfolio"])

# --- TAB 1: MARKET SCREENER ---
with tab1:
    if st.button("🔍 Run Multi-Factor Intelligence Scan"):
        results = []
        prog = st.progress(0)
        ticks = [t for t in available_cols if t != n_sym]
        
        for i, t in enumerate(ticks):
            try:
                c_data = close_prices[t].dropna()
                if len(c_data) < 200: continue
                
                curr = c_data.iloc[-1]
                e20, e50, e200 = c_data.ewm(span=20).mean().iloc[-1], c_data.ewm(span=50).mean().iloc[-1], c_data.ewm(span=200).mean().iloc[-1]
                
                # 1. Technical Baseline (Score: 0-6)
                t_score = 2 + m_bonus
                if curr > e200: t_score += 2
                if curr > e20: t_score += 1
                if e20 > e50: t_score += 1
                
                # RSI Check
                delta = c_data.diff()
                up, down = delta.clip(lower=0).ewm(com=13).mean(), -1*delta.clip(upper=0).ewm(com=13).mean()
                rsi = (100 - (100 / (1 + (up/down)))).iloc[-1]
                if 45 <= rsi <= 68: t_score += 1
                elif rsi > 75: t_score -= 2
                
                # 2. Fundamental & M-Cap Strength Check
                clean_t = clean_sym(t)
                stock_meta = meta[meta['SYMBOL'] == clean_t]
                m_cap = stock_meta['MARKET_CAP'].values[0] if not stock_meta.empty else 0
                sector = stock_meta['SECTOR'].values[0] if not stock_meta.empty else "Other"
                
                # Logical Strengthening for Small/Mid Caps
                if m_cap < 20000:
                    if isinstance(data.columns, pd.MultiIndex) and 'Volume' in data.columns.levels[0]:
                        v_data = data['Volume'][t].dropna()
                        if v_data.iloc[-1] < v_data.rolling(20).mean().iloc[-1] * 1.3:
                            t_score -= 2 # Penalize low-volume breakout
                
                # 3. Sentiment Integration (only for potential buys)
                sentiment_val = 0
                if t_score >= 6:
                    sentiment_val = get_sentiment_score(clean_t)
                    t_score += sentiment_val
                
                # 4. ATR Target Calculation
                target = 0.0
                if isinstance(data.columns, pd.MultiIndex) and 'High' in data.columns.levels[0]:
                    h, l = data['High'][t].dropna(), data['Low'][t].dropna()
                    tr = pd.concat([h-l, (h-c_data.shift()).abs(), (l-c_data.shift()).abs()], axis=1).max(axis=1)
                    atr = tr.rolling(14).mean().iloc[-1]
                    target = round(float(curr + (2.5 * atr)), 1)
                else:
                    target = round(float(curr * 1.12), 1) # Fallback 12% target

                final_rating = max(0, min(10, int(t_score)))
                results.append({
                    "Stock": clean_t, "Rating": final_rating, "Sector": sector, "M-Cap (Cr)": int(m_cap),
                    "Current Price": round(float(curr), 1), "Target": target,
                    "Upside %": round(((target - curr) / curr) * 100, 1) if curr > 0 else 0,
                    "Sentiment": "Positive" if sentiment_val > 0 else "Negative" if sentiment_val < 0 else "Neutral",
                    "Verdict": "🔥 Strong Buy" if final_rating >= 8 else "✅ Momentum Play" if final_rating >= 6 else "↔️ Hold/Avoid"
                })
            except: continue
            if i % 100 == 0: prog.progress(i/len(ticks))
        
        st.session_state['raw_results'] = pd.DataFrame(results)
        prog.empty()

    if 'raw_results' in st.session_state and not st.session_state['raw_results'].empty:
        df_show = st.session_state['raw_results'].copy()
        # Apply Sidebar Filters
        if mcap_focus == "Large Cap (>20k Cr)": df_show = df_show[df_show['M-Cap (Cr)'] >= 20000]
        elif mcap_focus == "Mid/Small Cap (<20k Cr)": df_show = df_show[df_show['M-Cap (Cr)'] < 20000]
        if sel_sectors: df_show = df_show[df_show['Sector'].isin(sel_sectors)]
        
        df_show = df_show[df_show['Rating'] >= min_rating]
        search = st.text_input("🔍 Search Stock:").upper()
        if search: df_show = df_show[df_show['Stock'].str.contains(search)]
        
        st.dataframe(df_show.sort_values(by="Rating", ascending=False), use_container_width=True, hide_index=True)

# --- TAB 2: MY PORTFOLIO ---
with tab2:
    if 'portfolio' not in st.session_state: 
        st.session_state['portfolio'] = load_portfolio()
    
    with st.expander("➕ Add / Edit Position", expanded=False):
        with st.form("p_form"):
            all_syms = sorted(meta['SYMBOL'].astype(str).unique().tolist())
            s_sym = st.selectbox("Select Stock", [""] + all_syms)
            ex = st.session_state['portfolio'].get(s_sym, {"price": 0.0, "qty": 0})
            c1, c2 = st.columns(2)
            s_p = c1.number_input("Avg Buy Price", value=float(ex['price']))
            s_q = c2.number_input("Quantity", value=int(ex['qty']), min_value=0)
            if st.form_submit_button("Save Position"):
                if s_sym:
                    if s_q == 0: st.session_state['portfolio'].pop(s_sym, None)
                    else: st.session_state['portfolio'][s_sym] = {"price": s_p, "qty": s_q}
                    save_portfolio(st.session_state['portfolio'])
                    st.rerun()

    if st.session_state['portfolio']:
        p_list, a_sell, a_add = [], [], []
        m_df = st.session_state.get('raw_results', pd.DataFrame())

        for sym, d in st.session_state['portfolio'].items():
            inv = d['price'] * d['qty']
            row = {"Stock": sym, "Avg Price": round(d['price'], 1), "Qty": d['qty'], "Invested": round(inv, 1)}
            
            if not m_df.empty and sym in m_df['Stock'].values:
                live = m_df[m_df['Stock'] == sym].iloc[0]
                cp = float(live['Current Price'])
                val = cp * d['qty']
                pnl = val - inv
                p_pct = (pnl / inv * 100) if inv > 0 else 0
                t_price = live['Target']
                t_increase = round(((t_price - cp) / cp * 100), 1) if cp > 0 else 0

                if p_pct <= -7.0 or int(live['Rating']) <= 3: 
                    v, a_sell = "🔴 SELL", a_sell + [sym]
                elif int(live['Rating']) >= 8: 
                    v, a_add = "🔵 BUY/ADD", a_add + [sym]
                else: v = "🟡 HOLD"

                row.update({
                    "Current Price": cp, "Current Value": round(val, 1),
                    "Unrealised Profit": round(pnl, 1), "Profit %": round(p_pct, 1),
                    "Target Price": t_price, "Target Addl. %": t_increase, "Verdict": v
                })
            else:
                row.update({k: 0.0 for k in ["Current Price", "Current Value", "Unrealised Profit", "Profit %", "Target Price", "Target Addl. %"]})
                row.update({"Verdict": "Scan Market First"})
            p_list.append(row)

        df_p = pd.DataFrame(p_list)
        if not df_p.empty and "Current Value" in df_p.columns:
            total_inv = df_p["Invested"].sum()
            total_pnl = df_p["Unrealised Profit"].sum()
            total_row = pd.DataFrame([{
                "Stock": "TOTAL", "Invested": round(total_inv, 1), 
                "Current Value": round(df_p["Current Value"].sum(), 1), 
                "Unrealised Profit": round(total_pnl, 1), 
                "Profit %": round((total_pnl/total_inv*100),1) if total_inv > 0 else 0
            }])
            df_p = pd.concat([df_p, total_row], ignore_index=True).fillna("-")

        st.dataframe(df_p, use_container_width=True, hide_index=True)
        
        st.subheader("📝 Trading Session Actionables")
        col1, col2 = st.columns(2)
        with col1:
            st.info("**Portfolio Risk Management**")
            if a_sell: st.error(f"⚠️ **EXIT:** {', '.join(a_sell)}")
            if a_add: st.success(f"💎 **ADD:** {', '.join(a_add)}")
            if not a_sell and not a_add: st.write("Portfolio remains technically sound.")
        with col2:
            st.success("**New Market Opportunities**")
            if not m_df.empty:
                top = m_df[~m_df['Stock'].isin(st.session_state['portfolio'].keys())].head(3)
                for _, r in top.iterrows(): 
                    st.write(f"🚀 **{r['Stock']}** (Target: {r['Target']} | {r['Upside %']}% Potential)")
                
