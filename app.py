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

# --- PERSISTENCE ---
def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r") as f:
                return json.load(f)
        except: return {}
    return {}

def save_portfolio(portfolio):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f)
    # GitHub Sync (optional - requires Secrets)
    if "GITHUB_TOKEN" in st.secrets:
        try:
            token, repo = st.secrets["GITHUB_TOKEN"], st.secrets["REPO_NAME"]
            url = f"https://api.github.com/repos/{repo}/contents/{PORTFOLIO_FILE}"
            headers = {"Authorization": f"token {token}"}
            res = requests.get(url, headers=headers)
            sha = res.json().get("sha") if res.status_code == 200 else None
            content = base64.b64encode(json.dumps(portfolio).encode()).decode()
            requests.put(url, headers=headers, json={"message":"Update","content":content,"sha":sha,"branch":"main"})
        except: pass

@st.cache_data
def load_data():
    try:
        data = pd.read_parquet("market_data.parquet")
        meta = pd.read_csv("tickers_enriched.csv")
        return data, meta
    except: return None, None

market_raw, meta = load_data()
if market_raw is None:
    st.error("Data missing. Please run the enrichment and download scripts.")
    st.stop()

# --- UTILITIES ---
def clean_sym(s): return str(s).replace(".NS", "")

@st.cache_data(ttl=3600)
def get_sentiment(symbol):
    try:
        t = yf.Ticker(f"{symbol}.NS")
        news = t.news[:5]
        score = 0
        pos = ['profit', 'growth', 'order', 'win', 'expansion', 'buy', 'upgrade']
        neg = ['loss', 'debt', 'fraud', 'penalty', 'sell', 'downgrade', 'slump']
        for n in news:
            txt = n.get('title', '').lower()
            score += sum(1 for w in pos if w in txt)
            score -= sum(1 for w in neg if w in txt)
        return 1 if score >= 1 else -1 if score <= -1 else 0
    except: return 0

# --- SIDEBAR & TREND ---
st.sidebar.title("🛡️ Sentinel Controls")
prices = market_raw['Close']
available = prices.columns.tolist()
n_sym = "^NSEI" if "^NSEI" in available else available[0]
n_c = prices[n_sym].dropna()
m_trend = "BULLISH" if n_c.iloc[-1] > n_c.ewm(span=50).mean().iloc[-1] else "BEARISH"
m_bonus = 2 if m_trend == "BULLISH" else -2

st.sidebar.markdown(f"**Market Status:** {'🟢' if m_trend == 'BULLISH' else '🔴'} {m_trend}")
min_rating = st.sidebar.slider("Min Rating", 0, 10, 7)
mcap_filter = st.sidebar.selectbox("Market Cap View", ["All", "Large Cap (>20k Cr)", "Mid/Small Cap (<20k Cr)"])

st.title("🎯 Quantum-Sentinel Pro")
tab1, tab2 = st.tabs(["📊 Intelligence Screener", "💼 Portfolio Analytics"])

# --- TAB 1: SCREENER ---
with tab1:
    if st.button("🔍 Run Multi-Factor Scan"):
        results = []
        prog = st.progress(0)
        ticks = [t for t in available if t != n_sym]
        
        for i, t in enumerate(ticks):
            try:
                c = prices[t].dropna()
                if len(c) < 200: continue
                
                curr = c.iloc[-1]
                e20, e200 = c.ewm(span=20).mean().iloc[-1], c.ewm(span=200).mean().iloc[-1]
                
                # Logic Scoring
                score = 3 + m_bonus
                if curr > e200: score += 2
                if curr > e20: score += 1
                
                # Metadata
                cs = clean_sym(t)
                m_row = meta[meta['SYMBOL'] == cs]
                m_cap = m_row['MARKET_CAP'].values[0] if not m_row.empty else 0
                sector = m_row['SECTOR'].values[0] if not m_row.empty else "Other"
                
                # Volume Strength Logic for Small Caps
                if m_cap < 20000:
                    vol = market_raw['Volume'][t].dropna()
                    if vol.iloc[-1] < vol.rolling(20).mean().iloc[-1] * 1.5:
                        score -= 2 # Filter noise
                
                # Sentiment Score
                sent_val = 0
                if score >= 6:
                    sent_val = get_sentiment(cs)
                    score += sent_val
                
                # ATR Target
                h, l = market_raw['High'][t].dropna(), market_raw['Low'][t].dropna()
                tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
                target = round(float(curr + (2.5 * tr.rolling(14).mean().iloc[-1])), 1)

                results.append({
                    "Stock": cs, "Rating": max(0, min(10, int(score))), "Sector": sector,
                    "M-Cap": int(m_cap), "Price": round(float(curr), 1), "Target": target,
                    "Upside %": round(((target-curr)/curr)*100, 1),
                    "Sentiment": "Positive" if sent_val > 0 else "Negative" if sent_val < 0 else "Neutral"
                })
            except: continue
            if i % 50 == 0: prog.progress(i/len(ticks))
        st.session_state['results'] = pd.DataFrame(results)
        prog.empty()

    if 'results' in st.session_state:
        df = st.session_state['results'].copy()
        if mcap_filter == "Large Cap (>20k Cr)": df = df[df['M-Cap'] >= 20000]
        elif mcap_filter == "Mid/Small Cap (<20k Cr)": df = df[df['M-Cap'] < 20000]
        st.dataframe(df[df['Rating'] >= min_rating].sort_values("Rating", ascending=False), hide_index=True)

# --- TAB 2: PORTFOLIO ---
with tab2:
    if 'portfolio' not in st.session_state: st.session_state['portfolio'] = load_portfolio()
    
    with st.expander("➕ Add/Update Stock"):
        with st.form("p_form"):
            s_sym = st.selectbox("Stock", sorted(meta['SYMBOL'].tolist()))
            c1, c2 = st.columns(2)
            s_p = c1.number_input("Avg Price", value=0.0)
            s_q = c2.number_input("Quantity", value=0, min_value=0)
            if st.form_submit_button("Save"):
                if s_q == 0: st.session_state['portfolio'].pop(s_sym, None)
                else: st.session_state['portfolio'][s_sym] = {"price": s_p, "qty": s_q}
                save_portfolio(st.session_state['portfolio'])
                st.rerun()

    if st.session_state['portfolio']:
        p_data = []
        res_df = st.session_state.get('results', pd.DataFrame())
        for sym, d in st.session_state['portfolio'].items():
            row = {"Stock": sym, "Qty": d['qty'], "Avg Price": d['price'], "Invested": d['price']*d['qty']}
            if not res_df.empty and sym in res_df['Stock'].values:
                live = res_df[res_df['Stock'] == sym].iloc[0]
                val = live['Price'] * d['qty']
                pnl = val - row['Invested']
                row.update({"Current Price": live['Price'], "Value": round(val, 1), "P&L %": round((pnl/row['Invested']*100), 1), "Verdict": live['Sentiment']})
            p_data.append(row)
        st.dataframe(pd.DataFrame(p_data), use_container_width=True, hide_index=True)
