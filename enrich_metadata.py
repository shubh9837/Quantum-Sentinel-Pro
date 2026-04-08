import yfinance as yf
import pandas as pd
import time
import os

def enrich_tickers():
    # Safety Check: Try both names
    file_name = "Tickers.csv" if os.path.exists("Tickers.csv") else "tickers.csv"
    
    if not os.path.exists(file_name):
        print(f"❌ Error: Neither Tickers.csv nor tickers.csv found!")
        return

    try:
        df = pd.read_csv(file_name)
        symbols = df['SYMBOL'].tolist()
        print(f"Found {len(symbols)} symbols in {file_name}")
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return

    enriched_data = []
    for sym in symbols:
        try:
            t = yf.Ticker(f"{sym}.NS")
            info = t.info
            m_cap = info.get('marketCap', 0) / 10000000 
            sector = info.get('sector', 'Other')
            enriched_data.append({"SYMBOL": sym, "SECTOR": sector, "MARKET_CAP": round(m_cap, 2)})
            print(f"✅ {sym} processed")
        except:
            enriched_data.append({"SYMBOL": sym, "SECTOR": "Other", "MARKET_CAP": 0})
        time.sleep(0.1)

    pd.DataFrame(enriched_data).to_csv("tickers_enriched.csv", index=False)
    print("✨ Metadata updated.")

if __name__ == "__main__":
    enrich_tickers()
