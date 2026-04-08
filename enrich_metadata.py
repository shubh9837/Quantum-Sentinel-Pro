import yfinance as yf
import pandas as pd
import time

def enrich_tickers():
    try:
        # FIXED: Changed to capital T to match your file name
        df = pd.read_csv("Tickers.csv") 
        symbols = df['SYMBOL'].tolist()
    except Exception as e:
        print(f"Error: Could not find Tickers.csv. Please check the name. {e}")
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
        time.sleep(0.2)

    pd.DataFrame(enriched_data).to_csv("tickers_enriched.csv", index=False)
    print("✨ Metadata updated.")

if __name__ == "__main__":
    enrich_tickers()
