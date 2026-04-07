import yfinance as yf
import pandas as pd
import time

def enrich_tickers():
    # 1. Load your raw list
    try:
        df = pd.read_csv("Tickers.csv")
        symbols = df['SYMBOL'].tolist()
    except Exception as e:
        print(f"Error reading Tickers.csv: {e}")
        return

    enriched_data = []
    print(f"Enriching {len(symbols)} symbols with Sector and Market Cap...")

    for sym in symbols:
        ticker_str = f"{sym}.NS"
        try:
            t = yf.Ticker(ticker_str)
            info = t.info
            
            # Market Cap in Crores (INR)
            m_cap_crore = info.get('marketCap', 0) / 10000000 
            sector = info.get('sector', 'Other')
            
            enriched_data.append({
                "SYMBOL": sym,
                "SECTOR": sector,
                "MARKET_CAP": round(m_cap_crore, 2)
            })
            print(f"✅ {sym}: {sector} | ₹{round(m_cap_crore, 2)} Cr")
        except:
            enriched_data.append({"SYMBOL": sym, "SECTOR": "Other", "MARKET_CAP": 0})
        
        time.sleep(0.2) # Small delay to avoid rate limiting

    # 2. Save for the App to use
    pd.DataFrame(enriched_data).to_csv("tickers_enriched.csv", index=False)
    print("✨ tickers_enriched.csv created successfully.")

if __name__ == "__main__":
    enrich_tickers()
  
