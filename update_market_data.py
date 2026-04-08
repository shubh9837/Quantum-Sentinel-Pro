import yfinance as yf
import pandas as pd
import os

def download_ohlcv():
    if not os.path.exists("tickers_enriched.csv"):
        print("❌ Error: tickers_enriched.csv missing!")
        return

    try:
        meta = pd.read_csv("tickers_enriched.csv")
        symbols = [f"{s}.NS" for s in meta['SYMBOL']]
        if "^NSEI" not in symbols:
            symbols.append("^NSEI")

        print("Downloading market data...")
        data = yf.download(symbols, period="2y", interval="1d", group_by='column')
        data.to_parquet("market_data.parquet")
        print("✅ Market data saved.")
    except Exception as e:
        print(f"❌ Error in download: {e}")

if __name__ == "__main__":
    download_ohlcv()
    
