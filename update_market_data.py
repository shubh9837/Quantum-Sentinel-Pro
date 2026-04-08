import yf
import pandas as pd

def download_ohlcv():
    # Load symbols from your enriched file
    meta = pd.read_csv("tickers_enriched.csv")
    symbols = [f"{s}.NS" for s in meta['SYMBOL']]
    
    # Force include Nifty 50 for Market Trend logic
    if "^NSEI" not in symbols:
        symbols.append("^NSEI")

    print(f"Downloading OHLCV data for {len(symbols)} tickers...")
    
    # Fetch 2 years of daily data (Open, High, Low, Close, Volume)
    data = yf.download(symbols, period="2y", interval="1d", group_by='column')
    
    data.to_parquet("market_data.parquet")
    print("✅ market_data.parquet updated.")

if __name__ == "__main__":
    download_ohlcv()
    
