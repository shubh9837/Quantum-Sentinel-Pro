import yfinance as yf
import pandas as pd

def download_ohlcv():
    try:
        # This now reads from the file created by the script above
        meta = pd.read_csv("tickers_enriched.csv")
        symbols = [f"{s}.NS" for s in meta['SYMBOL']]
        if "^NSEI" not in symbols:
            symbols.append("^NSEI")

        print("Downloading market data...")
        data = yf.download(symbols, period="2y", interval="1d", group_by='column')
        data.to_parquet("market_data.parquet")
        print("✅ Market data saved.")
    except Exception as e:
        print(f"Error in download: {e}")

if __name__ == "__main__":
    download_ohlcv()
