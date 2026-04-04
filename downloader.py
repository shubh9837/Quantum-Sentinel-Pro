import pandas as pd
import yfinance as yf
import os

def download_data():
    if not os.path.exists("tickers_enriched.csv"):
        print("⚠️ tickers_enriched.csv not found. Using tickers.csv instead.")
        ref_file = "tickers.csv"
    else:
        ref_file = "tickers_enriched.csv"

    df_tickers = pd.read_csv(ref_file)
    symbols = [f"{s}.NS" for s in df_tickers['SYMBOL'].astype(str).unique()]
    
    # Add Index for Relative Strength
    if "^NSEI.NS" not in symbols: symbols.append("^NSEI")

    print(f"Downloading {len(symbols)} assets...")
    # group_by='column' makes the data easier for the app to read
    data = yf.download(symbols, period="2y", interval="1d", group_by='column', auto_adjust=True)
    
    data.to_parquet("market_data.parquet")
    print("✅ System Ready: market_data.parquet created.")

if __name__ == "__main__":
    download_data()

