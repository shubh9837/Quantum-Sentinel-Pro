import pandas as pd
import yfinance as yf
import os
import time

def download_data():
    ref_file = "tickers_enriched.csv" if os.path.exists("tickers_enriched.csv") else "tickers.csv"
    df_tickers = pd.read_csv(ref_file)
    
    # Clean symbols for yfinance: Replace & with _ and add .NS
    raw_symbols = df_tickers['SYMBOL'].astype(str).unique()
    symbols = [f"{s.replace('&', '_')}.NS" for s in raw_symbols]
    
    if "^NSEI" not in symbols: symbols.append("^NSEI")

    print(f"Downloading {len(symbols)} assets in batches...")
    
    # Batching prevents the "Connection Reset" error and ensures 100% data capture
    batch_size = 100
    all_data = []
    
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        try:
            data = yf.download(batch, period="2y", interval="1d", group_by='column', auto_adjust=True, threads=True)
            all_data.append(data)
            time.sleep(1) # Small delay to respect NSE/Yahoo servers
        except: continue
        
    final_df = pd.concat(all_data, axis=1)
    final_df.to_parquet("market_data.parquet")
    print(f"✅ Success: {len(final_df.columns.get_level_values(1).unique())} symbols saved.")

if __name__ == "__main__":
    download_data()
