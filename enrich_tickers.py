import pandas as pd
import requests
from io import StringIO

def run_enrichment():
    sources = [
        "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
        "https://nsearchives.nseindia.com/content/indices/ind_niftymicrocap250list.csv"
    ]
    master_map = {}
    headers = {'User-Agent': 'Mozilla/5.0'}

    for url in sources:
        try:
            res = requests.get(url, headers=headers, timeout=10)
            temp_df = pd.read_csv(StringIO(res.text))
            sym_col = next(c for c in temp_df.columns if 'Symbol' in c)
            ind_col = next(c for c in temp_df.columns if 'Industry' in c or 'Sector' in c)
            for _, row in temp_df.iterrows():
                master_map[str(row[sym_col]).upper()] = row[ind_col]
        except: continue

    df = pd.read_csv("tickers.csv")
    df['SYMBOL'] = df['SYMBOL'].astype(str).str.upper()
    df['SECTOR'] = df['SYMBOL'].map(master_map).fillna("SmallCap/Other")
    df.to_csv("tickers_enriched.csv", index=False)
    print("✅ Sectors Mapped.")

if __name__ == "__main__":
    run_enrichment()
  
