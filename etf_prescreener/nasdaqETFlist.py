"""
Fetch all US-traded ETF tickers from Nasdaq's official traded symbols file.
Free, no API key required.

Output: etf_list.csv with Symbol, Security Name, and Exchange info.
"""

import pandas as pd

def get_all_etfs():
    url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
    
    # Read the pipe-delimited file
    df = pd.read_csv(url, sep="|")
    
    # Drop the last row (file creation timestamp row)
    df = df.dropna(subset=["ETF"])
    
    # Filter for ETFs only
    etfs = df[df["ETF"] == "Y"].copy()
    
    # Clean up column names (they have trailing spaces sometimes)
    etfs.columns = etfs.columns.str.strip()
    
    # Keep useful columns
    cols_to_keep = ["Symbol", "Security Name", "Listing Exchange", "Market Category", "ETF"]
    available_cols = [c for c in cols_to_keep if c in etfs.columns]
    etfs = etfs[available_cols].reset_index(drop=True)
    
    # Clean symbol column
    etfs["Symbol"] = etfs["Symbol"].str.strip()
    
    return etfs

if __name__ == "__main__":
    print("Fetching ETF list from Nasdaq...")
    etfs = get_all_etfs()
    
    print(f"\nTotal ETFs found: {len(etfs)}")
    print(f"\nFirst 20 ETFs:")
    print(etfs.head(20).to_string(index=False))
    
    # Save to CSV
    etfs.to_csv("etf_list.csv", index=False)
    print(f"\nFull list saved to etf_list.csv")
    
    # Also save just the ticker symbols as a plain text list
    etfs["Symbol"].to_csv("etf_tickers.txt", index=False, header=False)
    print(f"Ticker-only list saved to etf_tickers.txt")