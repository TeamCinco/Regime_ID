"""
ETF REGIME ENGINE — Optimized for Full Universe
================================================
- Batch yf.download() for price data (500 tickers at a time)
- Multithreaded .info calls for metadata
- tqdm progress bars throughout
- Category-Based Composite Scoring
- Single Visual Dashboard Output

Requirements: pip install yfinance pandas numpy matplotlib seaborn tqdm
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

ETF_FILE = "/Users/jazzhashzzz/Desktop/Regime_ID/equity_etf_tickers.txt"
BASE_OUTPUT = "/Users/jazzhashzzz/Desktop/Regime_ID/output"

SHORT_LOOKBACK = 20
LONG_LOOKBACK = 100
TRADING_DAYS = 252

# Speed settings
BATCH_SIZE = 50         # tickers per yf.download() batch (keep small to avoid rate limits)
BATCH_DELAY = 5         # seconds between download batches
MAX_THREADS = 10        # threads for .info metadata calls
INFO_DELAY = 0.2        # delay between .info calls to avoid throttle

# ============================================================
# OUTPUT FOLDER
# ============================================================

def create_output_folder():
    today = datetime.now()
    folder_name = today.strftime("%Y-%m-%d")
    path = Path(BASE_OUTPUT) / folder_name
    path.mkdir(parents=True, exist_ok=True)
    return path

# ============================================================
# LOAD ETF LIST
# ============================================================

def load_etfs(filepath):
    with open(filepath, "r") as f:
        tickers = [line.strip().upper() for line in f if line.strip()]
    return sorted(list(set(tickers)))

# ============================================================
# BATCH PRICE DOWNLOAD
# ============================================================

def batch_download_prices(tickers, batch_size=BATCH_SIZE):
    """
    Download 1Y daily prices for all tickers in batches.
    Returns dict of {ticker: DataFrame}.
    """
    all_data = {}
    total_batches = (len(tickers) + batch_size - 1) // batch_size

    print(f"\n[PRICES] Downloading 1Y data in {total_batches} batches of {batch_size}...")

    pbar = tqdm(range(0, len(tickers), batch_size),
                desc="Downloading prices",
                unit="batch",
                total=total_batches)

    for i in pbar:
        batch = tickers[i : i + batch_size]
        pbar.set_postfix(tickers=f"{i+len(batch)}/{len(tickers)}", found=len(all_data))

        retries = 3
        for attempt in range(retries):
            try:
                data = yf.download(
                    batch,
                    period="1y",
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                    group_by="ticker",
                )

                if data is None or data.empty:
                    break

                # Single ticker returns flat columns, multi returns MultiIndex
                if len(batch) == 1:
                    ticker = batch[0]
                    if len(data) >= LONG_LOOKBACK:
                        if isinstance(data.columns, pd.MultiIndex):
                            data.columns = data.columns.get_level_values(0)
                        all_data[ticker] = data
                else:
                    for ticker in batch:
                        try:
                            if ticker not in data.columns.get_level_values(0):
                                continue
                            ticker_data = data[ticker].dropna(how="all")
                            if len(ticker_data) >= LONG_LOOKBACK:
                                all_data[ticker] = ticker_data
                        except (KeyError, TypeError):
                            continue

                break  # success, no retry needed

            except Exception as e:
                err_msg = str(e).lower()
                if "rate" in err_msg or "too many" in err_msg:
                    wait = BATCH_DELAY * (attempt + 2)
                    pbar.write(f"  Rate limited — waiting {wait}s (attempt {attempt+1}/{retries})")
                    time.sleep(wait)
                else:
                    pbar.write(f"  Batch error: {e}")
                    break

        # Delay between batches
        if i + batch_size < len(tickers):
            time.sleep(BATCH_DELAY)

    print(f"  Price data retrieved for {len(all_data)}/{len(tickers)} tickers")
    return all_data

# ============================================================
# THREADED METADATA PULL
# ============================================================

def fetch_single_metadata(ticker):
    """Pull only classification data — category, fund family, quote type."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        time.sleep(INFO_DELAY)
        return ticker, {
            "category": info.get("category", "Unknown") or "Unknown",
            "fund_family": info.get("fundFamily", "Unknown") or "Unknown",
            "quote_type": info.get("quoteType", "Unknown") or "Unknown",
        }
    except Exception:
        return ticker, {
            "category": "Unknown",
            "fund_family": "Unknown",
            "quote_type": "Unknown",
        }


def batch_fetch_metadata(tickers, max_threads=MAX_THREADS):
    """Pull .info for all tickers using thread pool."""
    metadata = {}

    print(f"\n[METADATA] Pulling .info for {len(tickers)} tickers ({max_threads} threads)...")

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {executor.submit(fetch_single_metadata, t): t for t in tickers}

        pbar = tqdm(as_completed(futures),
                    desc="Fetching metadata",
                    unit="ticker",
                    total=len(futures))

        for future in pbar:
            ticker = futures[future]
            try:
                t, meta = future.result()
                metadata[t] = meta
                pbar.set_postfix(last=t)
            except Exception:
                pbar.set_postfix(last=f"{ticker} ERR")

    print(f"  Metadata retrieved for {len(metadata)}/{len(tickers)} tickers")
    return metadata

# ============================================================
# REGIME CLASSIFICATION
# ============================================================

def classify_regime(momentum, vol_ratio):
    if momentum > 0 and vol_ratio > 1:
        return "Bull Expansion"
    elif momentum > 0 and vol_ratio <= 1:
        return "Bull Compression"
    elif momentum < 0 and vol_ratio > 1:
        return "Bear Expansion"
    else:
        return "Bear Compression"

# ============================================================
# ANALYZE ALL ETFs (VECTORIZED)
# ============================================================

def analyze_all(price_data, metadata):
    """
    Compute momentum, vol ratio, and regime for all tickers.
    Uses pre-downloaded price data and metadata.
    """
    results = []
    tickers = list(price_data.keys())

    print(f"\n[ANALYSIS] Computing regimes for {len(tickers)} tickers...")

    for ticker in tqdm(tickers, desc="Analyzing", unit="etf"):
        try:
            data = price_data[ticker]
            prices = data["Close"].dropna()

            if len(prices) < LONG_LOOKBACK:
                continue

            returns = prices.pct_change().dropna()

            short_mom = ((prices.iloc[-1] / prices.iloc[-SHORT_LOOKBACK]) - 1) * 100
            long_mom = ((prices.iloc[-1] / prices.iloc[-LONG_LOOKBACK]) - 1) * 100

            short_vol = returns.iloc[-SHORT_LOOKBACK:].std() * np.sqrt(TRADING_DAYS)
            long_vol = returns.iloc[-LONG_LOOKBACK:].std() * np.sqrt(TRADING_DAYS)

            vol_ratio = short_vol / long_vol if long_vol != 0 else 1

            short_regime = classify_regime(short_mom, vol_ratio)
            long_regime = classify_regime(long_mom, vol_ratio)

            meta = metadata.get(ticker, {})

            results.append({
                "ticker": ticker,
                "category": meta.get("category", "Unknown"),
                "fund_family": meta.get("fund_family", "Unknown"),
                "quote_type": meta.get("quote_type", "Unknown"),
                "short_momentum_%": round(float(short_mom), 2),
                "long_momentum_%": round(float(long_mom), 2),
                "vol_ratio": round(float(vol_ratio), 2),
                "short_regime": short_regime,
                "long_regime": long_regime,
            })

        except Exception:
            continue

    return results

# ============================================================
# COMPOSITE SCORING
# ============================================================

def add_composite_scores(df):
    score_map = {
        "Bull Expansion": 2,
        "Bull Compression": 1,
        "Bear Compression": -1,
        "Bear Expansion": -2
    }

    df["short_score"] = df["short_regime"].map(score_map)
    df["long_score"] = df["long_regime"].map(score_map)

    df["composite_score"] = (
        df["short_score"] * 0.4 +
        df["long_score"] * 0.6
    )

    return df

# ============================================================
# CATEGORY AGGREGATION
# ============================================================

def aggregate_by_category(df):
    grouped = df.groupby("category").agg(
        count=("ticker", "count"),
        avg_composite=("composite_score", "mean"),
        avg_short_mom=("short_momentum_%", "mean"),
        avg_long_mom=("long_momentum_%", "mean"),
        bull_expansion_pct=("long_regime",
                            lambda x: (x == "Bull Expansion").mean() * 100),
        bear_expansion_pct=("long_regime",
                            lambda x: (x == "Bear Expansion").mean() * 100)
    )

    return grouped.sort_values("avg_composite", ascending=False)

# ============================================================
# SINGLE DASHBOARD
# ============================================================

def create_dashboard(df, category_df, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # 1. Long Regime Distribution
    df["long_regime"].value_counts().plot(
        kind="bar", ax=axes[0, 0]
    )
    axes[0, 0].set_title("Long-Term Regime Distribution")
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 2. Momentum Scatter
    axes[0, 1].scatter(
        df["short_momentum_%"],
        df["long_momentum_%"],
        alpha=0.5,
        s=15,
    )
    axes[0, 1].axhline(0, color="gray", linewidth=0.5)
    axes[0, 1].axvline(0, color="gray", linewidth=0.5)
    axes[0, 1].set_title("Momentum Map")
    axes[0, 1].set_xlabel("Short Momentum %")
    axes[0, 1].set_ylabel("Long Momentum %")

    # 3. Top/Bottom 20 Category Composite Strength
    top_cats = category_df.head(20)
    top_cats["avg_composite"].plot(kind="bar", ax=axes[1, 0])
    axes[1, 0].set_title("Top 20 ETF Categories by Composite Strength")
    axes[1, 0].tick_params(axis='x', rotation=45)

    # 4. Heatmap — top 20 categories
    heatmap_data = top_cats[["avg_composite"]].T
    sns.heatmap(
        heatmap_data,
        cmap="RdYlGn",
        center=0,
        annot=True,
        fmt=".1f",
        ax=axes[1, 1]
    )
    axes[1, 1].set_title("ETF Category Heatmap (Top 20)")
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_path / "regime_dashboard.png", dpi=150)
    plt.close()

# ============================================================
# MAIN
# ============================================================

def run_regime():
    start_time = time.time()

    print("\n" + "=" * 60)
    print("ETF REGIME ENGINE — Full Universe")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    output_path = create_output_folder()
    etfs = load_etfs(ETF_FILE)
    print(f"\nLoaded {len(etfs)} ETFs from {ETF_FILE}")

    # Step 1: Batch download all price data
    price_data = batch_download_prices(etfs, BATCH_SIZE)

    # Step 2: Multithreaded metadata pull (only for tickers with price data)
    valid_tickers = list(price_data.keys())
    metadata = batch_fetch_metadata(valid_tickers, MAX_THREADS)

    # Step 3: Analyze — compute momentum, vol, regimes
    results = analyze_all(price_data, metadata)

    if len(results) == 0:
        print("\nNo valid ETF data found.")
        return

    df = pd.DataFrame(results)

    # Step 4: Composite scoring
    df = add_composite_scores(df)
    category_df = aggregate_by_category(df)

    # Step 5: Save outputs
    print(f"\n[SAVE] Writing output files...")

    df.to_csv(output_path / "etf_regime_output.csv", index=False)
    category_df.to_csv(output_path / "category_composite.csv")

    # Step 6: Dashboard
    print("[DASHBOARD] Generating visualization...")
    create_dashboard(df, category_df, output_path)

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"DONE")
    print(f"  Analyzed:   {len(df)} ETFs")
    print(f"  Skipped:    {len(etfs) - len(df)}")
    print(f"  Categories: {df['category'].nunique()}")
    print(f"  Runtime:    {elapsed/60:.1f} minutes")
    print(f"  Output:     {output_path}")
    print(f"{'=' * 60}")

    # Quick regime summary
    print(f"\nRegime Breakdown:")
    print(df["long_regime"].value_counts().to_string())

    print(f"\nTop 10 Categories:")
    print(category_df.head(10)[["count", "avg_composite", "bull_expansion_pct"]].to_string())


if __name__ == "__main__":
    run_regime()