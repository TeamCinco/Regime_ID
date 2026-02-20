"""
ETF REGIME ENGINE
Category-Based
Composite Scoring
Single Visual Dashboard Output
(VERTICAL HEATMAP FIX)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
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
        return [line.strip().upper() for line in f if line.strip()]

# ============================================================
# SAFE PRICE DOWNLOAD
# ============================================================

def safe_download(ticker):
    try:
        data = yf.download(
            ticker,
            period="1y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False
        )

        if data is None or len(data) < LONG_LOOKBACK:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        return data

    except:
        return None

# ============================================================
# ETF METADATA
# ============================================================

def get_etf_metadata(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        category = info.get("category", "Unknown")
        fund_family = info.get("fundFamily", "Unknown")
        legal_type = info.get("legalType")
        return category, fund_family, legal_type
    except:
        return "Unknown", "Unknown", None

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
# ANALYZE ETF
# ============================================================

def analyze_etf(ticker):

    data = safe_download(ticker)
    if data is None:
        return None

    prices = data["Close"].dropna()
    returns = prices.pct_change().dropna()

    short_mom = ((prices.iloc[-1] / prices.iloc[-SHORT_LOOKBACK]) - 1) * 100
    long_mom = ((prices.iloc[-1] / prices.iloc[-LONG_LOOKBACK]) - 1) * 100

    short_vol = returns.iloc[-SHORT_LOOKBACK:].std() * np.sqrt(TRADING_DAYS)
    long_vol = returns.iloc[-LONG_LOOKBACK:].std() * np.sqrt(TRADING_DAYS)

    vol_ratio = short_vol / long_vol if long_vol != 0 else 1

    short_regime = classify_regime(short_mom, vol_ratio)
    long_regime = classify_regime(long_mom, vol_ratio)

    category, fund_family, legal_type = get_etf_metadata(ticker)

    return {
        "ticker": ticker,
        "category": category,
        "fund_family": fund_family,
        "short_momentum_%": round(short_mom, 2),
        "long_momentum_%": round(long_mom, 2),
        "vol_ratio": round(vol_ratio, 2),
        "short_regime": short_regime,
        "long_regime": long_regime
    }

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
# SINGLE DASHBOARD (VERTICAL HEATMAP)
# ============================================================

def create_dashboard(df, category_df, output_path):

    category_df = category_df.sort_values("avg_composite", ascending=False)

    fig, axes = plt.subplots(
        2, 2,
        figsize=(24, 16),
        gridspec_kw={'height_ratios':[1, 1.4]}
    )

    # 1. Long Regime Distribution
    ax = axes[0,0]

    regime_counts = df["long_regime"].value_counts().reindex([
        "Bull Expansion",
        "Bull Compression",
        "Bear Expansion",
        "Bear Compression"
    ])

    ax.bar(regime_counts.index, regime_counts.values, width=0.6)

    ax.set_title("Long-Term Regime Distribution", fontsize=14, fontweight='bold')
    ax.set_xlabel("Regime")
    ax.set_ylabel("Count")

    ax.tick_params(axis='x', rotation=35, labelsize=11)

    for label in ax.get_xticklabels():
        label.set_horizontalalignment('right')


    # 2. Momentum Scatter
    ax = axes[0,1]

    ax.scatter(
        df["short_momentum_%"],
        df["long_momentum_%"],
        alpha=0.7
    )

    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)

    ax.set_title("Momentum Map", fontsize=14, fontweight='bold')
    ax.set_xlabel("Short Momentum %")
    ax.set_ylabel("Long Momentum %")


    # 3. Category Composite Strength
    ax = axes[1,0]

    ax.bar(
        category_df.index,
        category_df["avg_composite"],
        width=0.7
    )

    ax.set_title("ETF Category Composite Strength", fontsize=14, fontweight='bold')
    ax.set_ylabel("Composite Score")

    ax.tick_params(axis='x', rotation=60, labelsize=10)

    for label in ax.get_xticklabels():
        label.set_horizontalalignment('right')

    ax.margins(x=0.01)


    # 4. Vertical Heatmap (FIXED)
    ax = axes[1,1]

    heatmap_data = category_df[["avg_composite"]]

    sns.heatmap(
        heatmap_data,
        cmap="RdYlGn",
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size":10},
        linewidths=0.5,
        cbar_kws={"shrink":0.8},
        ax=ax
    )

    ax.set_title("ETF Category Heatmap", fontsize=14, fontweight='bold')

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=0,
        fontsize=11
    )

    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        fontsize=9
    )


    plt.tight_layout(pad=4.0, w_pad=3.0, h_pad=3.0)

    plt.savefig(
        output_path / "regime_dashboard.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ============================================================
# MAIN
# ============================================================

def run_regime():

    print("\n==============================")
    print("ETF REGIME ENGINE (DASHBOARD)")
    print("==============================")

    output_path = create_output_folder()
    etfs = load_etfs(ETF_FILE)

    print(f"\nLoaded {len(etfs)} ETFs")

    results = []
    skipped = 0

    for ticker in etfs:
        res = analyze_etf(ticker)
        if res:
            results.append(res)
        else:
            skipped += 1

    if len(results) == 0:
        print("No valid ETF data.")
        return

    df = pd.DataFrame(results)

    df = add_composite_scores(df)

    category_df = aggregate_by_category(df)

    df.to_csv(output_path / "etf_regime_output.csv", index=False)

    category_df.to_csv(output_path / "category_composite.csv")

    create_dashboard(df, category_df, output_path)

    print(f"\nAnalyzed ETFs: {len(df)}")
    print(f"Skipped: {skipped}")
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    run_regime()