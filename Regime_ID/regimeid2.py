"""
ETF REGIME ENGINE — INSTITUTIONAL VERSION
Adds Volatility Shock Prediction Layer

Outputs:
- ETF regime table
- Category composite strength
- Volatility shock scores
- Dashboard
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

ETF_FILE = "/Users/jazzhashzzz/Desktop/Regime_ID/etf_only.txt"
BASE_OUTPUT = "/Users/jazzhashzzz/Desktop/Regime_ID/output"

SHORT_LOOKBACK = 20
LONG_LOOKBACK = 100
ACCEL_LOOKBACK = 5
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
# SAFE DOWNLOAD
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

        if data is None or len(data) < LONG_LOOKBACK + ACCEL_LOOKBACK:
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
        info = yf.Ticker(ticker).info or {}
        return (
            info.get("category", "Unknown"),
            info.get("fundFamily", "Unknown"),
            info.get("legalType", "Unknown")
        )
    except:
        return "Unknown", "Unknown", "Unknown"

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
# SHOCK METRICS
# ============================================================

def compute_shock_metrics(returns):

    short_vol_series = (
        returns.rolling(SHORT_LOOKBACK).std() * np.sqrt(TRADING_DAYS)
    )

    long_vol_series = (
        returns.rolling(LONG_LOOKBACK).std() * np.sqrt(TRADING_DAYS)
    )

    vol_ratio_series = short_vol_series / long_vol_series

    current_ratio = vol_ratio_series.iloc[-1]
    past_ratio = vol_ratio_series.iloc[-ACCEL_LOOKBACK]

    vol_acceleration = current_ratio - past_ratio

    compression_depth = (
        long_vol_series.iloc[-1] - short_vol_series.iloc[-1]
    ) / long_vol_series.iloc[-1]

    momentum_series = returns.rolling(SHORT_LOOKBACK).mean()
    momentum_instability = momentum_series.iloc[-SHORT_LOOKBACK:].std()

    return (
        current_ratio,
        vol_acceleration,
        compression_depth,
        momentum_instability
    )

# ============================================================
# ANALYZE ETF
# ============================================================

def analyze_etf(ticker):

    data = safe_download(ticker)

    if data is None:
        return None

    prices = data["Close"].dropna()
    returns = prices.pct_change().dropna()

    # Momentum
    short_mom = (
        prices.iloc[-1] / prices.iloc[-SHORT_LOOKBACK] - 1
    ) * 100

    long_mom = (
        prices.iloc[-1] / prices.iloc[-LONG_LOOKBACK] - 1
    ) * 100

    # Volatility
    short_vol = (
        returns.iloc[-SHORT_LOOKBACK:].std()
        * np.sqrt(TRADING_DAYS)
    )

    long_vol = (
        returns.iloc[-LONG_LOOKBACK:].std()
        * np.sqrt(TRADING_DAYS)
    )

    vol_ratio = short_vol / long_vol if long_vol != 0 else 1

    # Shock metrics
    (
        current_ratio,
        vol_acceleration,
        compression_depth,
        momentum_instability
    ) = compute_shock_metrics(returns)

    short_regime = classify_regime(short_mom, vol_ratio)
    long_regime = classify_regime(long_mom, vol_ratio)

    category, fund_family, legal_type = get_etf_metadata(ticker)

    return {

        "ticker": ticker,

        "category": category,

        "fund_family": fund_family,

        "short_momentum_%": round(short_mom, 2),

        "long_momentum_%": round(long_mom, 2),

        "vol_ratio": round(current_ratio, 3),

        "vol_acceleration": round(vol_acceleration, 4),

        "compression_depth": round(compression_depth, 4),

        "momentum_instability": round(momentum_instability, 5),

        "short_regime": short_regime,

        "long_regime": long_regime
    }

# ============================================================
# SCORING
# ============================================================

def add_scores(df):

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

    # Normalize shock inputs
    df["vol_accel_norm"] = (
        df["vol_acceleration"] -
        df["vol_acceleration"].mean()
    ) / df["vol_acceleration"].std()

    df["compression_norm"] = (
        df["compression_depth"] -
        df["compression_depth"].mean()
    ) / df["compression_depth"].std()

    df["momentum_instability_norm"] = (
        df["momentum_instability"] -
        df["momentum_instability"].mean()
    ) / df["momentum_instability"].std()

    # Shock score
    df["shock_score"] = (
        0.35 * df["vol_accel_norm"] +
        0.25 * df["compression_norm"] +
        0.20 * df["momentum_instability_norm"] +
        0.20 * df["composite_score"]
    )

    return df

# ============================================================
# CATEGORY AGGREGATION
# ============================================================

def aggregate_category(df):

    grouped = df.groupby("category").agg(

        count=("ticker", "count"),

        avg_composite=("composite_score", "mean"),

        avg_shock=("shock_score", "mean"),

        bull_expansion_pct=(
            "long_regime",
            lambda x: (x == "Bull Expansion").mean() * 100
        ),

        bear_expansion_pct=(
            "long_regime",
            lambda x: (x == "Bear Expansion").mean() * 100
        )
    )

    return grouped.sort_values("avg_composite", ascending=False)

# ============================================================
# DASHBOARD
# ============================================================

def create_dashboard(df, category_df, output):

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    df["long_regime"].value_counts().plot(
        kind="bar", ax=axes[0,0]
    )

    axes[0,0].set_title("Regime Distribution")

    axes[0,1].scatter(
        df["composite_score"],
        df["shock_score"]
    )

    axes[0,1].set_title("Composite vs Shock Score")

    category_df["avg_composite"].plot(
        kind="bar", ax=axes[1,0]
    )

    axes[1,0].set_title("Category Strength")

    sns.heatmap(
        category_df[["avg_composite","avg_shock"]].T,
        cmap="RdYlGn",
        center=0,
        annot=True,
        ax=axes[1,1]
    )

    axes[1,1].set_title("Category Heatmap")

    plt.tight_layout()

    plt.savefig(output / "institutional_dashboard.png")

    plt.close()

# ============================================================
# MAIN
# ============================================================

def run():

    print("\nINSTITUTIONAL ETF REGIME ENGINE")

    output = create_output_folder()

    etfs = load_etfs(ETF_FILE)

    results = []

    for ticker in etfs:

        r = analyze_etf(ticker)

        if r:
            results.append(r)

    df = pd.DataFrame(results)

    df = add_scores(df)

    category_df = aggregate_category(df)

    df = df.sort_values("shock_score", ascending=False)

    df.to_csv(output / "etf_regime_output.csv", index=False)

    category_df.to_csv(output / "category_composite.csv")

    create_dashboard(df, category_df, output)

    print("\nTop Shock Candidates:")
    print(df[["ticker","shock_score"]].head(10))

    print(f"\nSaved to: {output}")

# ============================================================

if __name__ == "__main__":
    run()