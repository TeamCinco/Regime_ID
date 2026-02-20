"""
ETF REGIME ENGINE — INSTITUTIONAL VERSION
Adds Volatility Shock Prediction Layer
(FIXED DASHBOARD WITH TALL HEATMAP)
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

ETF_FILE = "/Users/jazzhashzzz/Desktop/Regime_ID/equity_etf_tickers.txt"
BASE_OUTPUT = "/Users/jazzhashzzz/Desktop/Regime_ID/output"

SHORT_LOOKBACK = 30
LONG_LOOKBACK = 100
ACCEL_LOOKBACK = 20
TRADING_DAYS = 252


def create_output_folder():
    today = datetime.now()
    folder_name = today.strftime("%Y-%m-%d")
    path = Path(BASE_OUTPUT) / folder_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_etfs(filepath):
    with open(filepath, "r") as f:
        return [line.strip().upper() for line in f if line.strip()]


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


def classify_regime(momentum, vol_ratio):
    if momentum > 0 and vol_ratio > 1:
        return "Bull Expansion"
    elif momentum > 0:
        return "Bull Compression"
    elif momentum < 0 and vol_ratio > 1:
        return "Bear Expansion"
    else:
        return "Bear Compression"


def compute_shock_metrics(returns):

    short_vol = returns.rolling(SHORT_LOOKBACK).std() * np.sqrt(TRADING_DAYS)
    long_vol = returns.rolling(LONG_LOOKBACK).std() * np.sqrt(TRADING_DAYS)

    vol_ratio = short_vol / long_vol

    current_ratio = vol_ratio.iloc[-1]
    past_ratio = vol_ratio.iloc[-ACCEL_LOOKBACK]

    vol_acceleration = current_ratio - past_ratio

    compression_depth = (
        long_vol.iloc[-1] - short_vol.iloc[-1]
    ) / long_vol.iloc[-1]

    momentum_instability = (
        returns.rolling(SHORT_LOOKBACK).mean()
        .iloc[-SHORT_LOOKBACK:]
        .std()
    )

    return current_ratio, vol_acceleration, compression_depth, momentum_instability


def analyze_etf(ticker):

    data = safe_download(ticker)
    if data is None:
        return None

    prices = data["Close"].dropna()
    returns = prices.pct_change().dropna()

    short_mom = (prices.iloc[-1]/prices.iloc[-SHORT_LOOKBACK]-1)*100
    long_mom = (prices.iloc[-1]/prices.iloc[-LONG_LOOKBACK]-1)*100

    short_vol = returns.iloc[-SHORT_LOOKBACK:].std()*np.sqrt(TRADING_DAYS)
    long_vol = returns.iloc[-LONG_LOOKBACK:].std()*np.sqrt(TRADING_DAYS)

    vol_ratio = short_vol/long_vol if long_vol != 0 else 1

    shock = compute_shock_metrics(returns)

    short_regime = classify_regime(short_mom, vol_ratio)
    long_regime = classify_regime(long_mom, vol_ratio)

    category, family, legal = get_etf_metadata(ticker)

    return {

        "ticker": ticker,
        "category": category,
        "short_momentum_%": round(short_mom,2),
        "long_momentum_%": round(long_mom,2),

        "vol_ratio": round(shock[0],3),
        "vol_acceleration": round(shock[1],4),
        "compression_depth": round(shock[2],4),
        "momentum_instability": round(shock[3],5),

        "short_regime": short_regime,
        "long_regime": long_regime
    }


def add_scores(df):

    score_map = {
        "Bull Expansion":2,
        "Bull Compression":1,
        "Bear Compression":-1,
        "Bear Expansion":-2
    }

    df["short_score"] = df["short_regime"].map(score_map)
    df["long_score"] = df["long_regime"].map(score_map)

    df["composite_score"] = df["short_score"]*0.4 + df["long_score"]*0.6

    df["shock_score"] = (
        (df["vol_acceleration"]-df["vol_acceleration"].mean())/
        df["vol_acceleration"].std()
    )

    return df


def aggregate_category(df):

    grouped = df.groupby("category").agg(
        count=("ticker","count"),
        avg_composite=("composite_score","mean"),
        avg_shock=("shock_score","mean")
    )

    return grouped.sort_values("avg_composite",ascending=False)


# ============================================================
# DASHBOARD FIXED (TALL HEATMAP)
# ============================================================

def create_dashboard(df, category_df, output):

    fig, axes = plt.subplots(
        2,2,
        figsize=(24,16),
        gridspec_kw={'height_ratios':[1,1.4]}
    )

    # Regime distribution
    regime_counts = df["long_regime"].value_counts()
    axes[0,0].bar(regime_counts.index, regime_counts.values)
    axes[0,0].set_title("Regime Distribution")
    axes[0,0].tick_params(axis='x',rotation=35)

    # Scatter
    axes[0,1].scatter(df["composite_score"], df["shock_score"])
    axes[0,1].set_title("Composite vs Shock")

    # Category strength
    axes[1,0].bar(category_df.index, category_df["avg_composite"])
    axes[1,0].set_title("Category Composite Strength")
    axes[1,0].tick_params(axis='x',rotation=60)

    # TALL HEATMAP FIX
    sns.heatmap(
        category_df[["avg_composite","avg_shock"]],
        cmap="RdYlGn",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=.5,
        ax=axes[1,1]
    )

    axes[1,1].set_title("Category Heatmap")

    plt.tight_layout(pad=4)
    plt.savefig(output/"institutional_dashboard.png",dpi=300,bbox_inches="tight")
    plt.close()


def run():

    output=create_output_folder()
    etfs=load_etfs(ETF_FILE)

    results=[analyze_etf(t) for t in etfs]
    results=[r for r in results if r]

    df=pd.DataFrame(results)

    df=add_scores(df)

    category_df=aggregate_category(df)

    create_dashboard(df,category_df,output)

    df.to_csv(output/"institutional_results.csv",index=False)


if __name__=="__main__":
    run()