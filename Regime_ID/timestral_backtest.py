"""
TIMESTRAL VS SHOCK SCORE — WALK-FORWARD BACKTEST
================================================
Question: does Timestral (Mistral's time-series model) predict volatility
expansion better than the V2 shock score from regimeid2.py?

At each monthly origin date, using only data up to that date:
- shock      : V2 shock score (vol_ratio acceleration, z-scored across ETFs)
- naive      : today's 20d vol carried forward
- timestral  : Timestral forecast of the next 20d vol from the log-vol history
- timestral+ : same, with the daily composite score as a past covariate

Then it checks what really happened over the next 20 trading days:
- Vol error  : MAE of log(forecast vol) vs log(realized vol), lower is better
- Coverage   : share of realized vols inside Timestral's 10-90% band (target 0.80)
- AUC        : how well each signal ranks the ETFs that expanded
               (realized 20d vol > EXPANSION_MULTIPLE x trailing 100d vol), 0.5 = coin flip
- Rank IC    : mean per-date Spearman between signal and realized/long vol ratio

Requirements: pip install yfinance pandas numpy torch git+https://github.com/mistralai/timestral.git
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from timestral import TimestralForecaster
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
ETF_FILE = REPO_ROOT / "equity_etf_tickers.txt"
BASE_OUTPUT = REPO_ROOT / "output"

MODEL = "mistralai/Timestral-28M"
HISTORY = "6y"

SHORT_LOOKBACK = 30
LONG_LOOKBACK = 100
ACCEL_LOOKBACK = 20
TRADING_DAYS = 252

HORIZON = 20
CONTEXT = 1000
ORIGIN_STEP = 21
ORIGIN_COUNT = 24
EXPANSION_MULTIPLE = 1.25

SCORE_MAP = {
    "Bull Expansion": 2,
    "Bull Compression": 1,
    "Bear Compression": -1,
    "Bear Expansion": -2
}


def create_output_folder():
    path = BASE_OUTPUT / datetime.now().strftime("%Y-%m-%d")
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_etfs(filepath):
    with open(filepath, "r") as f:
        return [line.strip().upper() for line in f if line.strip()]


def download_closes(tickers):
    data = yf.download(
        tickers,
        period=HISTORY,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True
    )
    closes = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
    return closes.dropna(axis=1, how="all")


# ============================================================
# FEATURES (same definitions as regimeid2.py, computed every day)
# ============================================================

def annualized_vol(returns, window):
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)


def regime_scores(momentum, vol_ratio):
    bull = momentum > 0
    expanding = vol_ratio > 1
    scores = np.select(
        [bull & expanding, bull, expanding],
        [SCORE_MAP["Bull Expansion"], SCORE_MAP["Bull Compression"], SCORE_MAP["Bear Expansion"]],
        SCORE_MAP["Bear Compression"]
    )
    return pd.DataFrame(scores, index=momentum.index, columns=momentum.columns).where(momentum.notna())


def build_features(closes):
    returns = closes.pct_change(fill_method=None)
    short_vol = annualized_vol(returns, SHORT_LOOKBACK)
    long_vol = annualized_vol(returns, LONG_LOOKBACK)
    vol_ratio = short_vol / long_vol

    short_mom = closes / closes.shift(SHORT_LOOKBACK - 1) - 1
    long_mom = closes / closes.shift(LONG_LOOKBACK - 1) - 1
    composite = regime_scores(short_mom, vol_ratio) * 0.4 + regime_scores(long_mom, vol_ratio) * 0.6

    return {
        "horizon_vol": annualized_vol(returns, HORIZON),
        "long_vol": long_vol,
        "vol_ratio": vol_ratio,
        "vol_acceleration": vol_ratio - vol_ratio.shift(ACCEL_LOOKBACK - 1),
        "composite": composite
    }


# ============================================================
# TIMESTRAL FORECASTS
# ============================================================

def context_window(frame, origin):
    start = max(0, origin - CONTEXT + 1)
    return frame.iloc[start:origin + 1].T.to_numpy(dtype=np.float32)


def forecast_log_vol(forecaster, log_vol, origin, covariate=None):
    context = context_window(log_vol, origin)[:, None, :]
    past = None if covariate is None else context_window(covariate, origin)[:, None, :]
    forecast = forecaster.forecast(
        context,
        prediction_length=HORIZON,
        past_covariates=past,
        clip_nonnegative=False
    )
    last_step = forecast.quantiles[:, 0, -1, :].cpu().numpy()
    levels = list(forecast.quantile_levels)
    return {
        "median": last_step[:, levels.index(0.5)],
        "low": last_step[:, levels.index(0.1)],
        "high": last_step[:, levels.index(0.9)]
    }


# ============================================================
# WALK-FORWARD
# ============================================================

def origin_positions(dates):
    last = len(dates) - 1 - HORIZON
    return sorted(last - ORIGIN_STEP * i for i in range(ORIGIN_COUNT))


def evaluate_origin(forecaster, features, origin, dates):
    log_vol = np.log(features["horizon_vol"])
    plain = forecast_log_vol(forecaster, log_vol, origin)
    with_composite = forecast_log_vol(forecaster, log_vol, origin, features["composite"])

    rows = pd.DataFrame({
        "date": dates[origin],
        "ticker": log_vol.columns,
        "realized_vol": features["horizon_vol"].iloc[origin + HORIZON].to_numpy(),
        "long_vol": features["long_vol"].iloc[origin].to_numpy(),
        "naive_vol": features["horizon_vol"].iloc[origin].to_numpy(),
        "vol_acceleration": features["vol_acceleration"].iloc[origin].to_numpy(),
        "timestral_vol": np.exp(plain["median"]),
        "timestral_low": np.exp(plain["low"]),
        "timestral_high": np.exp(plain["high"]),
        "timestral_plus_vol": np.exp(with_composite["median"])
    })
    rows = rows.dropna(subset=["realized_vol", "long_vol", "naive_vol", "vol_acceleration"])
    rows = rows[(rows["realized_vol"] > 0) & (rows["naive_vol"] > 0)]

    accel = rows["vol_acceleration"]
    rows["shock_score"] = (accel - accel.mean()) / accel.std()
    rows["expanded"] = rows["realized_vol"] > EXPANSION_MULTIPLE * rows["long_vol"]
    return rows


def run_backtest(forecaster, features, dates):
    origins = origin_positions(dates)
    results = []
    for i, origin in enumerate(origins, 1):
        print(f"  origin {i}/{len(origins)}: {dates[origin].date()}")
        results.append(evaluate_origin(forecaster, features, origin, dates))
    return pd.concat(results, ignore_index=True)


# ============================================================
# SCORING
# ============================================================

def auc(signal, outcome):
    ranks = signal.rank()
    positives = outcome.sum()
    negatives = len(outcome) - positives
    if positives == 0 or negatives == 0:
        return np.nan
    return (ranks[outcome].sum() - positives * (positives + 1) / 2) / (positives * negatives)


def rank_ic(results, signal_column):
    by_date = results.groupby("date")
    return by_date.apply(lambda g: g[signal_column].corr(g["realized_ratio"], method="spearman")).mean()


def score(results):
    results = results.copy()
    results["realized_ratio"] = results["realized_vol"] / results["long_vol"]
    results["naive_signal"] = results["naive_vol"] / results["long_vol"]
    results["timestral_signal"] = results["timestral_vol"] / results["long_vol"]
    results["timestral_plus_signal"] = results["timestral_plus_vol"] / results["long_vol"]

    log_realized = np.log(results["realized_vol"])
    methods = [
        ("shock", "shock_score", None),
        ("naive", "naive_signal", "naive_vol"),
        ("timestral", "timestral_signal", "timestral_vol"),
        ("timestral+composite", "timestral_plus_signal", "timestral_plus_vol")
    ]

    summary = []
    for name, signal, level in methods:
        summary.append({
            "method": name,
            "vol_mae_log": np.nan if level is None else (np.log(results[level]) - log_realized).abs().mean(),
            "expansion_auc": auc(results[signal], results["expanded"]),
            "rank_ic": rank_ic(results, signal)
        })

    coverage = results["realized_vol"].between(results["timestral_low"], results["timestral_high"]).mean()
    return pd.DataFrame(summary).set_index("method").round(4), round(coverage, 3)


def run():
    output = create_output_folder()
    etfs = load_etfs(ETF_FILE)

    print(f"Downloading {len(etfs)} ETFs...")
    closes = download_closes(etfs)
    print(f"  {closes.shape[1]} ETFs with data, {len(closes)} days")

    features = build_features(closes)

    print(f"Loading {MODEL}...")
    forecaster = TimestralForecaster.from_pretrained(MODEL)

    print("Walk-forward backtest...")
    results = run_backtest(forecaster, features, closes.index)
    summary, coverage = score(results)

    results.to_csv(output / "timestral_backtest_rows.csv", index=False)
    summary.to_csv(output / "timestral_backtest_summary.csv")

    print(f"\n{len(results)} ETF-dates, {results['expanded'].mean():.1%} expanded "
          f"(next {HORIZON}d vol > {EXPANSION_MULTIPLE}x 100d vol)\n")
    print(summary.to_string())
    print(f"\nTimestral 10-90% band coverage: {coverage} (target 0.80)")
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    run()
