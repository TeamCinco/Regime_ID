import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path
from timestral import TimestralForecaster

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_OUTPUT = REPO_ROOT / "output"

MODEL = "mistralai/Timestral-28M"
UNDERLYING = "SPY"
IMPLIED_VOL_INDEX = "^VIX"
PRICE_HISTORY_START = "2005-01-01"
BACKTEST_START = "2012-01-01"

TRADING_DAYS = 252
OPTION_HORIZON_DAYS = 21
CONTEXT_DAYS = 1000
ORIGIN_STEP_DAYS = 5
FORECAST_BATCH_SIZE = 64

LOW_QUANTILE = 0.1
MEDIAN_QUANTILE = 0.5
TAIL_QUANTILE = 0.9
MIN_PREMIUM_VOL_POINTS = 0.02

SELLING_RULES = {
    "always sell": None,
    "VIX > trailing 21d vol + 2 pts": ("trailing_realized_vol", MIN_PREMIUM_VOL_POINTS),
    "VIX > Timestral median + 2 pts": ("forecast_median_vol", MIN_PREMIUM_VOL_POINTS),
    "VIX > Timestral 90th pct": ("forecast_tail_vol", 0.0),
}


def run_spy_vix_backtest() -> None:
    closing_prices = fetch_closing_prices()
    realized_vol = rolling_realized_vol(closing_prices[UNDERLYING])
    origins = origin_positions(closing_prices.index)
    forecaster = TimestralForecaster.from_pretrained(MODEL)
    vol_forecasts = forecast_realized_vol(forecaster, realized_vol, origins)
    trades = build_trade_table(closing_prices, realized_vol, origins, vol_forecasts)
    rule_summary = summarize_rules(trades)
    calibration = check_calibration(trades)
    output_folder = save_results(trades, rule_summary)
    print_report(trades, rule_summary, calibration, output_folder)


def fetch_closing_prices() -> pd.DataFrame:
    tickers = [UNDERLYING, IMPLIED_VOL_INDEX]
    downloaded = yf.download(tickers, start=PRICE_HISTORY_START, auto_adjust=True, progress=False)
    closing_prices = downloaded["Close"].dropna()
    missing_tickers = set(tickers) - set(closing_prices.columns)
    if missing_tickers or closing_prices.empty:
        raise ValueError(f"Yahoo returned no closing prices for {sorted(missing_tickers) or tickers}")
    return closing_prices


def rolling_realized_vol(prices: pd.Series) -> pd.Series:
    daily_returns = prices.pct_change()
    return daily_returns.rolling(OPTION_HORIZON_DAYS).std() * np.sqrt(TRADING_DAYS)


def origin_positions(dates: pd.DatetimeIndex) -> list[int]:
    first_origin = max(dates.searchsorted(pd.Timestamp(BACKTEST_START)), CONTEXT_DAYS)
    last_origin = len(dates) - 1 - OPTION_HORIZON_DAYS
    return list(range(first_origin, last_origin + 1, ORIGIN_STEP_DAYS))


def forecast_realized_vol(forecaster: TimestralForecaster, realized_vol: pd.Series, origins: list[int]) -> pd.DataFrame:
    log_vol_history = np.log(realized_vol).to_numpy(dtype=np.float32)
    batches = [origins[start:start + FORECAST_BATCH_SIZE] for start in range(0, len(origins), FORECAST_BATCH_SIZE)]
    forecasts = [forecast_batch(forecaster, log_vol_history, batch) for batch in batches]
    return pd.concat(forecasts)


def forecast_batch(forecaster: TimestralForecaster, log_vol_history: np.ndarray, origins: list[int]) -> pd.DataFrame:
    contexts = np.stack([log_vol_history[origin - CONTEXT_DAYS + 1:origin + 1] for origin in origins])
    forecast = forecaster.forecast(contexts, prediction_length=OPTION_HORIZON_DAYS, clip_nonnegative=False)
    horizon_end = forecast.quantiles[:, 0, -1, :].cpu().numpy()
    levels = list(forecast.quantile_levels)
    return pd.DataFrame({
        "forecast_low_vol": quantile_vol(horizon_end, levels, LOW_QUANTILE),
        "forecast_median_vol": quantile_vol(horizon_end, levels, MEDIAN_QUANTILE),
        "forecast_tail_vol": quantile_vol(horizon_end, levels, TAIL_QUANTILE),
    }, index=origins)


def quantile_vol(horizon_end: np.ndarray, levels: list[float], quantile: float) -> np.ndarray:
    return np.exp(horizon_end[:, levels.index(quantile)])


def build_trade_table(
    closing_prices: pd.DataFrame, realized_vol: pd.Series, origins: list[int], vol_forecasts: pd.DataFrame
) -> pd.DataFrame:
    expiries = [origin + OPTION_HORIZON_DAYS for origin in origins]
    trades = vol_forecasts.copy()
    trades["date"] = closing_prices.index[origins]
    trades["implied_vol"] = closing_prices[IMPLIED_VOL_INDEX].iloc[origins].to_numpy() / 100
    trades["trailing_realized_vol"] = realized_vol.iloc[origins].to_numpy()
    trades["next_realized_vol"] = realized_vol.iloc[expiries].to_numpy()
    trades["vol_points_earned"] = trades["implied_vol"] - trades["next_realized_vol"]
    trades["variance_points_earned"] = trades["implied_vol"] ** 2 - trades["next_realized_vol"] ** 2
    return trades.set_index("date")


def summarize_rules(trades: pd.DataFrame) -> pd.DataFrame:
    rows = [summarize_rule(trades, name, rule) for name, rule in SELLING_RULES.items()]
    return pd.DataFrame(rows).set_index("rule").round(4)


def summarize_rule(trades: pd.DataFrame, name: str, rule: tuple[str, float] | None) -> dict:
    sold = trades[sells_on(trades, rule)]
    earned = sold["vol_points_earned"]
    return {
        "rule": name,
        "trades": len(sold),
        "share_of_weeks": len(sold) / len(trades),
        "win_rate": (earned > 0).mean(),
        "avg_vol_points": earned.mean(),
        "worst_vol_points": earned.min(),
        "variance_points_per_week": sold["variance_points_earned"].sum() / len(trades),
    }


def sells_on(trades: pd.DataFrame, rule: tuple[str, float] | None) -> pd.Series:
    if rule is None:
        return pd.Series(True, index=trades.index)
    threshold_column, premium = rule
    return trades["implied_vol"] > trades[threshold_column] + premium


def check_calibration(trades: pd.DataFrame) -> dict:
    realized = trades["next_realized_vol"]
    return {
        "inside Timestral 10-90% band (target 0.80)": realized.between(
            trades["forecast_low_vol"], trades["forecast_tail_vol"]
        ).mean(),
        "above Timestral 90th pct (target 0.10)": (realized > trades["forecast_tail_vol"]).mean(),
        "above VIX": (realized > trades["implied_vol"]).mean(),
    }


def save_results(trades: pd.DataFrame, rule_summary: pd.DataFrame) -> Path:
    output_folder = BASE_OUTPUT / datetime.now().strftime("%Y-%m-%d")
    output_folder.mkdir(parents=True, exist_ok=True)
    trades.to_csv(output_folder / "spy_vix_trades.csv")
    rule_summary.to_csv(output_folder / "spy_vix_summary.csv")
    return output_folder


def print_report(trades: pd.DataFrame, rule_summary: pd.DataFrame, calibration: dict, output_folder: Path) -> None:
    first_date = trades.index[0].date()
    last_date = trades.index[-1].date()
    print(f"\n{len(trades)} weekly {OPTION_HORIZON_DAYS}-day windows, {first_date} to {last_date}\n")
    print(rule_summary.to_string())
    print()
    for check, share in calibration.items():
        print(f"{check}: {share:.3f}")
    print(f"\nSaved to {output_folder}")


if __name__ == "__main__":
    run_spy_vix_backtest()
