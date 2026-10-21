import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from timestral import TimestralForecaster

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORKBOOK = REPO_ROOT / "Regime_ID" / "BBG_DATA_DATABDH.xlsx"
DATA_SHEET = "Data"
BASE_OUTPUT = REPO_ROOT / "output"

MODEL = "mistralai/Timestral-28M"
PRICE_FIELD = "PX_LAST"
IMPLIED_VOL_FIELD = "30DAY_IMPVOL_100.0%MNY_DF"
BLOOMBERG_MISSING = "#N/A N/A"

TRADING_DAYS = 252
OPTION_HORIZON_DAYS = 21
CONTEXT_DAYS = 1000
MIN_HISTORY_DAYS = 120
ORIGIN_STEP_DAYS = 5
FORECAST_BATCH_SIZE = 64

LOW_QUANTILE = 0.1
MEDIAN_QUANTILE = 0.5
TAIL_QUANTILE = 0.9
MIN_PREMIUM_VOL_POINTS = 0.02
SELLOFF_RETURN = -0.10

SELLING_RULES = {
    "always sell": [],
    "after selloff": ["after_selloff"],
    "IV > Timestral median + 2 pts": ["beats_median_forecast"],
    "IV > Timestral 90th pct": ["beats_tail_forecast"],
    "selloff + IV > Timestral median + 2 pts": ["after_selloff", "beats_median_forecast"],
    "selloff + IV > Timestral 90th pct": ["after_selloff", "beats_tail_forecast"],
}


def run_bbg_vol_backtest(workbook_path: Path) -> None:
    price_blocks = read_bloomberg_blocks(workbook_path)
    forecaster = TimestralForecaster.from_pretrained(MODEL)
    trades = pd.concat([build_ticker_trades(forecaster, ticker, block) for ticker, block in price_blocks.items()])
    rule_summary = summarize_rules_by_ticker(trades)
    calibration = check_calibration_by_ticker(trades)
    output_folder = save_results(trades, rule_summary, calibration)
    print_report(trades, rule_summary, calibration, output_folder)


def read_bloomberg_blocks(workbook_path: Path) -> dict[str, pd.DataFrame]:
    sheet = pd.read_excel(workbook_path, sheet_name=DATA_SHEET, header=None)
    block_columns = [column for column in sheet.columns if sheet.iat[1, column] == "Date"]
    filled_columns = [column for column in block_columns if is_filled_ticker(sheet.iat[0, column])]
    blocks = {sheet.iat[0, column]: parse_block(sheet, column) for column in filled_columns}
    if not blocks:
        raise ValueError(f"No ticker blocks with a 'Date' header in row 2 of {workbook_path} [{DATA_SHEET}]")
    return blocks


def is_filled_ticker(ticker_cell) -> bool:
    return isinstance(ticker_cell, str) and ticker_cell.strip() != ""


def parse_block(sheet: pd.DataFrame, date_column: int) -> pd.DataFrame:
    headers = sheet.iloc[1, date_column:].tolist()
    width = block_width(headers)
    block = sheet.iloc[2:, date_column:date_column + width].replace(BLOOMBERG_MISSING, np.nan)
    block.columns = headers[:width]
    block = block.dropna(subset=["Date"]).set_index("Date")
    missing_fields = {PRICE_FIELD, IMPLIED_VOL_FIELD} - set(block.columns)
    if missing_fields:
        raise ValueError(f"{sheet.iat[0, date_column]} block is missing {sorted(missing_fields)}")
    return block[[PRICE_FIELD, IMPLIED_VOL_FIELD]].astype(float)


def block_width(headers: list) -> int:
    blank_positions = [position for position, header in enumerate(headers) if pd.isna(header)]
    return blank_positions[0] if blank_positions else len(headers)


def build_ticker_trades(forecaster: TimestralForecaster, ticker: str, block: pd.DataFrame) -> pd.DataFrame:
    prices = block[PRICE_FIELD].dropna()
    realized_vol = rolling_realized_vol(prices)
    origins = origin_positions(len(prices))
    print(f"  {ticker}: {len(origins)} weekly windows")
    vol_forecasts = forecast_realized_vol(forecaster, realized_vol, origins)
    trades = build_trade_table(prices, block[IMPLIED_VOL_FIELD], realized_vol, origins, vol_forecasts)
    trades.insert(0, "ticker", ticker)
    return trades.dropna(subset=["implied_vol", "next_realized_vol", "trailing_realized_vol"])


def rolling_realized_vol(prices: pd.Series) -> pd.Series:
    daily_returns = prices.pct_change()
    return daily_returns.rolling(OPTION_HORIZON_DAYS).std() * np.sqrt(TRADING_DAYS)


def origin_positions(history_length: int) -> list[int]:
    last_origin = history_length - 1 - OPTION_HORIZON_DAYS
    return list(range(MIN_HISTORY_DAYS, last_origin + 1, ORIGIN_STEP_DAYS))


def forecast_realized_vol(forecaster: TimestralForecaster, realized_vol: pd.Series, origins: list[int]) -> pd.DataFrame:
    log_vol_history = np.log(realized_vol).to_numpy(dtype=np.float32)
    batches = [origins[start:start + FORECAST_BATCH_SIZE] for start in range(0, len(origins), FORECAST_BATCH_SIZE)]
    forecasts = [forecast_batch(forecaster, log_vol_history, batch) for batch in batches]
    return pd.concat(forecasts)


def forecast_batch(forecaster: TimestralForecaster, log_vol_history: np.ndarray, origins: list[int]) -> pd.DataFrame:
    contexts = np.stack([padded_context(log_vol_history, origin) for origin in origins])
    forecast = forecaster.forecast(contexts, prediction_length=OPTION_HORIZON_DAYS, clip_nonnegative=False)
    horizon_end = forecast.quantiles[:, 0, -1, :].cpu().numpy()
    levels = list(forecast.quantile_levels)
    return pd.DataFrame({
        "forecast_low_vol": quantile_vol(horizon_end, levels, LOW_QUANTILE),
        "forecast_median_vol": quantile_vol(horizon_end, levels, MEDIAN_QUANTILE),
        "forecast_tail_vol": quantile_vol(horizon_end, levels, TAIL_QUANTILE),
    }, index=origins)


def padded_context(log_vol_history: np.ndarray, origin: int) -> np.ndarray:
    available = log_vol_history[max(0, origin - CONTEXT_DAYS + 1):origin + 1]
    padding = np.full(CONTEXT_DAYS - len(available), np.nan, dtype=np.float32)
    return np.concatenate([padding, available])


def quantile_vol(horizon_end: np.ndarray, levels: list[float], quantile: float) -> np.ndarray:
    return np.exp(horizon_end[:, levels.index(quantile)])


def build_trade_table(
    prices: pd.Series, implied_vol: pd.Series, realized_vol: pd.Series, origins: list[int], vol_forecasts: pd.DataFrame
) -> pd.DataFrame:
    expiries = [origin + OPTION_HORIZON_DAYS for origin in origins]
    trades = vol_forecasts.copy()
    trades["date"] = prices.index[origins]
    trades["implied_vol"] = implied_vol.reindex(prices.index).iloc[origins].to_numpy() / 100
    trades["trailing_return"] = (prices / prices.shift(OPTION_HORIZON_DAYS) - 1).iloc[origins].to_numpy()
    trades["trailing_realized_vol"] = realized_vol.iloc[origins].to_numpy()
    trades["next_realized_vol"] = realized_vol.iloc[expiries].to_numpy()
    trades["vol_points_earned"] = trades["implied_vol"] - trades["next_realized_vol"]
    trades["variance_points_earned"] = trades["implied_vol"] ** 2 - trades["next_realized_vol"] ** 2
    add_rule_conditions(trades)
    return trades.set_index("date")


def add_rule_conditions(trades: pd.DataFrame) -> None:
    trades["after_selloff"] = trades["trailing_return"] <= SELLOFF_RETURN
    trades["beats_median_forecast"] = trades["implied_vol"] > trades["forecast_median_vol"] + MIN_PREMIUM_VOL_POINTS
    trades["beats_tail_forecast"] = trades["implied_vol"] > trades["forecast_tail_vol"]


def summarize_rules_by_ticker(trades: pd.DataFrame) -> pd.DataFrame:
    rows = [summarize_rule(ticker_trades, ticker, rule_name) for ticker, ticker_trades in trades.groupby("ticker")
            for rule_name in SELLING_RULES]
    return pd.DataFrame(rows).set_index(["ticker", "rule"]).round(4)


def summarize_rule(trades: pd.DataFrame, ticker: str, rule_name: str) -> dict:
    sold = trades[sells_on(trades, SELLING_RULES[rule_name])]
    earned = sold["vol_points_earned"]
    return {
        "ticker": ticker,
        "rule": rule_name,
        "trades": len(sold),
        "win_rate": (earned > 0).mean(),
        "avg_vol_points": earned.mean(),
        "worst_vol_points": earned.min(),
        "variance_points_per_week": sold["variance_points_earned"].sum() / len(trades),
    }


def sells_on(trades: pd.DataFrame, conditions: list[str]) -> pd.Series:
    every_week = pd.Series(True, index=trades.index)
    return trades[conditions].all(axis=1) if conditions else every_week


def check_calibration_by_ticker(trades: pd.DataFrame) -> pd.DataFrame:
    rows = [check_calibration(ticker_trades, ticker) for ticker, ticker_trades in trades.groupby("ticker")]
    return pd.DataFrame(rows).set_index("ticker").round(3)


def check_calibration(trades: pd.DataFrame, ticker: str) -> dict:
    realized = trades["next_realized_vol"]
    return {
        "ticker": ticker,
        "inside_10_90_band (target 0.80)": realized.between(
            trades["forecast_low_vol"], trades["forecast_tail_vol"]
        ).mean(),
        "above_90th_pct (target 0.10)": (realized > trades["forecast_tail_vol"]).mean(),
        "above_implied_vol": (realized > trades["implied_vol"]).mean(),
    }


def save_results(trades: pd.DataFrame, rule_summary: pd.DataFrame, calibration: pd.DataFrame) -> Path:
    output_folder = BASE_OUTPUT / datetime.now().strftime("%Y-%m-%d")
    output_folder.mkdir(parents=True, exist_ok=True)
    trades.to_csv(output_folder / "bbg_vol_trades.csv")
    rule_summary.to_csv(output_folder / "bbg_vol_summary.csv")
    calibration.to_csv(output_folder / "bbg_vol_calibration.csv")
    return output_folder


def print_report(
    trades: pd.DataFrame, rule_summary: pd.DataFrame, calibration: pd.DataFrame, output_folder: Path
) -> None:
    print(f"\n{len(trades)} weekly {OPTION_HORIZON_DAYS}-day windows across {trades['ticker'].nunique()} tickers\n")
    print(rule_summary.to_string())
    print(f"\n{calibration.to_string()}")
    print(f"\nSaved to {output_folder}")


def workbook_from_command_line() -> Path:
    return Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_WORKBOOK


if __name__ == "__main__":
    run_bbg_vol_backtest(workbook_from_command_line())
