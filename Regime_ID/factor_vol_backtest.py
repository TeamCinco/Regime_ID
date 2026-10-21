import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORKBOOK = REPO_ROOT / "Regime_ID" / "bbgdatadump.xlsx"
DATA_SHEET = "Data"
BASE_OUTPUT = REPO_ROOT / "output"
BLOOMBERG_MISSING = "#N/A N/A"

PEER_GROUPS = {
    "Health care": ["UNH", "BSX", "SYK", "EW", "IDXX", "LLY", "JNJ", "ABBV", "MRK"],
    "Energy": ["SLB", "WMB", "KMI", "XOM", "CVX", "COP"],
    "Insurance": ["PFG", "WTM", "MCY", "AIG", "MET", "RNR"],
    "Staples": ["PEP", "KO", "PG", "MDLZ"],
}

PRICE_FIELD = "PX_LAST"
IMPLIED_VOL_FIELD = "30DAY_IMPVOL_100.0%MNY_DF"
TRAILING_PE_FIELD = "PE_RATIO"
RETURN_ON_EQUITY_FIELD = "RETURN_COM_EQY"

TRADING_DAYS = 252
OPTION_HORIZON_DAYS = 21
LAG_LOOKBACK_DAYS = 63
VALUATION_HISTORY_DAYS = 756
MIN_VALUATION_HISTORY_DAYS = 504
REPORT_LAG_DAYS = 45
MIN_PRICED_SHARE = 0.9

LAGGING_PEER_PERCENTILE = 1 / 3
MIN_RETURN_ON_EQUITY = 10.0

DISLOCATED = "dislocated: lagging peers, cheap vs own history, ROE >= 10%"
FALLING_KNIFE = "lagging peers, not cheap or weak ROE"
NOT_LAGGING = "not lagging peers"


def run_factor_vol_backtest(workbook_path: Path) -> None:
    field_panels = read_field_panels(workbook_path)
    peer_group_of = peer_group_by_security(field_panels[PRICE_FIELD].columns)
    signals = build_signals(field_panels, peer_group_of)
    trades = sample_monthly_trades(signals)
    bucket_summary = summarize(trades, ["bucket"])
    group_summary = summarize(trades[trades["bucket"] == DISLOCATED], ["peer_group"])
    output_folder = save_results(trades, bucket_summary, group_summary)
    print_report(trades, bucket_summary, group_summary, output_folder)


def read_field_panels(workbook_path: Path) -> dict[str, pd.DataFrame]:
    sheet = pd.read_excel(workbook_path, sheet_name=DATA_SHEET, header=None)
    date_columns = [column for column in sheet.columns if sheet.iat[1, column] == "Date"]
    blocks = {sheet.iat[0, column]: parse_block(sheet, column) for column in date_columns
              if is_filled_security(sheet.iat[0, column])}
    if not blocks:
        raise ValueError(f"No ticker blocks with a 'Date' header in row 2 of {workbook_path} [{DATA_SHEET}]")
    return align_on_common_dates(blocks)


def is_filled_security(security_cell) -> bool:
    return isinstance(security_cell, str) and security_cell.strip() != ""


def parse_block(sheet: pd.DataFrame, date_column: int) -> pd.DataFrame:
    headers = sheet.iloc[1, date_column:].tolist()
    width = block_width(headers)
    block = sheet.iloc[2:, date_column:date_column + width].replace(BLOOMBERG_MISSING, np.nan)
    block.columns = headers[:width]
    block = block.dropna(subset=["Date"]).set_index("Date")
    required_fields = {PRICE_FIELD, IMPLIED_VOL_FIELD, TRAILING_PE_FIELD, RETURN_ON_EQUITY_FIELD}
    missing_fields = required_fields - set(block.columns)
    if missing_fields:
        raise ValueError(f"{sheet.iat[0, date_column]} block is missing {sorted(missing_fields)}")
    return block.astype(float)


def block_width(headers: list) -> int:
    blank_positions = [position for position, header in enumerate(headers) if pd.isna(header)]
    return blank_positions[0] if blank_positions else len(headers)


def align_on_common_dates(blocks: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    fields = [PRICE_FIELD, IMPLIED_VOL_FIELD, TRAILING_PE_FIELD, RETURN_ON_EQUITY_FIELD]
    panels = {field: pd.DataFrame({security: block[field] for security, block in blocks.items()}) for field in fields}
    priced_share = panels[PRICE_FIELD].notna().mean(axis=1)
    trading_dates = priced_share.index[priced_share >= MIN_PRICED_SHARE]
    return {field: panel.sort_index().ffill().loc[trading_dates] for field, panel in panels.items()}


def peer_group_by_security(securities: pd.Index) -> dict[str, str]:
    group_of_ticker = {ticker: group for group, tickers in PEER_GROUPS.items() for ticker in tickers}
    unmapped = [security for security in securities if security.split()[0] not in group_of_ticker]
    if unmapped:
        raise ValueError(f"Add these tickers to PEER_GROUPS: {unmapped}")
    return {security: group_of_ticker[security.split()[0]] for security in securities}


def build_signals(field_panels: dict[str, pd.DataFrame], peer_group_of: dict[str, str]) -> dict[str, pd.DataFrame]:
    prices = field_panels[PRICE_FIELD]
    realized_vol = rolling_realized_vol(prices)
    implied_vol = field_panels[IMPLIED_VOL_FIELD] / 100
    return {
        "implied_vol": implied_vol,
        "next_realized_vol": realized_vol.shift(-OPTION_HORIZON_DAYS),
        "trailing_return": prices / prices.shift(LAG_LOOKBACK_DAYS) - 1,
        "is_lagging_peers": is_lagging_peers(prices, peer_group_of),
        "is_cheap": is_cheap_vs_own_history(prices, field_panels[TRAILING_PE_FIELD]),
        "has_solid_returns": field_panels[RETURN_ON_EQUITY_FIELD].shift(REPORT_LAG_DAYS) >= MIN_RETURN_ON_EQUITY,
    }


def rolling_realized_vol(prices: pd.DataFrame) -> pd.DataFrame:
    daily_returns = prices.pct_change()
    return daily_returns.rolling(OPTION_HORIZON_DAYS).std() * np.sqrt(TRADING_DAYS)


def is_lagging_peers(prices: pd.DataFrame, peer_group_of: dict[str, str]) -> pd.DataFrame:
    trailing_return = prices / prices.shift(LAG_LOOKBACK_DAYS) - 1
    peer_percentile = trailing_return.T.groupby(peer_group_of).rank(pct=True).T
    return (peer_percentile <= LAGGING_PEER_PERCENTILE).where(trailing_return.notna())


def is_cheap_vs_own_history(prices: pd.DataFrame, trailing_pe: pd.DataFrame) -> pd.DataFrame:
    trailing_eps = (prices / trailing_pe).shift(REPORT_LAG_DAYS)
    reported_pe = (prices / trailing_eps).where(trailing_eps > 0)
    own_median_pe = reported_pe.rolling(VALUATION_HISTORY_DAYS, min_periods=MIN_VALUATION_HISTORY_DAYS).median()
    return (reported_pe < own_median_pe).where(own_median_pe.notna())


def sample_monthly_trades(signals: dict[str, pd.DataFrame]) -> pd.DataFrame:
    origin_dates = signals["implied_vol"].index[::OPTION_HORIZON_DAYS]
    stacked = {name: panel.loc[origin_dates].stack(future_stack=True) for name, panel in signals.items()}
    trades = pd.DataFrame(stacked).dropna()
    trades.index.names = ["date", "security"]
    trades = trades.reset_index()
    trades["peer_group"] = trades["security"].map(peer_group_by_security(trades["security"].unique()))
    return add_outcomes(trades)


def add_outcomes(trades: pd.DataFrame) -> pd.DataFrame:
    trades["vol_points_earned"] = trades["implied_vol"] - trades["next_realized_vol"]
    trades["variance_points_earned"] = trades["implied_vol"] ** 2 - trades["next_realized_vol"] ** 2
    peer_median_earned = trades.groupby(["date", "peer_group"])["vol_points_earned"].transform("median")
    trades["vol_points_vs_peers"] = trades["vol_points_earned"] - peer_median_earned
    trades["bucket"] = trades.apply(bucket_of, axis=1)
    return trades


def bucket_of(trade: pd.Series) -> str:
    if not trade["is_lagging_peers"]:
        return NOT_LAGGING
    if trade["is_cheap"] and trade["has_solid_returns"]:
        return DISLOCATED
    return FALLING_KNIFE


def summarize(trades: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    grouped = trades.groupby(group_columns)
    summary = pd.DataFrame({
        "trades": grouped.size(),
        "win_rate": grouped["vol_points_earned"].apply(share_positive),
        "avg_vol_points": grouped["vol_points_earned"].mean(),
        "avg_vol_points_vs_peers": grouped["vol_points_vs_peers"].mean(),
        "t_stat_vs_peers": grouped["vol_points_vs_peers"].apply(t_stat),
        "worst_vol_points": grouped["vol_points_earned"].min(),
        "avg_variance_points": grouped["variance_points_earned"].mean(),
    })
    return summary.round(4)


def share_positive(values: pd.Series) -> float:
    return (values > 0).mean()


def t_stat(values: pd.Series) -> float:
    if len(values) < 2:
        return np.nan
    return values.mean() / (values.std() / np.sqrt(len(values)))


def save_results(trades: pd.DataFrame, bucket_summary: pd.DataFrame, group_summary: pd.DataFrame) -> Path:
    output_folder = BASE_OUTPUT / datetime.now().strftime("%Y-%m-%d")
    output_folder.mkdir(parents=True, exist_ok=True)
    trades.to_csv(output_folder / "factor_vol_trades.csv", index=False)
    bucket_summary.to_csv(output_folder / "factor_vol_by_bucket.csv")
    group_summary.to_csv(output_folder / "factor_vol_dislocated_by_group.csv")
    return output_folder


def print_report(
    trades: pd.DataFrame, bucket_summary: pd.DataFrame, group_summary: pd.DataFrame, output_folder: Path
) -> None:
    first_date = trades["date"].min().date()
    last_date = trades["date"].max().date()
    print(f"\n{len(trades)} monthly stock-trades, {trades['security'].nunique()} stocks, {first_date} to {last_date}")
    print(f"Selling {OPTION_HORIZON_DAYS}-day at-the-money vol; fundamentals lagged {REPORT_LAG_DAYS} trading days\n")
    print(bucket_summary.to_string())
    print(f"\nDislocated names by peer group:\n{group_summary.to_string()}")
    print(f"\nSaved to {output_folder}")


def workbook_from_command_line() -> Path:
    return Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_WORKBOOK


if __name__ == "__main__":
    run_factor_vol_backtest(workbook_from_command_line())
