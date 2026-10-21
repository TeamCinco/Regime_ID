import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent
EVENTS_CSV = REPO_ROOT / "bbg" / "migration_events.csv"
DEFAULT_DATA_WORKBOOK = REPO_ROOT / "bbg" / "bbg_migration_data.xlsx"
BASE_OUTPUT = REPO_ROOT / "output"
DATA_SHEET = "Data"
INPUTS_SHEET = "Inputs"
FIRST_TICKER_ROW = 11
BLOOMBERG_MISSING = "#N/A N/A"

PRICE_FIELD = "PX_LAST"
EV_TO_SALES_FIELD = "EV_TO_T12M_SALES"
FORWARD_PE_FIELD = "BEST_PE_RATIO"
CONSENSUS_EPS_FIELD = "BEST_EPS"
REPORTED_SALES_FIELD = "SALES_REV_TURN"
IMPLIED_VOL_FIELD = "30DAY_IMPVOL_100.0%MNY_DF"
FIELDS = [PRICE_FIELD, EV_TO_SALES_FIELD, FORWARD_PE_FIELD, CONSENSUS_EPS_FIELD, REPORTED_SALES_FIELD,
          IMPLIED_VOL_FIELD]

MARKET_BENCHMARK = "SPX Index"
SECTOR_BENCHMARKS = {
    "Information Technology": "S5INFT Index",
    "Health Care": "S5HLTH Index",
    "Financials": "S5FINL Index",
    "Consumer Discretionary": "S5COND Index",
    "Consumer Staples": "S5CONS Index",
    "Energy": "S5ENRS Index",
    "Industrials": "S5INDU Index",
    "Materials": "S5MATR Index",
    "Utilities": "S5UTIL Index",
    "Real Estate": "S5RLST Index",
    "Communication Services": "S5TELS Index",
}

WEEKS_BEFORE = 52
WEEKS_AFTER = 52
REVISION_WEEKS = 13
SALES_REPORT_LAG_WEEKS = 9
WEEKS_PER_YEAR = 52
WEEK_ENDING = "W-FRI"
REPORT_OFFSETS = [-52, -26, -13, 0, 13, 26, 52]
EVENT_OFFSETS = np.arange(-WEEKS_BEFORE, WEEKS_AFTER + 1)

REVISED_UP = "estimates revised up"
REVISED_DOWN = "estimates revised down"


def run_migration_event_study(data_workbook: Path) -> None:
    field_panels = read_field_panels(data_workbook)
    sector_benchmark_of = read_sector_benchmarks(data_workbook)
    events = read_events(field_panels[PRICE_FIELD])
    abnormal_returns = weekly_abnormal_returns(field_panels[PRICE_FIELD], sector_benchmark_of)
    events = describe_events(events, field_panels)
    paths = build_event_paths(events, field_panels, abnormal_returns)
    summary = summarize_paths(paths)
    output_folder = save_results(events, paths, summary)
    plot_abnormal_returns(paths, output_folder)
    print_report(events, summary, output_folder)


def read_field_panels(data_workbook: Path) -> dict[str, pd.DataFrame]:
    sheet = pd.read_excel(data_workbook, sheet_name=DATA_SHEET, header=None)
    date_columns = [column for column in sheet.columns if sheet.iat[1, column] == "Date"]
    blocks = {sheet.iat[0, column]: parse_block(sheet, column) for column in date_columns
              if is_filled_security(sheet.iat[0, column])}
    if not blocks:
        raise ValueError(f"No security blocks in {data_workbook} [{DATA_SHEET}]; refresh it on the terminal first")
    panels = {field: pd.DataFrame({security: block[field] for security, block in blocks.items()}) for field in FIELDS}
    return {field: on_weekly_calendar(panel) for field, panel in panels.items()}


def on_weekly_calendar(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.sort_index().resample(WEEK_ENDING).last().ffill()


def is_filled_security(security_cell) -> bool:
    return isinstance(security_cell, str) and security_cell.strip() != ""


def parse_block(sheet: pd.DataFrame, date_column: int) -> pd.DataFrame:
    headers = sheet.iloc[1, date_column:].tolist()
    width = block_width(headers)
    block = sheet.iloc[2:, date_column:date_column + width].replace(BLOOMBERG_MISSING, np.nan)
    block.columns = headers[:width]
    block = block.dropna(subset=["Date"]).set_index("Date")
    block.index = pd.to_datetime(block.index)
    return block.reindex(columns=FIELDS).apply(pd.to_numeric, errors="coerce")


def block_width(headers: list) -> int:
    blank_positions = [position for position, header in enumerate(headers) if pd.isna(header)]
    return blank_positions[0] if blank_positions else len(headers)


def read_sector_benchmarks(data_workbook: Path) -> dict[str, str]:
    inputs = pd.read_excel(data_workbook, sheet_name=INPUTS_SHEET, header=None, usecols=[0, 1])
    tickers = inputs.iloc[FIRST_TICKER_ROW - 1:].dropna(subset=[0])
    return {security: SECTOR_BENCHMARKS.get(sector, MARKET_BENCHMARK) for security, sector in tickers.values}


def read_events(prices: pd.DataFrame) -> pd.DataFrame:
    events = pd.read_csv(EVENTS_CSV, parse_dates=["event_date"])
    priced = events["security"].isin(prices.columns)
    if not priced.any():
        raise ValueError(f"None of the {len(events)} migrated stocks have prices in the data workbook")
    events = events[priced].reset_index(drop=True)
    events["event_week"] = prices.index.searchsorted(events["event_date"])
    return events[events["event_week"] < len(prices.index)].reset_index(drop=True)


def weekly_abnormal_returns(prices: pd.DataFrame, sector_benchmark_of: dict[str, str]) -> pd.DataFrame:
    weekly_returns = prices.pct_change(fill_method=None)
    stocks = [security for security in prices.columns if security in sector_benchmark_of]
    benchmark_returns = {stock: weekly_returns[benchmark_for(stock, sector_benchmark_of, prices)] for stock in stocks}
    return weekly_returns[stocks] - pd.DataFrame(benchmark_returns)


def benchmark_for(stock: str, sector_benchmark_of: dict[str, str], prices: pd.DataFrame) -> str:
    benchmark = sector_benchmark_of[stock]
    if benchmark not in prices.columns:
        raise ValueError(f"{stock} needs benchmark {benchmark}, which is missing from the data workbook")
    return benchmark


def describe_events(events: pd.DataFrame, field_panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    eps_revision = field_panels[CONSENSUS_EPS_FIELD].pct_change(REVISION_WEEKS, fill_method=None)
    reported_sales = field_panels[REPORTED_SALES_FIELD].shift(SALES_REPORT_LAG_WEEKS)
    sales_growth = reported_sales.pct_change(WEEKS_PER_YEAR, fill_method=None)
    sales_growth_change = sales_growth - sales_growth.shift(WEEKS_PER_YEAR // 2)
    described = events.copy()
    described["eps_revision_before"] = value_at_event(eps_revision, events)
    described["sales_growth"] = value_at_event(sales_growth, events)
    described["sales_growth_change_6m"] = value_at_event(sales_growth_change, events)
    described["revision_group"] = np.where(described["eps_revision_before"] >= 0, REVISED_UP, REVISED_DOWN)
    described.loc[described["eps_revision_before"].isna(), "revision_group"] = np.nan
    return described


def value_at_event(panel: pd.DataFrame, events: pd.DataFrame) -> np.ndarray:
    columns = panel.columns.get_indexer(events["security"])
    return panel.to_numpy()[events["event_week"].to_numpy(), columns]


def build_event_paths(
    events: pd.DataFrame, field_panels: dict[str, pd.DataFrame], abnormal_returns: pd.DataFrame
) -> pd.DataFrame:
    cumulative_abnormal = abnormal_returns.fillna(0).cumsum().where(abnormal_returns.notna())
    measures = {
        "abnormal_return": (cumulative_abnormal, subtract_event_value),
        "ev_to_sales_change": (field_panels[EV_TO_SALES_FIELD], ratio_to_event_value),
        "forward_pe_change": (field_panels[FORWARD_PE_FIELD], ratio_to_event_value),
        "implied_vol_change_pts": (field_panels[IMPLIED_VOL_FIELD], subtract_event_value),
    }
    paths = {name: event_time_matrix(panel, events, relative_to_event) for name, (panel, relative_to_event)
             in measures.items()}
    return stack_paths(paths, events)


def event_time_matrix(panel: pd.DataFrame, events: pd.DataFrame, relative_to_event) -> np.ndarray:
    values = panel.reindex(columns=events["security"].unique()).to_numpy()
    columns = pd.Index(events["security"].unique()).get_indexer(events["security"])
    rows = events["event_week"].to_numpy()[:, None] + EVENT_OFFSETS[None, :]
    inside = (rows >= 0) & (rows < len(values))
    window = np.where(inside, values[np.clip(rows, 0, len(values) - 1), columns[:, None]], np.nan)
    return relative_to_event(window, window[:, WEEKS_BEFORE][:, None])


def subtract_event_value(window: np.ndarray, event_value: np.ndarray) -> np.ndarray:
    return window - event_value


def ratio_to_event_value(window: np.ndarray, event_value: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(event_value > 0, window / event_value - 1, np.nan)


def stack_paths(paths: dict[str, np.ndarray], events: pd.DataFrame) -> pd.DataFrame:
    event_count = len(events)
    stacked = pd.DataFrame({
        "event_id": np.repeat(np.arange(event_count), len(EVENT_OFFSETS)),
        "week": np.tile(EVENT_OFFSETS, event_count),
    })
    for name, matrix in paths.items():
        stacked[name] = matrix.ravel()
    stacked["security"] = stacked["event_id"].map(events["security"])
    stacked["revision_group"] = stacked["event_id"].map(events["revision_group"])
    stacked["event_year"] = stacked["event_id"].map(events["event_date"].dt.year)
    return stacked


def summarize_paths(paths: pd.DataFrame) -> pd.DataFrame:
    at_report_weeks = paths[paths["week"].isin(REPORT_OFFSETS)]
    by_week = at_report_weeks.groupby("week")
    summary = pd.DataFrame({
        "events": by_week["abnormal_return"].count(),
        "abnormal_return": by_week["abnormal_return"].mean(),
        "years": by_week[["abnormal_return", "event_year"]].apply(years_with_data),
        "t_stat_by_year": by_week[["abnormal_return", "event_year"]].apply(t_stat_by_event_year),
        "ev_to_sales_change": by_week["ev_to_sales_change"].median(),
        "forward_pe_change": by_week["forward_pe_change"].median(),
        "implied_vol_change_pts": by_week["implied_vol_change_pts"].median(),
    })
    by_revision = at_report_weeks.pivot_table(index="week", columns="revision_group", values="abnormal_return")
    return summary.join(by_revision.add_prefix("abnormal_return, ")).round(4)


def years_with_data(week_paths: pd.DataFrame) -> int:
    return week_paths.dropna()["event_year"].nunique()


def t_stat_by_event_year(week_paths: pd.DataFrame) -> float:
    yearly_means = week_paths.dropna().groupby("event_year")["abnormal_return"].mean()
    return t_stat(yearly_means)


def t_stat(values: pd.Series) -> float:
    clean_values = values.dropna()
    if len(clean_values) < 2 or clean_values.std() == 0:
        return np.nan
    return clean_values.mean() / (clean_values.std() / np.sqrt(len(clean_values)))


def save_results(events: pd.DataFrame, paths: pd.DataFrame, summary: pd.DataFrame) -> Path:
    output_folder = BASE_OUTPUT / datetime.now().strftime("%Y-%m-%d")
    output_folder.mkdir(parents=True, exist_ok=True)
    events.to_csv(output_folder / "migration_events_described.csv", index=False)
    paths.to_csv(output_folder / "migration_event_paths.csv", index=False)
    summary.to_csv(output_folder / "migration_event_summary.csv")
    return output_folder


def plot_abnormal_returns(paths: pd.DataFrame, output_folder: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5))
    paths.groupby("week")["abnormal_return"].mean().plot(ax=axis, label="all migrations", linewidth=2)
    for group, group_paths in paths.dropna(subset=["revision_group"]).groupby("revision_group"):
        group_paths.groupby("week")["abnormal_return"].mean().plot(ax=axis, label=group, linestyle="--")
    axis.axvline(0, color="grey", linewidth=1)
    axis.axhline(0, color="grey", linewidth=1)
    axis.set_title("Return vs sector around Russell Growth -> Value migration")
    axis.set_xlabel("Weeks from migration")
    axis.set_ylabel("Cumulative abnormal return")
    axis.legend()
    figure.savefig(output_folder / "migration_event_study.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def print_report(events: pd.DataFrame, summary: pd.DataFrame, output_folder: Path) -> None:
    print(f"\n{len(events)} migrations, {events['security'].nunique()} stocks")
    print(f"Median sales growth at migration: {events['sales_growth'].median():.1%}, "
          f"6-month change: {events['sales_growth_change_6m'].median():+.1%}")
    print(f"\nWeeks from migration (0 = first snapshot outside Russell Growth):\n{summary.to_string()}")
    print(f"\nSaved results and migration_event_study.png to {output_folder}")


def data_workbook_from_command_line() -> Path:
    return Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA_WORKBOOK


if __name__ == "__main__":
    run_migration_event_study(data_workbook_from_command_line())
