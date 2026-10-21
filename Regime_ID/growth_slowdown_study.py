import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from migration_event_study import (
    EVENT_OFFSETS,
    PRICE_FIELD,
    REPORTED_SALES_FIELD,
    event_time_matrix,
    read_field_panels,
    read_sector_benchmarks,
    subtract_event_value,
    t_stat,
    weekly_abnormal_returns,
)

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_WORKBOOK = REPO_ROOT / "bbg" / "bbg_migration_data.xlsx"
EVENTS_CSV = REPO_ROOT / "bbg" / "migration_events.csv"
BASE_OUTPUT = REPO_ROOT / "output"

QUARTERS_PER_YEAR = 4
SALES_REPORT_LAG_WEEKS = 9
MIN_STARTING_GROWTH = 0.10
SLOWING_QUARTERS = 2
QUARTERS_BETWEEN_EVENTS = 4
REPORT_OFFSETS = [-26, -13, 0, 13, 26, 52]

SLOWDOWN = "sales growth slowing"
HOLDING_UP = "sales growth holding up"


def run_growth_slowdown_study(data_workbook: Path) -> None:
    field_panels = read_field_panels(data_workbook)
    sector_benchmark_of = read_sector_benchmarks(data_workbook)
    abnormal_returns = weekly_abnormal_returns(field_panels[PRICE_FIELD], sector_benchmark_of)
    quarters = reported_quarters(field_panels[REPORTED_SALES_FIELD], abnormal_returns.columns)
    events = classify_quarters(quarters)
    events = add_weeks_to_migration(events, field_panels[PRICE_FIELD].index)
    paths = build_paths(events, abnormal_returns)
    summary = summarize(paths)
    output_folder = save_results(events, summary)
    plot_paths(paths, output_folder)
    print_report(events, summary, output_folder)


def reported_quarters(reported_sales: pd.DataFrame, stocks: pd.Index) -> pd.DataFrame:
    quarters = [quarters_of(reported_sales[stock], stock) for stock in stocks if stock in reported_sales.columns]
    return pd.concat(quarters, ignore_index=True)


def quarters_of(sales: pd.Series, stock: str) -> pd.DataFrame:
    weekly_sales = sales.reset_index(drop=True)
    new_quarter_weeks = weekly_sales.index[weekly_sales.notna() & (weekly_sales.diff() != 0)]
    quarter_sales = weekly_sales.loc[new_quarter_weeks]
    return pd.DataFrame({
        "security": stock,
        "period_week": new_quarter_weeks,
        "report_week": new_quarter_weeks + SALES_REPORT_LAG_WEEKS,
        "sales_growth": (quarter_sales / quarter_sales.shift(QUARTERS_PER_YEAR) - 1).to_numpy(),
    })


def classify_quarters(quarters: pd.DataFrame) -> pd.DataFrame:
    by_stock = quarters.groupby("security")["sales_growth"]
    starting_growth = by_stock.shift(SLOWING_QUARTERS)
    slowed_every_quarter = pd.concat(
        [by_stock.shift(lag) < by_stock.shift(lag + 1) for lag in range(SLOWING_QUARTERS)], axis=1
    ).all(axis=1)
    was_growth_company = starting_growth >= MIN_STARTING_GROWTH
    classified = quarters.assign(starting_growth=starting_growth)[was_growth_company].copy()
    classified["group"] = np.where(slowed_every_quarter[was_growth_company], SLOWDOWN, HOLDING_UP)
    classified = classified[(classified["group"] == HOLDING_UP) | is_first_in_a_year(classified)]
    return classified.reset_index(drop=True)


def is_first_in_a_year(classified: pd.DataFrame) -> pd.Series:
    slowdowns = classified[classified["group"] == SLOWDOWN]
    keep = pd.Series(False, index=classified.index)
    for _, stock_slowdowns in slowdowns.groupby("security"):
        keep.loc[spaced_out(stock_slowdowns["report_week"])] = True
    return keep


def spaced_out(report_weeks: pd.Series) -> list:
    kept = []
    last_kept_week = -np.inf
    for index, week in report_weeks.items():
        if week - last_kept_week >= QUARTERS_BETWEEN_EVENTS * 13:
            kept.append(index)
            last_kept_week = week
    return kept


def add_weeks_to_migration(events: pd.DataFrame, weeks: pd.DatetimeIndex) -> pd.DataFrame:
    if not EVENTS_CSV.exists():
        return events.assign(weeks_to_migration=np.nan)
    migrations = pd.read_csv(EVENTS_CSV, parse_dates=["event_date"])
    migration_week = migrations.assign(week=weeks.searchsorted(migrations["event_date"]))
    first_migration_week = migration_week.groupby("security")["week"].min()
    weeks_to_migration = events["security"].map(first_migration_week) - events["report_week"]
    return events.assign(weeks_to_migration=weeks_to_migration)


def build_paths(events: pd.DataFrame, abnormal_returns: pd.DataFrame) -> pd.DataFrame:
    cumulative_abnormal = abnormal_returns.fillna(0).cumsum().where(abnormal_returns.notna())
    timed_events = events.rename(columns={"report_week": "event_week"})
    matrix = event_time_matrix(cumulative_abnormal, timed_events, subtract_event_value)
    paths = pd.DataFrame({
        "event_id": np.repeat(np.arange(len(events)), len(EVENT_OFFSETS)),
        "week": np.tile(EVENT_OFFSETS, len(events)),
        "abnormal_return": matrix.ravel(),
    })
    paths["group"] = paths["event_id"].map(events["group"])
    report_dates = abnormal_returns.index[np.clip(events["report_week"], 0, len(abnormal_returns) - 1)]
    paths["event_year"] = paths["event_id"].map(pd.Series(report_dates.year))
    return paths


def summarize(paths: pd.DataFrame) -> pd.DataFrame:
    at_report_weeks = paths[paths["week"].isin(REPORT_OFFSETS)].dropna(subset=["abnormal_return"])
    by_group = at_report_weeks.pivot_table(index="week", columns="group", values="abnormal_return", aggfunc="mean")
    counts = at_report_weeks.pivot_table(index="week", columns="group", values="abnormal_return", aggfunc="count")
    yearly = at_report_weeks.pivot_table(index=["week", "event_year"], columns="group", values="abnormal_return")
    yearly_gap = (yearly[SLOWDOWN] - yearly[HOLDING_UP]).groupby("week")
    summary = by_group.join(counts.add_suffix(" events"))
    summary["gap"] = summary[SLOWDOWN] - summary[HOLDING_UP]
    summary["gap_t_stat_by_year"] = yearly_gap.apply(t_stat)
    return summary.round(4)


def save_results(events: pd.DataFrame, summary: pd.DataFrame) -> Path:
    output_folder = BASE_OUTPUT / datetime.now().strftime("%Y-%m-%d")
    output_folder.mkdir(parents=True, exist_ok=True)
    events.to_csv(output_folder / "growth_slowdown_events.csv", index=False)
    summary.to_csv(output_folder / "growth_slowdown_summary.csv")
    return output_folder


def plot_paths(paths: pd.DataFrame, output_folder: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5))
    for group, group_paths in paths.groupby("group"):
        group_paths.groupby("week")["abnormal_return"].mean().plot(ax=axis, label=group, linewidth=2)
    axis.axvline(0, color="grey", linewidth=1)
    axis.axhline(0, color="grey", linewidth=1)
    axis.set_title("Return vs sector around the sales report")
    axis.set_xlabel("Weeks from report (period end + 9 weeks)")
    axis.set_ylabel("Cumulative abnormal return")
    axis.legend()
    figure.savefig(output_folder / "growth_slowdown_study.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def print_report(events: pd.DataFrame, summary: pd.DataFrame, output_folder: Path) -> None:
    slowdowns = events[events["group"] == SLOWDOWN]
    print(f"\n{len(slowdowns)} slowdowns ({slowdowns['security'].nunique()} stocks) vs "
          f"{(events['group'] == HOLDING_UP).sum()} quarters with growth holding up")
    print(f"Median weeks from slowdown report to Russell migration: {slowdowns['weeks_to_migration'].median():.0f}")
    print(f"\nWeeks from report:\n{summary.to_string()}")
    print(f"\nSaved results and growth_slowdown_study.png to {output_folder}")


def data_workbook_from_command_line() -> Path:
    return Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA_WORKBOOK


if __name__ == "__main__":
    run_growth_slowdown_study(data_workbook_from_command_line())
