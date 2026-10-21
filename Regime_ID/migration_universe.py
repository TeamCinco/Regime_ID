import sys
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MEMBERS_WORKBOOK = REPO_ROOT / "bbg" / "bbg_russell_style_members.xlsx"
EVENTS_CSV = REPO_ROOT / "bbg" / "migration_events.csv"
MEMBERS_SHEET = "Members"
MEMBER_HEADER = "Member"

GROWTH_INDEX = "RLG Index"
VALUE_INDEX = "RLV Index"
EQUITY_YELLOW_KEY = "Equity"


def run_migration_universe(members_workbook: Path) -> None:
    memberships = read_memberships(members_workbook)
    migrations = find_growth_to_value_migrations(memberships)
    migrations.to_csv(EVENTS_CSV, index=False)
    print_report(memberships, migrations)


def read_memberships(members_workbook: Path) -> pd.DataFrame:
    sheet = pd.read_excel(members_workbook, sheet_name=MEMBERS_SHEET, header=None)
    header_columns = [column for column in sheet.columns if sheet.iat[2, column] == MEMBER_HEADER]
    snapshots = [read_snapshot(sheet, column) for column in header_columns]
    memberships = pd.concat(snapshots, ignore_index=True)
    if memberships.empty:
        raise ValueError(f"No members in {members_workbook} [{MEMBERS_SHEET}]; refresh it on the terminal first")
    return memberships


def read_snapshot(sheet: pd.DataFrame, member_column: int) -> pd.DataFrame:
    members = sheet.iloc[3:, member_column:member_column + 2].dropna()
    members.columns = ["member", "weight"]
    members = members[members["member"].map(is_bloomberg_member)]
    members["index"] = sheet.iat[0, member_column]
    members["snapshot_date"] = pd.to_datetime(sheet.iat[1, member_column])
    return members


def is_bloomberg_member(member_cell) -> bool:
    return isinstance(member_cell, str) and member_cell.strip() != "" and not member_cell.startswith("#N/A")


def find_growth_to_value_migrations(memberships: pd.DataFrame) -> pd.DataFrame:
    members_by_index_and_date = memberships.groupby(["index", "snapshot_date"])["member"].agg(set)
    snapshot_dates = sorted(memberships["snapshot_date"].unique())
    migrations = [
        migrations_between(members_by_index_and_date, previous_date, current_date)
        for previous_date, current_date in zip(snapshot_dates, snapshot_dates[1:])
    ]
    return pd.concat(migrations, ignore_index=True).sort_values(["event_date", "security"])


def migrations_between(members_by_index_and_date: pd.Series, previous_date, current_date) -> pd.DataFrame:
    previous_growth = members_by_index_and_date.get((GROWTH_INDEX, previous_date), set())
    current_growth = members_by_index_and_date.get((GROWTH_INDEX, current_date), set())
    current_value = members_by_index_and_date.get((VALUE_INDEX, current_date), set())
    migrated = sorted((previous_growth - current_growth) & current_value)
    return pd.DataFrame({
        "security": [f"{member} {EQUITY_YELLOW_KEY}" for member in migrated],
        "event_date": current_date,
        "last_growth_snapshot": previous_date,
    })


def print_report(memberships: pd.DataFrame, migrations: pd.DataFrame) -> None:
    snapshot_count = memberships["snapshot_date"].nunique()
    print(f"{len(memberships)} memberships across {snapshot_count} snapshots")
    print(f"{len(migrations)} Growth -> Value migrations, {migrations['security'].nunique()} stocks")
    print(migrations.groupby(migrations["event_date"].dt.year).size().to_string())
    print(f"\nSaved {EVENTS_CSV}; next run bbg/build_migration_data_template.py")


def members_workbook_from_command_line() -> Path:
    return Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MEMBERS_WORKBOOK


if __name__ == "__main__":
    run_migration_universe(members_workbook_from_command_line())
