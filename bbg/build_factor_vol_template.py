from datetime import date
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

OUTPUT_PATH = Path(__file__).resolve().parent / "bbg_factor_vol_data.xlsx"
TICKER_SLOTS = 50
FIRST_TICKER_ROW = 11
EXAMPLE_PEERS = [("SNDK", "Storage"), ("WDC", "Storage"), ("STX", "Storage"), ("MU", "Storage")]
DEFAULT_YELLOW_KEY = "US Equity"
START_DATE = date(2015, 1, 1)
FIELDS = [
    ("PX_LAST", "Price", "Last price"),
    ("30DAY_IMPVOL_100.0%MNY_DF", "Vol", "30-day at-the-money implied vol, in % points"),
    ("1M_PUT_IMP_VOL_25DELTA_DFLT", "Vol", "1-month 25-delta put implied vol (downside skew), in % points"),
    ("CUR_MKT_CAP", "Size", "Market cap, in millions"),
    ("BEST_PE_RATIO", "Value", "Forward P/E on consensus EPS (blended next 12 months)"),
    ("PE_RATIO", "Value", "Trailing P/E"),
    ("PX_TO_BOOK_RATIO", "Value", "Price to book"),
    ("EV_TO_T12M_EBITDA", "Value", "Enterprise value to trailing 12-month EBITDA"),
    ("RETURN_COM_EQY", "Quality", "Return on common equity, in %"),
    ("TOT_DEBT_TO_TOT_EQY", "Quality", "Total debt to total equity, in %"),
    ("BEST_EPS", "Estimates", "Consensus EPS (blended next 12 months); watch for revisions"),
]
STATIC_FIELDS = [("GICS_SECTOR_NAME", "GICS sector"), ("GICS_INDUSTRY_NAME", "GICS industry")]
BLOCK_WIDTH = len(FIELDS) + 2
BDH_OPTIONS = '"Dir=V","Days=T","Fill=P"'

BASE_FONT = Font(name="Arial", size=10)
BOLD_FONT = Font(name="Arial", size=10, bold=True)
TITLE_FONT = Font(name="Arial", size=14, bold=True)
INPUT_FONT = Font(name="Arial", size=10, color="0000FF")
LINK_FONT = Font(name="Arial", size=10, color="008000")
TICKER_LINK_FONT = Font(name="Arial", size=11, bold=True, color="008000")
INPUT_FILL = PatternFill("solid", start_color="FFFF00")
HEADER_FILL = PatternFill("solid", start_color="D9E1F2")


def build_workbook():
    workbook = Workbook()
    write_inputs_sheet(workbook.active)
    write_data_sheet(workbook.create_sheet("Data"))
    write_fields_sheet(workbook.create_sheet("Fields"))
    workbook.save(OUTPUT_PATH)


def write_inputs_sheet(sheet):
    sheet.title = "Inputs"
    write_cell(sheet, "A1", "Bloomberg factor vol data pull", TITLE_FONT)
    write_cell(sheet, "A3", "Start date", BOLD_FONT)
    write_input(sheet, "B3", START_DATE).number_format = "m/d/yyyy"
    write_cell(sheet, "A4", "End date (blank = today)", BOLD_FONT)
    write_input(sheet, "B4", None).number_format = "m/d/yyyy"
    write_cell(sheet, "A6", "Fields", BOLD_FONT)
    for offset, (field, _, _) in enumerate(FIELDS):
        write_input(sheet, cell_at(2 + offset, 6), field)
    write_legend(sheet)
    write_ticker_table(sheet)
    sheet.column_dimensions["A"].width = 26
    for offset in range(1, len(FIELDS) + 1):
        sheet.column_dimensions[get_column_letter(1 + offset)].width = 24


def write_legend(sheet):
    write_cell(sheet, "A8", "How to use", BOLD_FONT)
    lines = [
        "Type tickers in column A and an optional peer group in column D (yellow). Rows 11-14 are an example "
        "peer group; overwrite them.",
        "Set the dates in B3:B4, then Bloomberg tab > Refresh Workbook. Each ticker fills its own block on the "
        "Data tab. Fields in row 6 can be swapped (check FLDS <GO>).",
    ]
    for offset, line in enumerate(lines):
        write_cell(sheet, f"B{8 + offset}", line)


def write_ticker_table(sheet):
    headers = ["Ticker", "Yellow key", "Bloomberg security", "Peer group"]
    headers += [label for _, label in STATIC_FIELDS] + ["Data tab column"]
    for offset, header in enumerate(headers):
        write_cell(sheet, cell_at(1 + offset, 10), header, BOLD_FONT).fill = HEADER_FILL
    for slot in range(TICKER_SLOTS):
        write_ticker_row(sheet, slot)
    sheet.freeze_panes = "A11"


def write_ticker_row(sheet, slot):
    row = FIRST_TICKER_ROW + slot
    ticker, peer_group = EXAMPLE_PEERS[slot] if slot < len(EXAMPLE_PEERS) else (None, None)
    write_input(sheet, f"A{row}", ticker)
    write_input(sheet, f"B{row}", DEFAULT_YELLOW_KEY)
    write_cell(sheet, f"C{row}", f'=IF(A{row}="","",A{row}&" "&B{row})')
    write_input(sheet, f"D{row}", peer_group)
    for offset, (field, _) in enumerate(STATIC_FIELDS):
        write_cell(sheet, cell_at(5 + offset, row), f'=IF(C{row}="","",BDP(C{row},"{field}"))')
    write_cell(sheet, cell_at(5 + len(STATIC_FIELDS), row), get_column_letter(block_first_column(slot)))


def write_data_sheet(sheet):
    for slot in range(TICKER_SLOTS):
        write_data_block(sheet, slot)
    sheet.freeze_panes = "A4"


def write_data_block(sheet, slot):
    first_column = block_first_column(slot)
    security_cell = f"Inputs!$C${FIRST_TICKER_ROW + slot}"
    write_cell(sheet, cell_at(first_column, 1), f"={security_cell}", TICKER_LINK_FONT)
    write_block_headers(sheet, first_column)
    write_cell(sheet, cell_at(first_column, 3), bdh_formula(security_cell))
    set_block_widths(sheet, first_column)


def write_block_headers(sheet, first_column):
    write_cell(sheet, cell_at(first_column, 2), "Date", BOLD_FONT).fill = HEADER_FILL
    for offset in range(len(FIELDS)):
        field_cell = f"Inputs!{get_column_letter(2 + offset)}$6"
        header = write_cell(sheet, cell_at(first_column + 1 + offset, 2), f"={field_cell}", LINK_FONT)
        header.fill = HEADER_FILL
        header.alignment = Alignment(wrap_text=True)


def bdh_formula(security_cell):
    last_field_column = get_column_letter(1 + len(FIELDS))
    fields_range = f"Inputs!$B$6:${last_field_column}$6"
    end_date = 'IF(Inputs!$B$4="","",Inputs!$B$4)'
    bdh = f"BDH({security_cell},{fields_range},Inputs!$B$3,{end_date},{BDH_OPTIONS})"
    return f'=IF({security_cell}="","",{bdh})'


def set_block_widths(sheet, first_column):
    sheet.column_dimensions[get_column_letter(first_column)].width = 11
    for offset in range(1, len(FIELDS) + 1):
        sheet.column_dimensions[get_column_letter(first_column + offset)].width = 13
    sheet.column_dimensions[get_column_letter(first_column + len(FIELDS) + 1)].width = 3


def write_fields_sheet(sheet):
    for offset, header in enumerate(["Field", "Factor", "What it is"]):
        write_cell(sheet, cell_at(1 + offset, 1), header, BOLD_FONT).fill = HEADER_FILL
    for offset, (field, factor, meaning) in enumerate(FIELDS):
        write_cell(sheet, f"A{2 + offset}", field)
        write_cell(sheet, f"B{2 + offset}", factor)
        write_cell(sheet, f"C{2 + offset}", meaning)
    write_notes(sheet, 3 + len(FIELDS))
    sheet.column_dimensions["A"].width = 32
    sheet.column_dimensions["B"].width = 12
    sheet.column_dimensions["C"].width = 70


def write_notes(sheet, first_row):
    notes = [
        "Field names come from Bloomberg user forums and docs; confirm each in FLDS <GO> before backtesting.",
        "Vol fields are in percentage points (25 = 25%).",
        "GICS sector and industry (BDP) are today's classification, not point-in-time.",
        "Fundamental fields in BDH change on report dates; check FLDS for the as-of convention so the backtest "
        "only uses figures known at the time.",
        "BDH options: Dir=V (dates down), Days=T (trading days), Fill=P (carry last value over gaps).",
        "If a block stays blank after refresh, remove the IF( wrapper so the cell is just =BDH(...).",
    ]
    write_cell(sheet, f"A{first_row}", "Notes", BOLD_FONT)
    for offset, note in enumerate(notes):
        write_cell(sheet, f"A{first_row + 1 + offset}", note)


def block_first_column(slot):
    return 1 + slot * BLOCK_WIDTH


def cell_at(column, row):
    return f"{get_column_letter(column)}{row}"


def write_cell(sheet, address, value, font=BASE_FONT):
    cell = sheet[address]
    cell.value = value
    cell.font = font
    return cell


def write_input(sheet, address, value):
    cell = write_cell(sheet, address, value, INPUT_FONT)
    cell.fill = INPUT_FILL
    return cell


if __name__ == "__main__":
    build_workbook()
