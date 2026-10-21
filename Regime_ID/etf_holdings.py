from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from regimeid2 import BASE_OUTPUT, LONG_LOOKBACK, SHORT_LOOKBACK

RESULTS_FILE_NAME = "institutional_results.csv"
EXCLUDED_CATEGORY_PREFIXES = (
    "Trading--",
    "Digital Assets",
    "Commodities",
    "Single Currency",
    "Bank Loan",
    "Derivative Income",
    "Unknown",
)
MIN_COMPOSITE_SCORE = 1.0
MIN_SECTOR_WEIGHT = 0.6
DIVERSIFIED = "diversified"
PRICE_HISTORY = "1y"
REPORT_ROWS = 25


@dataclass(frozen=True)
class FundProfile:
    ticker: str
    sector_weights: dict[str, float]
    top_holdings: pd.DataFrame

    @property
    def main_sector_weight(self) -> float:
        return max(self.sector_weights.values(), default=0.0)

    @property
    def main_sector(self) -> str:
        if self.main_sector_weight < MIN_SECTOR_WEIGHT:
            return DIVERSIFIED
        return max(self.sector_weights, key=self.sector_weights.get)


def run_etf_holdings() -> None:
    results_path = latest_results_path()
    regime_results = pd.read_csv(results_path)
    equity_etfs = keep_equity_etfs(regime_results)
    fund_profiles = fetch_fund_profiles(equity_etfs["ticker"])
    etf_sectors = tag_main_sectors(equity_etfs, fund_profiles)
    strong_sector_etfs = pick_strong_sector_etfs(etf_sectors)
    holdings = collect_holdings(strong_sector_etfs, fund_profiles)
    closing_prices = fetch_closing_prices(holdings["stock"].tolist())
    candidates = rank_laggards(holdings, closing_prices)
    save_results(results_path.parent, etf_sectors, candidates)
    print_report(equity_etfs, fund_profiles, strong_sector_etfs, candidates, results_path.parent)


def latest_results_path() -> Path:
    results_paths = sorted(BASE_OUTPUT.glob(f"*/{RESULTS_FILE_NAME}"))
    if not results_paths:
        raise FileNotFoundError(f"No {RESULTS_FILE_NAME} under {BASE_OUTPUT}; run regimeid2.py first")
    return results_paths[-1]


def keep_equity_etfs(regime_results: pd.DataFrame) -> pd.DataFrame:
    unique_etfs = regime_results.drop_duplicates("ticker")
    is_index = unique_etfs["ticker"].str.startswith("^")
    is_excluded_category = unique_etfs["category"].str.startswith(EXCLUDED_CATEGORY_PREFIXES)
    return unique_etfs[~is_index & ~is_excluded_category].reset_index(drop=True)


def fetch_fund_profiles(tickers: pd.Series) -> dict[str, FundProfile]:
    print(f"Fetching sector weights and holdings for {len(tickers)} equity ETFs...")
    profiles = {ticker: fetch_fund_profile(ticker) for ticker in tickers}
    return {ticker: profile for ticker, profile in profiles.items() if profile is not None}


def fetch_fund_profile(ticker: str) -> FundProfile | None:
    try:
        funds = yf.Ticker(ticker).funds_data
        sector_weights = dict(funds.sector_weightings)
        top_holdings = funds.top_holdings
    except Exception as error:
        print(f"  {ticker}: no fund data from Yahoo ({error})")
        return None
    return FundProfile(ticker, sector_weights, top_holdings)


def tag_main_sectors(equity_etfs: pd.DataFrame, fund_profiles: dict[str, FundProfile]) -> pd.DataFrame:
    main_sectors = {ticker: profile.main_sector for ticker, profile in fund_profiles.items()}
    main_sector_weights = {ticker: profile.main_sector_weight for ticker, profile in fund_profiles.items()}
    etf_sectors = equity_etfs.copy()
    etf_sectors["main_sector"] = etf_sectors["ticker"].map(main_sectors)
    etf_sectors["main_sector_weight"] = etf_sectors["ticker"].map(main_sector_weights).round(3)
    return etf_sectors


def pick_strong_sector_etfs(etf_sectors: pd.DataFrame) -> pd.DataFrame:
    is_strong = etf_sectors["composite_score"] >= MIN_COMPOSITE_SCORE
    is_sector_fund = etf_sectors["main_sector"].notna() & (etf_sectors["main_sector"] != DIVERSIFIED)
    strong_sector_etfs = etf_sectors[is_strong & is_sector_fund]
    if strong_sector_etfs.empty:
        raise ValueError(f"No sector ETFs with composite >= {MIN_COMPOSITE_SCORE}; lower MIN_COMPOSITE_SCORE")
    return strong_sector_etfs.sort_values("long_momentum_%", ascending=False)


def collect_holdings(strong_sector_etfs: pd.DataFrame, fund_profiles: dict[str, FundProfile]) -> pd.DataFrame:
    holdings_by_etf = [holdings_of(etf, fund_profiles[etf["ticker"]]) for _, etf in strong_sector_etfs.iterrows()]
    every_holding = pd.concat(holdings_by_etf, ignore_index=True)
    return merge_shared_holdings(every_holding)


def holdings_of(etf: pd.Series, profile: FundProfile) -> pd.DataFrame:
    holdings = profile.top_holdings.reset_index()
    return pd.DataFrame({
        "stock": holdings["Symbol"],
        "name": holdings["Name"],
        "holding_percent": holdings["Holding Percent"],
        "etf": etf["ticker"],
        "sector": etf["main_sector"],
        "etf_long_momentum_%": etf["long_momentum_%"],
    })


def merge_shared_holdings(every_holding: pd.DataFrame) -> pd.DataFrame:
    largest_first = every_holding.sort_values("holding_percent", ascending=False)
    held_by = largest_first.groupby("stock")["etf"].agg(", ".join)
    holdings = largest_first.drop_duplicates("stock").copy()
    holdings["held_by"] = holdings["stock"].map(held_by)
    return holdings.reset_index(drop=True)


def fetch_closing_prices(stocks: list[str]) -> pd.DataFrame:
    downloaded = yf.download(stocks, period=PRICE_HISTORY, auto_adjust=True, progress=False)
    closing_prices = downloaded["Close"].dropna(axis=1, how="all")
    if closing_prices.empty:
        raise ValueError(f"Yahoo returned no prices for {len(stocks)} holdings")
    return closing_prices


def rank_laggards(holdings: pd.DataFrame, closing_prices: pd.DataFrame) -> pd.DataFrame:
    candidates = holdings.copy()
    candidates["short_momentum_%"] = candidates["stock"].map(momentum_percent(closing_prices, SHORT_LOOKBACK))
    candidates["long_momentum_%"] = candidates["stock"].map(momentum_percent(closing_prices, LONG_LOOKBACK))
    candidates["lag_vs_etf_%"] = candidates["long_momentum_%"] - candidates["etf_long_momentum_%"]
    return candidates.sort_values("lag_vs_etf_%").round(2).reset_index(drop=True)


def momentum_percent(closing_prices: pd.DataFrame, lookback: int) -> pd.Series:
    return closing_prices.apply(trailing_change_percent, lookback=lookback)


def trailing_change_percent(prices: pd.Series, lookback: int) -> float:
    traded_prices = prices.dropna()
    if len(traded_prices) < lookback:
        return np.nan
    return (traded_prices.iloc[-1] / traded_prices.iloc[-lookback] - 1) * 100


def save_results(output_folder: Path, etf_sectors: pd.DataFrame, candidates: pd.DataFrame) -> None:
    etf_sectors.to_csv(output_folder / "etf_sectors.csv", index=False)
    candidates.to_csv(output_folder / "stock_candidates.csv", index=False)


def print_report(
    equity_etfs: pd.DataFrame,
    fund_profiles: dict[str, FundProfile],
    strong_sector_etfs: pd.DataFrame,
    candidates: pd.DataFrame,
    output_folder: Path,
) -> None:
    missing_profiles = sorted(set(equity_etfs["ticker"]) - set(fund_profiles))
    print(f"\n{len(equity_etfs)} equity ETFs kept, {len(missing_profiles)} without Yahoo fund data: {missing_profiles}")
    print(f"\nStrong sector ETFs (composite >= {MIN_COMPOSITE_SCORE}):")
    print(strong_sector_etfs[["ticker", "main_sector", "main_sector_weight", "long_momentum_%"]].to_string(index=False))
    print(f"\nBiggest laggards inside those ETFs (top {REPORT_ROWS}):")
    report_columns = ["stock", "name", "sector", "held_by", "long_momentum_%", "etf_long_momentum_%", "lag_vs_etf_%"]
    print(candidates[report_columns].head(REPORT_ROWS).to_string(index=False))
    print(f"\nSaved etf_sectors.csv and stock_candidates.csv to {output_folder}")


if __name__ == "__main__":
    run_etf_holdings()
