from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, Optional

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"portfolio", "date", "ticker", "action", "quantity", "price"}

TICKER_MAP: dict[str, str] = {
    # corrections
    "QPOWER": "QPOWER.NS",
    "APARINDS": "APARIND.NS",
    # explicit mappings
    "SAGILITY": "SAGILITY.NS",
    "TDPOWERSYS": "TDPOWERSYS.NS",
    # REIT fix
    "MINDSPACE-RR": "MINDSPACE-RR.NS",
}

BSE_OVERRIDE: dict[str, str] = {
    "MINDSPACE-RR": "MINDSPACE",
}

NS_TO_BSE_TICKER: dict[str, str] = {
    # Some instruments (especially ETFs) use numeric symbols on BSE.
    # MODEFENCE ETF: BSE symbol 590152 (per issuer factsheet).
    "MODEFENCE.NS": "590152.BO",
}

BENCHMARK_TICKER = "^CRSLDX"

SECTOR_MAP: dict[str, str] = {
    # Financials
    "ABCAPITAL.NS": "Financials",
    "FEDERALBNK.NS": "Financials",
    "SHRIRAMFIN.NS": "Financials",
    # Industrials / Capital Goods
    "LT.NS": "Industrials",
    "KPIL.NS": "Industrials",
    "ELECON.NS": "Industrials",
    "TRITURBINE.NS": "Industrials",
    "INOXWIND.NS": "Industrials",
    "GENUSPOWER.NS": "Industrials",
    "TDPOWERSYS.NS": "Industrials",
    "SHAKTIPUMP.NS": "Industrials",
    "WABAG.NS": "Industrials",
    "APLAPOLLO.NS": "Industrials",
    # Consumer
    "VBL.NS": "Consumer",
    "NESTLEIND.NS": "Consumer",
    "TRAVELFOOD.NS": "Consumer",
    # Real Estate / REIT
    "GODREJPROP.NS": "Real Estate",
    "MINDSPACE-RR.NS": "Real Estate",
    # Healthcare / Pharma
    "LAURUSLABS.NS": "Healthcare",
    "NAVINFLUOR.NS": "Chemicals",
    "EMCURE.NS": "Healthcare",
    "SAGILITY.NS": "Healthcare",
    # Chemicals / Materials
    "GRAVITA.NS": "Materials",
    "GALAXYSURF.NS": "Chemicals",
    "APARIND.NS": "Industrials",
    # Metals
    "NATIONALUM.NS": "Metals",
    # Technology / Electronics
    "DIXON.NS": "Technology",
    "NETWEB.NS": "Technology",
    # Defence
    "MODEFENCE.NS": "Defence",
    # FMCG / Agri
    "CCL.NS": "Consumer",
    # Energy / Renewables
    "WAAREEENER.NS": "Energy",
    # Misc
    "MANORAMA.NS": "Consumer",
    "NH.NS": "Healthcare",
    "ULTRACEMCO.NS": "Materials",
    "SAILIFE.NS": "Healthcare",
    "LGEINDIA.NS": "Consumer",
    "SETL.NS": "Industrials",
    "SCHAEFFLER.NS": "Industrials",
    "ENRIN.NS": "Energy",
    "QPOWER.NS": "Energy",
}


class TransactionsFormatError(ValueError):
    pass


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename: dict[str, str] = {}
    for col in df.columns:
        rename[col] = str(col).strip().lower()
    df = df.rename(columns=rename)

    alias = {
        "client": "portfolio",
        "name": "portfolio",
        "dt": "date",
        "transaction_date": "date",
        "qty": "quantity",
        "units": "quantity",
        "rate": "price",
        "amount": "price",
        "symbol": "ticker",
    }
    df = df.rename(columns={k: v for k, v in alias.items() if k in df.columns})
    return df


def load_transactions_excel(excel_bytes: bytes, password: Optional[str] = None) -> pd.DataFrame:
    try:
        return pd.read_excel(io.BytesIO(excel_bytes))
    except Exception:
        if not password:
            raise

    import msoffcrypto

    decrypted = io.BytesIO()
    office_file = msoffcrypto.OfficeFile(io.BytesIO(excel_bytes))
    office_file.load_key(password=password)
    office_file.decrypt(decrypted)
    decrypted.seek(0)
    return pd.read_excel(decrypted)


def prepare_transactions(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_columns(df_raw)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise TransactionsFormatError(
            f"Missing required columns: {sorted(missing)}. Found: {sorted(df.columns)}"
        )

    df = df[list(REQUIRED_COLUMNS)].copy()
    df["portfolio"] = df["portfolio"].astype(str).str.strip()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["action"] = df["action"].astype(str).str.strip().str.upper()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.normalize()

    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["quantity", "price"])

    df["signed_qty"] = np.where(df["action"] == "BUY", df["quantity"], -df["quantity"])

    df["ticker_yf"] = (
        df["ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
        .map(TICKER_MAP)
        .fillna(df["ticker"].astype(str).str.upper().str.strip() + ".NS")
    )

    return df.sort_values(["portfolio", "date", "ticker_yf"]).reset_index(drop=True)


def current_positions(df: pd.DataFrame) -> pd.DataFrame:
    positions = (
        df.groupby(["portfolio", "ticker_yf"], as_index=False)["signed_qty"]
        .sum()
        .rename(columns={"signed_qty": "qty"})
    )
    return positions[positions["qty"] > 0].sort_values(["portfolio", "qty"], ascending=[True, False]).reset_index(
        drop=True
    )


def bse_ticker_from_ns(ticker_ns: str) -> str:
    base = ticker_ns.replace(".NS", "")
    base = BSE_OVERRIDE.get(base, base)
    return f"{base}.BO"


def portfolio_returns(nav_df: pd.DataFrame) -> pd.DataFrame:
    # Compute returns after each portfolio has a valid starting NAV.
    nav = nav_df.copy()
    for col in nav.columns:
        s = pd.to_numeric(nav[col], errors="coerce")
        # Start from first strictly positive NAV to avoid divide-by-zero / inf.
        idx = s[s > 0].index
        if len(idx) == 0:
            nav[col] = np.nan
            continue
        first = idx[0]
        s.loc[s.index < first] = np.nan
        nav[col] = s
    return nav.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")


def portfolio_stats(returns: pd.Series) -> pd.Series:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    if returns.empty or len(returns) < 2:
        return pd.Series(
            {
                "Annual Return": np.nan,
                "Volatility": np.nan,
                "Sharpe Ratio": np.nan,
                "Max Drawdown": np.nan,
            }
        )

    # Use geometric annualization from cumulative return when possible.
    n = len(returns)
    cumulative = (1 + returns).cumprod()
    total_return = float(cumulative.iloc[-1] - 1)
    annual_return = float((1 + total_return) ** (252 / n) - 1) if n > 0 and (1 + total_return) > 0 else np.nan
    volatility = returns.std(ddof=1) * np.sqrt(252)
    sharpe = annual_return / volatility if volatility and volatility > 0 else np.nan
    rolling_max = cumulative.cummax()
    drawdown = cumulative / rolling_max - 1
    max_dd = drawdown.min()

    return pd.Series(
        {
            "Annual Return": annual_return,
            "Volatility": volatility,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_dd,
        }
    )


def stats_table(portfolio_returns_df: pd.DataFrame) -> pd.DataFrame:
    stats = portfolio_returns_df.apply(portfolio_stats).T
    if "Annual Return" in stats.columns:
        stats = stats.sort_values("Annual Return", ascending=False)
    return stats


@dataclass(frozen=True)
class DownloadResult:
    prices: pd.DataFrame
    successful_mappings: list[str]
    failed_tickers: list[str]


def download_prices(
    yf_module,
    tickers_ns: Iterable[str],
    start: datetime | date,
    include_benchmark: bool = True,
) -> DownloadResult:
    all_prices: list[pd.Series] = []
    successful: list[str] = []
    failed: list[str] = []

    def _extract_close(df: pd.DataFrame, symbol: str):
        if df is None or getattr(df, "empty", True):
            return None
        if "Close" in df.columns:
            close = df["Close"]
        elif isinstance(df.columns, pd.MultiIndex) and ("Close", symbol) in df.columns:
            close = df[("Close", symbol)]
        else:
            return None
        if isinstance(close, pd.DataFrame):
            close = close.squeeze()
        if np.isscalar(close):
            return None
        s = pd.Series(close)
        return s

    for ticker in tickers_ns:
        candidates: list[str] = []
        # Primary: BSE symbol (stable), with special-case overrides.
        candidates.append(NS_TO_BSE_TICKER.get(ticker, bse_ticker_from_ns(ticker)))
        # Fallback: original ticker (usually *.NS).
        candidates.append(ticker)

        got = None
        used = None
        for sym in candidates:
            try:
                temp = yf_module.download(sym, start=start, auto_adjust=True, progress=False)
                s = _extract_close(temp, sym)
                if s is None or s.dropna().empty:
                    continue
                got = s
                used = sym
                break
            except Exception:
                continue

        if got is None or used is None:
            failed.append(ticker)
            continue

        got.name = ticker
        all_prices.append(got)
        successful.append(f"{ticker} -> {used}")

    prices = pd.concat(all_prices, axis=1) if all_prices else pd.DataFrame()

    if include_benchmark:
        try:
            bench = yf_module.download(BENCHMARK_TICKER, start=start, auto_adjust=True, progress=False)
            if bench is not None and not bench.empty and "Close" in bench.columns:
                prices[BENCHMARK_TICKER] = bench["Close"]
        except Exception:
            pass

    return DownloadResult(prices=prices, successful_mappings=successful, failed_tickers=failed)


def build_daily_positions(df: pd.DataFrame, prices_index: pd.Index) -> dict[str, pd.DataFrame]:
    daily_positions: dict[str, pd.DataFrame] = {}

    for portfolio in sorted(df["portfolio"].unique().tolist()):
        temp = df[df["portfolio"] == portfolio].copy()
        txn = (
            temp.pivot_table(
                index="date",
                columns="ticker_yf",
                values="signed_qty",
                aggfunc="sum",
            )
            .fillna(0)
        )

        holdings = txn.cumsum()
        holdings = holdings.reindex(prices_index).ffill().fillna(0)
        daily_positions[str(portfolio)] = holdings

    return daily_positions


def build_nav_df(daily_positions: dict[str, pd.DataFrame], prices: pd.DataFrame) -> tuple[pd.DataFrame, set[str]]:
    nav_df = pd.DataFrame(index=prices.index)
    missing_tickers: set[str] = set()

    for portfolio, holdings in daily_positions.items():
        available = [t for t in holdings.columns if t in prices.columns]
        missing_tickers.update(set(holdings.columns) - set(available))
        if not available:
            nav_df[portfolio] = np.nan
            continue
        nav_df[portfolio] = (holdings[available] * prices[available]).sum(axis=1)

    return nav_df, missing_tickers


def drawdown_series(nav: pd.Series) -> pd.Series:
    nav = nav.dropna()
    if nav.empty:
        return pd.Series(dtype=float)
    rolling_max = nav.cummax()
    return nav / rolling_max - 1


def performance_attribution(holdings: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    """
    Simple attribution like the notebook:
    sum over time of (yesterday weights * today returns) per ticker.
    """
    tickers = [t for t in holdings.columns if t in prices.columns]
    if not tickers:
        return pd.Series(dtype=float)

    portfolio_prices = prices[tickers]
    stock_returns = portfolio_prices.pct_change().fillna(0)

    holdings_aligned = holdings[tickers].reindex(stock_returns.index).ffill()
    portfolio_value = (holdings_aligned * portfolio_prices).sum(axis=1)
    weights = holdings_aligned.multiply(portfolio_prices).div(portfolio_value, axis=0).fillna(0)

    contribution = (weights.shift(1) * stock_returns).sum().sort_values(ascending=False)
    return contribution


def overlap_matrix(positions_df: pd.DataFrame) -> pd.DataFrame:
    portfolios = sorted(positions_df["portfolio"].unique().tolist())
    om = pd.DataFrame(index=portfolios, columns=portfolios, dtype=float)

    sets: dict[str, set[str]] = {}
    for p in portfolios:
        sets[p] = set(positions_df[positions_df["portfolio"] == p]["ticker_yf"].dropna().astype(str).tolist())

    for p1 in portfolios:
        for p2 in portfolios:
            set1, set2 = sets[p1], sets[p2]
            denom = len(set1.union(set2))
            om.loc[p1, p2] = round((len(set1.intersection(set2)) / denom) * 100, 1) if denom else 0.0

    return om


def risk_contribution(holdings: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    """
    Risk contribution per ticker using annualized covariance matrix (like the notebook).
    Returns absolute contribution (not percentages).
    """
    tickers = [t for t in holdings.columns if t in prices.columns]
    if len(tickers) < 2:
        return pd.Series(dtype=float)

    stock_returns = prices[tickers].pct_change().dropna()
    if stock_returns.empty:
        return pd.Series(dtype=float)

    latest_holdings = holdings.iloc[-1]
    latest_prices = prices.loc[prices.index[-1], tickers]
    market_values = latest_holdings[tickers] * latest_prices
    if market_values.sum() == 0:
        return pd.Series(dtype=float)

    weights = market_values / market_values.sum()
    cov_matrix = stock_returns.cov() * 252

    portfolio_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))
    if not portfolio_vol or portfolio_vol <= 0:
        return pd.Series(dtype=float)

    marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
    rc = (weights * marginal_contrib)
    return pd.Series(rc, index=tickers).sort_values(ascending=False)


def categorize_market_caps_inr(
    market_caps_inr: pd.Series,
    *,
    large_crore: float = 100000,
    mid_crore: float = 30000,
) -> pd.Series:
    """
    Categorize tickers by market cap (INR).
    Defaults (crore INR): Large > 1,00,000; Mid 30,000–1,00,000; else Small.
    """
    caps = market_caps_inr.copy()
    caps = pd.to_numeric(caps, errors="coerce")
    large = large_crore * 1e7
    mid = mid_crore * 1e7

    def _cat(v: float) -> str:
        if pd.isna(v):
            return "Unknown"
        if v >= large:
            return "Large Cap"
        if v >= mid:
            return "Mid Cap"
        return "Small Cap"

    return caps.apply(_cat)


def cap_split_weights(
    holdings: pd.DataFrame,
    prices: pd.DataFrame,
    market_caps_inr: pd.Series,
    *,
    large_crore: float = 100000,
    mid_crore: float = 30000,
) -> pd.Series:
    """
    Returns weights by cap bucket using latest holdings and prices.
    """
    tickers = [t for t in holdings.columns if t in prices.columns]
    if not tickers:
        return pd.Series(dtype=float)

    latest_h = holdings.iloc[-1][tickers]
    latest_p = prices.loc[prices.index[-1], tickers]
    mv = (latest_h * latest_p).dropna()
    mv = mv[mv > 0]
    if mv.empty:
        return pd.Series(dtype=float)

    cats = categorize_market_caps_inr(
        market_caps_inr.reindex(mv.index),
        large_crore=large_crore,
        mid_crore=mid_crore,
    )
    split = mv.groupby(cats).sum()
    return (split / split.sum()).sort_values(ascending=False)
