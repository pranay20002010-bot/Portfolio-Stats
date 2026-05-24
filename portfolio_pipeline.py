from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, Optional

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"portfolio", "date", "ticker", "action", "quantity", "price"}

# Shared correlation heatmap colormap: green(-1) → yellow(0) → red(+1)
# Import matplotlib.colors lazily to avoid hard dependency at module load time.
def get_corr_cmap():
    import matplotlib.colors as mcolors
    return mcolors.LinearSegmentedColormap.from_list(
        "corr_cmap",
        [(0.0, "#2ecc71"), (0.5, "#f1c40f"), (1.0, "#e74c3c")],
    )

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
    # Simple pct_change on NAV — kept for backward compat.
    # NOTE: use build_twr_returns for accurate time-weighted returns.
    nav = nav_df.copy()
    for col in nav.columns:
        s = pd.to_numeric(nav[col], errors="coerce")
        idx = s[s > 0].index
        if len(idx) == 0:
            nav[col] = np.nan
            continue
        first = idx[0]
        s.loc[s.index < first] = np.nan
        nav[col] = s
    return nav.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")


def build_twr_returns(
    daily_positions: dict[str, pd.DataFrame],
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute daily Time-Weighted Returns (TWR) for each portfolio.

    For each day t:
        r_t = sum(holdings_{t-1} * price_t) / sum(holdings_{t-1} * price_{t-1}) - 1

    This uses yesterday's holdings with today's and yesterday's prices, so
    cash inflows/outflows (new buys/sells) do NOT inflate the return.
    Returns NaN for days where the prior-day portfolio value is zero.
    """
    result = pd.DataFrame(index=prices.index)

    for portfolio, holdings in daily_positions.items():
        available = [t for t in holdings.columns if t in prices.columns]
        if not available:
            result[portfolio] = np.nan
            continue

        h = holdings[available].reindex(prices.index).ffill().fillna(0)
        p = prices[available].reindex(prices.index)

        # Portfolio value using yesterday's weights (shift(1)) with today's prices
        prev_h = h.shift(1)
        nav_today    = (prev_h * p).sum(axis=1)
        nav_yesterday = (prev_h * p.shift(1)).sum(axis=1)

        # Avoid division by zero / tiny base
        twr = (nav_today / nav_yesterday.replace(0, np.nan) - 1)
        twr = twr.replace([np.inf, -np.inf], np.nan)

        # Zero out returns on transaction days where prior-day NAV was 0
        # (portfolio didn't exist yet — mark as NaN until first real position)
        first_valid = h[(h > 0).any(axis=1)].index
        if len(first_valid):
            twr.loc[twr.index < first_valid[0]] = np.nan

        # Clip extreme daily returns caused by large cash inflows on days where
        # prior-day NAV was tiny (e.g. first buy day).  Cap at ±75% per day —
        # no real equity moves that much in a single session.
        twr = twr.clip(lower=-0.75, upper=0.75)

        result[portfolio] = twr

    return result.dropna(how="all")


def annualized_return_from_twr(twr_returns: pd.Series) -> float:
    """
    Annualized return from a series of daily TWR returns.

    Uses actual calendar-day scaling:
        (prod(1+r_i))^(365 / calendar_days) - 1

    Falls back to trading-day scaling (252) if the index has no DatetimeIndex.
    """
    r = pd.to_numeric(twr_returns, errors="coerce").dropna()
    r = r.replace([np.inf, -np.inf], np.nan).dropna()
    n = len(r)
    if n < 2:
        return np.nan
    cumulative = float((1 + r).prod())
    if cumulative <= 0:
        return np.nan

    # Prefer calendar-day scaling when index carries dates
    if isinstance(r.index, pd.DatetimeIndex) and len(r.index) >= 2:
        cal_days = (r.index[-1] - r.index[0]).days
        if cal_days >= 2:
            return float(cumulative ** (365.0 / cal_days) - 1)

    # Fallback: trading-day scaling
    return float(cumulative ** (252.0 / n) - 1)


def annualized_return_from_nav(nav: pd.Series) -> float:
    """Legacy: kept for backward compat. Prefer annualized_return_from_twr."""
    nav = pd.to_numeric(nav, errors="coerce").dropna()
    nav = nav[nav > 0]
    if len(nav) < 2:
        return np.nan
    n = len(nav) - 1
    total = float(nav.iloc[-1] / nav.iloc[0] - 1)
    if (1 + total) <= 0 or n <= 0:
        return np.nan
    return float((1 + total) ** (252 / n) - 1)


def annualized_volatility(returns: pd.Series) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(252))


def sortino_ratio(returns: pd.Series, rf_annual: float = 0.0) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    r = r.replace([np.inf, -np.inf], np.nan).dropna()
    n = len(r)
    if n < 2:
        return np.nan
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    downside = r[r < rf_daily] - rf_daily
    if len(downside) < 2:
        return np.nan
    downside_dev = float(downside.std(ddof=1) * np.sqrt(252))
    if downside_dev <= 0:
        return np.nan
    ann_ret = annualized_return_from_twr(r)
    if pd.isna(ann_ret):
        return np.nan
    return float((ann_ret - rf_annual) / downside_dev)


def beta_to_benchmark(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    r = pd.to_numeric(returns, errors="coerce")
    b = pd.to_numeric(benchmark_returns, errors="coerce")
    df = pd.concat([r.rename("p"), b.rename("b")], axis=1).dropna()
    if len(df) < 3:
        return np.nan
    var_b = float(df["b"].var(ddof=1))
    if var_b <= 0:
        return np.nan
    cov = float(df["p"].cov(df["b"]))
    return float(cov / var_b)


def jensen_alpha(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    rf_annual: float = 0.0,
) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    b = pd.to_numeric(benchmark_returns, errors="coerce").dropna()
    if len(r) < 2 or len(b) < 2:
        return np.nan
    beta = beta_to_benchmark(r, b)
    if pd.isna(beta):
        return np.nan
    ann_p = annualized_return_from_twr(r)
    ann_b = annualized_return_from_twr(b)
    if pd.isna(ann_p) or pd.isna(ann_b):
        return np.nan
    return float(ann_p - (rf_annual + beta * (ann_b - rf_annual)))


def treynor_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    rf_annual: float = 0.0,
) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    b = pd.to_numeric(benchmark_returns, errors="coerce").dropna()
    if len(r) < 2 or len(b) < 2:
        return np.nan
    beta = beta_to_benchmark(r, b)
    if pd.isna(beta) or beta == 0:
        return np.nan
    ann_p = annualized_return_from_twr(r)
    if pd.isna(ann_p):
        return np.nan
    return float((ann_p - rf_annual) / beta)


def var_99(returns: pd.Series) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) < 10:
        return np.nan
    q = float(r.quantile(0.01))
    return float(-q)


def portfolio_metrics(
    nav: pd.Series,
    benchmark_nav: pd.Series | None = None,
    rf_annual: float = 0.0,
    twr_returns: pd.Series | None = None,
) -> pd.Series:
    """
    Compute portfolio metrics.

    If `twr_returns` is provided, all return/vol/ratio calculations use those
    time-weighted daily returns (correct when the portfolio has cash flows).
    Otherwise falls back to nav.pct_change() which over-estimates returns.
    """
    _empty = pd.Series(
        {
            "Annualized Return": np.nan,
            "Annualized Volatility": np.nan,
            "Sharpe": np.nan,
            "Sortino": np.nan,
            "Beta (Nifty 500)": np.nan,
            "Jensen Alpha": np.nan,
            "Treynor": np.nan,
            "VaR 99% (Daily)": np.nan,
            "Max Drawdown": np.nan,
        }
    )

    nav = pd.to_numeric(nav, errors="coerce").where(lambda s: s > 0).dropna()
    if len(nav) < 2:
        return _empty

    if twr_returns is not None:
        rets = pd.to_numeric(twr_returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    else:
        rets = nav.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    if len(rets) < 2:
        return _empty

    ann_ret = annualized_return_from_twr(rets)
    ann_vol = annualized_volatility(rets)
    sharpe = (ann_ret - rf_annual) / ann_vol if ann_vol and ann_vol > 0 and not pd.isna(ann_ret) else np.nan

    cumulative = (1 + rets).cumprod()
    dd = cumulative / cumulative.cummax() - 1
    max_dd = float(dd.min()) if not dd.empty else np.nan

    out = {
        "Annualized Return": ann_ret,
        "Annualized Volatility": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino_ratio(rets, rf_annual=rf_annual),
        "Beta (Nifty 500)": np.nan,
        "Jensen Alpha": np.nan,
        "Treynor": np.nan,
        "VaR 99% (Daily)": var_99(rets),
        "Max Drawdown": max_dd,
    }

    if benchmark_nav is not None:
        bnav = pd.to_numeric(benchmark_nav, errors="coerce").dropna()
        if len(bnav) >= 2:
            b_rets = bnav.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
            out["Beta (Nifty 500)"] = beta_to_benchmark(rets, b_rets)
            out["Jensen Alpha"] = jensen_alpha(rets, b_rets, rf_annual=rf_annual)
            out["Treynor"] = treynor_ratio(rets, b_rets, rf_annual=rf_annual)

    return pd.Series(out)


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
    # Backward-compat helper (kept), but prefer `portfolio_metrics(...)` for richer stats.
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


def compute_pnl(df: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute realized and unrealized P&L per portfolio using FIFO cost basis.

    Returns a DataFrame with columns:
        portfolio, realized_pnl, unrealized_pnl, total_pnl
    """
    results = []

    for portfolio in sorted(df["portfolio"].unique().tolist()):
        txns = df[df["portfolio"] == portfolio].sort_values("date").copy()
        # FIFO queue per ticker: list of (qty, cost_price) lots
        lots: dict[str, list[tuple[float, float]]] = {}
        realized = 0.0

        for _, row in txns.iterrows():
            ticker = str(row["ticker_yf"])
            qty = float(row["quantity"])
            price = float(row["price"])
            action = str(row["action"]).upper()

            if action == "BUY":
                lots.setdefault(ticker, []).append((qty, price))
            elif action == "SELL":
                remaining_sell = qty
                ticker_lots = lots.get(ticker, [])
                new_lots = []
                for lot_qty, lot_price in ticker_lots:
                    if remaining_sell <= 0:
                        new_lots.append((lot_qty, lot_price))
                        continue
                    if lot_qty <= remaining_sell:
                        realized += lot_qty * (price - lot_price)
                        remaining_sell -= lot_qty
                    else:
                        realized += remaining_sell * (price - lot_price)
                        new_lots.append((lot_qty - remaining_sell, lot_price))
                        remaining_sell = 0
                lots[ticker] = new_lots

        # Unrealized: latest price vs avg cost for open lots
        unrealized = 0.0
        last_date = prices.index[-1] if not prices.empty else None
        for ticker, ticker_lots in lots.items():
            if not ticker_lots:
                continue
            open_qty = sum(q for q, _ in ticker_lots)
            if open_qty <= 0:
                continue
            # Get latest market price
            mkt_price = None
            if ticker in prices.columns and last_date is not None:
                s = prices[ticker].dropna()
                if not s.empty:
                    mkt_price = float(s.iloc[-1])
            if mkt_price is None:
                continue
            avg_cost = sum(q * p for q, p in ticker_lots) / open_qty
            unrealized += open_qty * (mkt_price - avg_cost)

        results.append({
            "portfolio": portfolio,
            "realized_pnl": round(realized, 2),
            "unrealized_pnl": round(unrealized, 2),
            "total_pnl": round(realized + unrealized, 2),
        })

    return pd.DataFrame(results).set_index("portfolio")


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
