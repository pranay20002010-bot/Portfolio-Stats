from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, Optional

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"portfolio", "date", "ticker", "action", "quantity", "price"}
REQUIRED_HOLDINGS_COLUMNS = {"portfolio", "startdate", "ticker", "quantity", "avgcostprice"}
REQUIRED_CASHFLOW_COLUMNS = {"portfolio", "date", "amount"}

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


def _normalize_columns_basic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lowercase/strip column names only — no alias remapping.

    Used for sheets (like Cashflows) whose columns must NOT go through the
    transaction-sheet alias map, since that map redirects "amount" -> "price"
    (meant for transaction sheets that call the trade value "Amount"), which
    would otherwise silently destroy the Cashflows "Amount" column.
    """
    df = df.copy()
    df = df.rename(columns={col: str(col).strip().lower() for col in df.columns})
    return df


def _find_sheet(sheets: dict[str, pd.DataFrame], *names: str) -> pd.DataFrame:
    """Case/whitespace-insensitive lookup of a worksheet by name."""
    lookup = {str(k).strip().lower(): k for k in sheets.keys()}
    for name in names:
        key = name.strip().lower()
        if key in lookup:
            return sheets[lookup[key]]
    raise TransactionsFormatError(
        f"Could not find a sheet named one of {list(names)}. "
        f"Found sheets: {list(sheets.keys())}"
    )


def load_transactions_excel(excel_bytes: bytes, password: Optional[str] = None) -> dict[str, pd.DataFrame]:
    """
    Load the full workbook (all sheets) as a dict {sheet_name: DataFrame}.

    The transactions master file has three sheets:
      - "Initial Holdings": positions each portfolio held before the tracked
        transaction history begins.
      - "Cashflows": deposits/withdrawals per portfolio.
      - "Transactions": the buy/sell trade log.
    """
    def _read(bio: io.BytesIO) -> dict[str, pd.DataFrame]:
        return pd.read_excel(bio, sheet_name=None)

    try:
        return _read(io.BytesIO(excel_bytes))
    except Exception:
        if not password:
            raise

    import msoffcrypto

    decrypted = io.BytesIO()
    office_file = msoffcrypto.OfficeFile(io.BytesIO(excel_bytes))
    office_file.load_key(password=password)
    office_file.decrypt(decrypted)
    decrypted.seek(0)
    return _read(decrypted)


def prepare_transactions(sheets: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the working transactions DataFrame and a separate cashflows DataFrame
    from the raw workbook sheets.

    "Initial Holdings" rows are converted into synthetic opening BUY transactions
    dated at each portfolio's StartDate, priced at AvgCostPrice. This lets FIFO
    cost-basis, daily-position, NAV, and attribution logic work correctly from
    each portfolio's true inception rather than only from its first *recorded*
    trade — without this, any position that predates the transaction log would
    be invisible to the rest of the pipeline (wrong holdings, wrong P&L, wrong
    NAV/returns).

    Returns:
        (transactions_df, cashflows_df)
    """
    holdings_raw = _find_sheet(sheets, "Initial Holdings", "InitialHoldings", "Holdings")
    txns_raw = _find_sheet(sheets, "Transactions")
    cash_raw = _find_sheet(sheets, "Cashflows", "Cash Flows", "CashFlows")

    # ---- Initial Holdings -> synthetic opening BUY transactions ----
    h = _normalize_columns(holdings_raw)
    missing_h = REQUIRED_HOLDINGS_COLUMNS - set(h.columns)
    if missing_h:
        raise TransactionsFormatError(
            f"'Initial Holdings' sheet missing columns: {sorted(missing_h)}. Found: {sorted(h.columns)}"
        )
    h_opening = pd.DataFrame(
        {
            "portfolio": h["portfolio"],
            "date": h["startdate"],
            "ticker": h["ticker"],
            "action": "BUY",
            "quantity": h["quantity"],
            "price": h["avgcostprice"],
        }
    )

    # ---- Transactions ----
    t = _normalize_columns(txns_raw)
    missing_t = REQUIRED_COLUMNS - set(t.columns)
    if missing_t:
        raise TransactionsFormatError(
            f"'Transactions' sheet missing columns: {sorted(missing_t)}. Found: {sorted(t.columns)}"
        )
    t = t[list(REQUIRED_COLUMNS)].copy()

    combined = pd.concat([h_opening, t], ignore_index=True, sort=False)

    combined["portfolio"] = combined["portfolio"].astype(str).str.strip()
    combined["ticker"] = combined["ticker"].astype(str).str.strip().str.upper()
    combined["action"] = combined["action"].astype(str).str.strip().str.upper()

    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.dropna(subset=["date"])
    combined["date"] = combined["date"].dt.normalize()

    combined["quantity"] = pd.to_numeric(combined["quantity"], errors="coerce")
    combined["price"] = pd.to_numeric(combined["price"], errors="coerce")
    combined = combined.dropna(subset=["quantity", "price"])

    combined["signed_qty"] = np.where(combined["action"] == "BUY", combined["quantity"], -combined["quantity"])

    combined["ticker_yf"] = (
        combined["ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
        .map(TICKER_MAP)
        .fillna(combined["ticker"].astype(str).str.upper().str.strip() + ".NS")
    )

    combined = combined.sort_values(["portfolio", "date", "ticker_yf"]).reset_index(drop=True)

    # ---- Cashflows ----
    c = _normalize_columns_basic(cash_raw)
    missing_c = REQUIRED_CASHFLOW_COLUMNS - set(c.columns)
    if missing_c:
        raise TransactionsFormatError(
            f"'Cashflows' sheet missing columns: {sorted(missing_c)}. Found: {sorted(c.columns)}"
        )
    c = c[list(REQUIRED_CASHFLOW_COLUMNS)].copy()
    c["portfolio"] = c["portfolio"].astype(str).str.strip()
    c["date"] = pd.to_datetime(c["date"], errors="coerce")
    c = c.dropna(subset=["date"])
    c["date"] = c["date"].dt.normalize()
    c["amount"] = pd.to_numeric(c["amount"], errors="coerce")
    c = c.dropna(subset=["amount"])
    c = c.sort_values(["portfolio", "date"]).reset_index(drop=True)

    return combined, c


def cashflow_summary(cashflows_df: pd.DataFrame) -> pd.Series:
    """Total net cash contributed per portfolio (sum of Cashflows.Amount)."""
    if cashflows_df is None or cashflows_df.empty:
        return pd.Series(dtype=float)
    return cashflows_df.groupby("portfolio")["amount"].sum().sort_values(ascending=False)


def current_positions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return open positions per portfolio with qty and average buy price (FIFO).
    Only tickers with net qty > 0 (still held) are returned.
    """
    rows = []
    for portfolio in sorted(df["portfolio"].unique()):
        txns = df[df["portfolio"] == portfolio].sort_values("date")
        # FIFO lots: ticker -> [(qty, price), ...]
        lots: dict[str, list[tuple[float, float]]] = {}
        for _, row in txns.iterrows():
            ticker = str(row["ticker_yf"])
            qty    = float(row["quantity"])
            price  = float(row["price"])
            action = str(row["action"]).upper()
            if action == "BUY":
                lots.setdefault(ticker, []).append((qty, price))
            elif action == "SELL":
                remaining = qty
                old_lots  = lots.get(ticker, [])
                new_lots  = []
                for lq, lp in old_lots:
                    if remaining <= 0:
                        new_lots.append((lq, lp))
                    elif lq <= remaining:
                        remaining -= lq          # lot fully consumed
                    else:
                        new_lots.append((lq - remaining, lp))
                        remaining = 0
                lots[ticker] = new_lots

        for ticker, ticker_lots in lots.items():
            open_qty = sum(q for q, _ in ticker_lots)
            if open_qty <= 0:
                continue
            avg_price = sum(q * p for q, p in ticker_lots) / open_qty
            rows.append({
                "portfolio":   portfolio,
                "ticker_yf":   ticker,
                "qty":         round(open_qty, 4),
                "avg_buy_price": round(avg_price, 2),
            })

    if not rows:
        return pd.DataFrame(columns=["portfolio", "ticker_yf", "qty", "avg_buy_price"])

    out = pd.DataFrame(rows)
    return out.sort_values(["portfolio", "qty"], ascending=[True, False]).reset_index(drop=True)


def bse_ticker_from_ns(ticker_ns: str) -> str:
    base = ticker_ns.replace(".NS", "")
    base = BSE_OVERRIDE.get(base, base)
    return f"{base}.BO"


def build_twr_returns(
    daily_positions: dict[str, pd.DataFrame],
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute daily Time-Weighted Returns (TWR) for each portfolio.

    For each day t:
        r_t = sum(holdings_{t-1} * price_t) / sum(holdings_{t-1} * price_{t-1}) - 1

    This uses YESTERDAY'S holdings with today's and yesterday's prices so that
    cash inflows/outflows (new buys/rebalancing sells) do NOT inflate the return.

    Days that are invalid for return measurement are set to NaN:
      - Days before the portfolio's first real position
      - Days where yesterday's NAV was below 2% of the portfolio's median NAV
        (these are rebalancing/entry days where the portfolio was temporarily
        fully-or-mostly-cash and the ratio produces nonsense)
    After filtering, returns are capped at ±25% per day — the hard limit for
    a diversified equity portfolio in a single session.
    """
    result = pd.DataFrame(index=prices.index)

    for portfolio, holdings in daily_positions.items():
        available = [t for t in holdings.columns if t in prices.columns]
        if not available:
            result[portfolio] = np.nan
            continue

        h = holdings[available].reindex(prices.index).ffill().fillna(0)
        p = prices[available].reindex(prices.index)

        prev_h = h.shift(1)
        nav_today     = (prev_h * p).sum(axis=1)
        nav_yesterday = (prev_h * p.shift(1)).sum(axis=1)

        # NaN out days where prior-day NAV was zero or below 2% of median NAV.
        # This handles full rebalancing days (sell-all → rebuy) where the prior
        # holding is near zero and produces a meaningless giant return.
        median_nav = nav_yesterday[nav_yesterday > 0].median()
        min_nav_threshold = median_nav * 0.02 if pd.notna(median_nav) and median_nav > 0 else 0
        valid_base = nav_yesterday.where(nav_yesterday >= min_nav_threshold)

        twr = (nav_today / valid_base - 1)
        twr = twr.replace([np.inf, -np.inf], np.nan)

        # NaN out everything before the first day the portfolio had real holdings
        first_valid = h[(h > 0).any(axis=1)].index
        if len(first_valid):
            twr.loc[twr.index < first_valid[0]] = np.nan

        # Hard cap: no equity portfolio moves more than ±25% in a single day
        twr = twr.clip(lower=-0.25, upper=0.25)

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


def cumulative_growth_from_twr(twr_returns: pd.Series) -> pd.Series:
    """
    Growth-of-₹1 index built purely from daily TWR returns (no NAV/dollar value
    involved). Used for max-drawdown and the PDF growth chart. Because it's
    built from TWR (which already excludes cash-flow effects), a drawdown
    computed off this curve reflects actual investment performance — unlike a
    drawdown computed off raw NAV, which would show a fake "loss" whenever the
    client simply withdrew cash.
    """
    r = pd.to_numeric(twr_returns, errors="coerce").dropna()
    if r.empty:
        return pd.Series(dtype=float)
    return (1 + r).cumprod()


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


def cvar_99(returns: pd.Series) -> float:
    """
    99% Conditional Value at Risk (Expected Shortfall).
    Average of all returns below the 1st percentile, sign-flipped to positive.
    More informative than VaR — measures the expected loss *given* we're in the tail.
    """
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) < 10:
        return np.nan
    cutoff = float(r.quantile(0.01))
    tail = r[r <= cutoff]
    if tail.empty:
        return np.nan
    return float(-tail.mean())


def portfolio_metrics(
    twr_returns: pd.Series,
    benchmark_prices: pd.Series | None = None,
    rf_annual: float = 0.0,
) -> pd.Series:
    """
    Compute portfolio metrics purely from daily Time-Weighted Returns.

    No NAV/dollar-value series is used anywhere here — annualized return,
    volatility, Sharpe, Sortino, beta/alpha/Treynor, CVaR, and max drawdown
    are all derived from `twr_returns` (and, for beta/alpha/Treynor, from
    `benchmark_prices` converted to its own daily returns). This keeps every
    metric immune to cash inflows/outflows, since TWR already excludes them.
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
            "CVaR 99% (Daily)": np.nan,
            "Max Drawdown": np.nan,
        }
    )

    rets = pd.to_numeric(twr_returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(rets) < 2:
        return _empty

    ann_ret = annualized_return_from_twr(rets)
    ann_vol = annualized_volatility(rets)
    sharpe = (ann_ret - rf_annual) / ann_vol if ann_vol and ann_vol > 0 and not pd.isna(ann_ret) else np.nan

    # Max drawdown: from the TWR growth-of-₹1 curve, not raw NAV — a cash
    # withdrawal should never show up as a "drawdown".
    growth = cumulative_growth_from_twr(rets)
    if len(growth) >= 2:
        dd_series = drawdown_series(growth)
        max_dd = float(dd_series.min()) if not dd_series.empty else np.nan
    else:
        max_dd = np.nan

    out = {
        "Annualized Return": ann_ret,
        "Annualized Volatility": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino_ratio(rets, rf_annual=rf_annual),
        "Beta (Nifty 500)": np.nan,
        "Jensen Alpha": np.nan,
        "Treynor": np.nan,
        "CVaR 99% (Daily)": cvar_99(rets),
        "Max Drawdown": max_dd,
    }

    if benchmark_prices is not None:
        bprices = pd.to_numeric(benchmark_prices, errors="coerce").dropna()
        if len(bprices) >= 2:
            b_rets = bprices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
            out["Beta (Nifty 500)"] = beta_to_benchmark(rets, b_rets)
            out["Jensen Alpha"] = jensen_alpha(rets, b_rets, rf_annual=rf_annual)
            out["Treynor"] = treynor_ratio(rets, b_rets, rf_annual=rf_annual)

    return pd.Series(out)


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


def build_daily_positions(
    df: pd.DataFrame, prices_index: pd.Index
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Returns (daily_positions, oversold_warnings).

    daily_positions[portfolio] is a date x ticker_yf quantity DataFrame. This is
    a long-only equity tracker, so a SELL that exceeds the quantity actually on
    record (e.g. a stock sold that was never recorded as bought — missing from
    both "Initial Holdings" and prior "Transactions" rows) must never be allowed
    to push the running position negative.

    The clamp is applied SEQUENTIALLY, per ticker, transaction by transaction —
    not as a single vectorized clip of the cumulative sum. Those are NOT the
    same thing: if a ticker is oversold and later bought again, clipping the
    raw cumulative sum after the fact silently loses part of that later buy
    (e.g. buy 10, oversell 15 [only 10 available], buy 20 more -> the correct
    running position is 20, but clip(cumsum, lower=0) on [10, -5, 15] gives 15).
    A sequential clamp — where each SELL is capped at whatever is actually
    held at that moment, and the running position floors at zero right there
    before the next transaction is applied — gives the correct answer and
    matches the FIFO lot logic in current_positions()/compute_pnl().

    oversold_warnings lists every (portfolio, ticker, date, oversold qty) where
    a SELL exceeded the quantity on record, so the data gap can be surfaced to
    the user instead of silently swallowed — it almost always means the sold
    position was opened before the tracked history begins and belongs in
    "Initial Holdings".
    """
    daily_positions: dict[str, pd.DataFrame] = {}
    warnings_rows: list[dict] = []

    for portfolio in sorted(df["portfolio"].unique().tolist()):
        temp = df[df["portfolio"] == portfolio].copy()
        temp = temp.sort_values(["ticker_yf", "date"], kind="mergesort")

        events: list[tuple] = []  # (date, ticker, running_qty_after_this_txn)

        for ticker, ticker_txns in temp.groupby("ticker_yf", sort=False):
            running = 0.0
            for _, row in ticker_txns.iterrows():
                qty = float(row["quantity"])
                action = str(row["action"]).upper()
                if action == "BUY":
                    running += qty
                elif action == "SELL":
                    actual_sell = min(qty, running)
                    oversold = qty - actual_sell
                    running = running - actual_sell
                    if oversold > 1e-9:
                        warnings_rows.append(
                            {
                                "portfolio": portfolio,
                                "ticker_yf": ticker,
                                "date": row["date"],
                                "oversold_qty": round(float(oversold), 4),
                            }
                        )
                events.append((row["date"], ticker, running))

        if events:
            ev_df = pd.DataFrame(events, columns=["date", "ticker_yf", "qty"])
            # If a ticker has multiple transactions on the same date, keep only
            # the running quantity AFTER the last one that day (end-of-day position).
            holdings_sparse = ev_df.groupby(["date", "ticker_yf"])["qty"].last().unstack("ticker_yf")
        else:
            holdings_sparse = pd.DataFrame(index=pd.DatetimeIndex([]))

        holdings = holdings_sparse.reindex(prices_index).ffill().fillna(0)
        daily_positions[str(portfolio)] = holdings

    oversold_warnings = (
        pd.DataFrame(warnings_rows)
        if warnings_rows
        else pd.DataFrame(columns=["portfolio", "ticker_yf", "date", "oversold_qty"])
    )
    return daily_positions, oversold_warnings


def missing_price_tickers(daily_positions: dict[str, pd.DataFrame], prices: pd.DataFrame) -> set[str]:
    """Tickers that appear in someone's holdings but have no downloaded price series."""
    missing: set[str] = set()
    for holdings in daily_positions.values():
        missing.update(set(holdings.columns) - set(prices.columns))
    return missing


def holdings_consistency_check(
    daily_positions: dict[str, pd.DataFrame], current_pos_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Cross-check: the latest quantity in build_daily_positions() should exactly
    match current_positions() for every (portfolio, ticker) — the two are
    computed independently (a vectorized daily walk vs. row-by-row FIFO lots)
    and should never disagree. A mismatch here means one of the two has a bug;
    surfacing it beats silently trusting either one.

    Returns a DataFrame of any mismatches (empty if everything reconciles).
    """
    cur_by_portfolio: dict[str, pd.Series] = {
        str(p): grp.set_index("ticker_yf")["qty"]
        for p, grp in current_pos_df.groupby("portfolio")
    }

    rows = []
    for portfolio, holdings in daily_positions.items():
        if holdings.empty:
            latest = pd.Series(dtype=float)
        else:
            latest = holdings.iloc[-1]
            latest = latest[latest.abs() > 1e-9]
        cp = cur_by_portfolio.get(str(portfolio), pd.Series(dtype=float))
        all_tickers = set(latest.index) | set(cp.index)
        for t in sorted(all_tickers):
            a = float(latest.get(t, 0.0))
            b = float(cp.get(t, 0.0))
            if abs(a - b) > 1e-6:
                rows.append(
                    {
                        "portfolio": portfolio,
                        "ticker_yf": t,
                        "daily_positions_qty": round(a, 4),
                        "current_positions_qty": round(b, 4),
                    }
                )

    return pd.DataFrame(
        rows, columns=["portfolio", "ticker_yf", "daily_positions_qty", "current_positions_qty"]
    )


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


def drawdown_series(value_series: pd.Series) -> pd.Series:
    """
    Generic peak-to-trough drawdown of any monotonic-ish value series.

    Feed this a TWR growth-of-₹1 curve (`cumulative_growth_from_twr`) for a
    performance drawdown — never raw NAV, which would register a client
    withdrawal as a "drawdown" even when nothing was actually lost.
    """
    value_series = value_series.dropna()
    if value_series.empty:
        return pd.Series(dtype=float)
    rolling_max = value_series.cummax()
    return value_series / rolling_max - 1


def performance_attribution(holdings: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    """
    Per-ticker contribution to total portfolio return: for each day, the
    prior day's portfolio weight of a ticker times that ticker's return that
    day, summed over the whole period.

        contribution_i = sum_t( weight_i(t-1) * stock_return_i(t) )

    This is standard daily-rebalanced ("arithmetic") return attribution: the
    sum of all tickers' contributions approximates total portfolio return
    (it won't match exactly — compounding cross-terms are the difference —
    but it should be close; a large gap signals a data problem, e.g. a
    ticker with a bad price series).

    `holdings` must already be non-negative (see build_daily_positions,
    which clips at zero) — a phantom negative holding would flip the sign
    of that ticker's weight and silently corrupt every other ticker's
    contribution too, since weights are normalized by total portfolio value.
    """
    tickers = [t for t in holdings.columns if t in prices.columns]
    if not tickers:
        return pd.Series(dtype=float)

    portfolio_prices = prices[tickers]
    stock_returns = portfolio_prices.pct_change().fillna(0)

    holdings_aligned = holdings[tickers].reindex(stock_returns.index).ffill().fillna(0)
    # Market value of the portfolio's price-covered holdings, day by day —
    # used only to normalize weights below, never displayed as "NAV".
    market_value = (holdings_aligned * portfolio_prices).sum(axis=1)
    weights = holdings_aligned.multiply(portfolio_prices).div(market_value, axis=0).fillna(0)

    contribution = (weights.shift(1).fillna(0) * stock_returns).sum().sort_values(ascending=False)
    return contribution


def attribution_reconciliation(
    holdings: pd.DataFrame, prices: pd.DataFrame, twr_returns: pd.Series
) -> dict[str, float]:
    """
    Sanity check for performance_attribution: compares the sum of all
    per-ticker contributions against the portfolio's actual cumulative TWR
    return over the same window. A small gap is expected (arithmetic vs.
    compounded return); a large one usually means a price or holdings issue.
    """
    attrib = performance_attribution(holdings, prices)
    contribution_total = float(attrib.sum()) if not attrib.empty else np.nan

    r = pd.to_numeric(twr_returns, errors="coerce").dropna()
    actual_total = float((1 + r).prod() - 1) if len(r) >= 1 else np.nan

    gap = (
        float(contribution_total - actual_total)
        if pd.notna(contribution_total) and pd.notna(actual_total)
        else np.nan
    )
    return {
        "contribution_total": contribution_total,
        "actual_cumulative_return": actual_total,
        "gap": gap,
    }


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
