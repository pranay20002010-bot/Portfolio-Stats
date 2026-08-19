from __future__ import annotations

import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from portfolio_pdf import generate_portfolio_pdf_bytes
from portfolio_pipeline import (
    BENCHMARKS,
    SECTOR_MAP,
    TransactionsFormatError,
    attribution_reconciliation,
    build_daily_ledger,
    build_daily_positions,
    cap_split_weights,
    categorize_market_caps_inr,
    cashflow_summary,
    compute_ledger_twr,
    compute_pnl,
    cumulative_growth_from_twr,
    current_positions,
    download_prices,
    drawdown_series,
    fetch_market_caps,
    format_inr,
    get_corr_cmap,
    load_transactions_excel,
    holdings_consistency_check,
    missing_price_tickers,
    performance_attribution,
    portfolio_metrics,
    prepare_transactions,
    risk_contribution,
)


st.set_page_config(page_title="Portfolio Stats", layout="wide")


def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8")


def _fmt_pct(x) -> str:
    try:
        v = float(x)
    except Exception:
        return "N/A"
    if pd.isna(v) or v in (float("inf"), float("-inf")):
        return "N/A"
    return f"{v*100:,.2f}%"


def _fmt_float(x) -> str:
    try:
        v = float(x)
    except Exception:
        return "N/A"
    if pd.isna(v) or v in (float("inf"), float("-inf")):
        return "N/A"
    return f"{v:,.2f}"


def _fmt_inr(x) -> str:
    return format_inr(x) or "N/A"


def _fmt_pct_signed(x) -> str:
    try:
        v = float(x)
    except Exception:
        return "N/A"
    if pd.isna(v) or v in (float("inf"), float("-inf")):
        return "N/A"
    return f"{v*100:+,.2f}%"


@st.cache_data(show_spinner=False, ttl=60 * 60)
def _download_prices_cached(
    tickers_ns: tuple[str, ...], start_iso: str, include_benchmark: bool, benchmark_tickers: tuple[str, ...]
):
    import yfinance as yf

    start_dt = pd.to_datetime(start_iso).to_pydatetime()
    return download_prices(
        yf,
        tickers_ns=tickers_ns,
        start=start_dt,
        include_benchmark=include_benchmark,
        benchmark_tickers=benchmark_tickers,
    )


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def _fetch_market_caps_cached(tickers_ns: tuple[str, ...]) -> pd.Series:
    """Best-effort market cap (INR) per ticker via yfinance, for the market-cap
    allocation pie chart. Cached for a day since market cap barely moves
    intraday and this is one Ticker() call per holding."""
    import yfinance as yf

    return fetch_market_caps(yf, tickers_ns)


# Show logo — support both png and jpg
import os as _os
_logo_path = next(
    (p for p in ["assets/vika_logo.png", "assets/vika_logo.jpg", "assets/Vika_Logo.jpg", "assets/Vika_Logo.png"]
     if _os.path.exists(p)),
    None,
)
if _logo_path:
    try:
        st.image(_logo_path, width=200)
    except Exception:
        pass

st.title("Portfolio Stats")
st.caption("Upload a transactions Excel file → view portfolio analytics → download reports per client.")


def _bar_chart_from_series(
    series: pd.Series, *, title: str, value_label: str = "Value", top_n: int = 12, as_percent: bool = True
):
    import altair as alt

    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        st.info("Not enough data to chart.")
        return
    s = s.sort_values(ascending=False).head(top_n)
    display_vals = s.values * 100 if as_percent else s.values
    dfc = pd.DataFrame({"Label": s.index.astype(str), value_label: display_vals})
    fmt = ",.2f" if as_percent else ",.6f"
    axis_title = f"{value_label} (%)" if as_percent else value_label
    chart = (
        alt.Chart(dfc)
        .mark_bar()
        .encode(
            y=alt.Y("Label:N", sort="-x", title=None),
            x=alt.X(f"{value_label}:Q", title=axis_title),
            tooltip=["Label:N", alt.Tooltip(f"{value_label}:Q", format=fmt, title=axis_title)],
        )
        .properties(height=min(28 * len(dfc), 360), title=title)
    )
    st.altair_chart(chart, use_container_width=True)


def _heatmap(df: pd.DataFrame, *, title: str):
    import matplotlib.pyplot as plt

    if df.empty:
        st.info("Not enough data.")
        return

    cmap = get_corr_cmap()

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(df.values.astype(float), aspect="auto", cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    ax.set_yticklabels(df.index)
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            v = df.iat[i, j]
            # Luminance-based contrast: the navy/gold colormap isn't symmetric
            # in brightness (navy darkens faster than gold), so a fixed |v|
            # cutoff picks the wrong text color near +1; check actual cell
            # luminance instead.
            r, g, b, _ = cmap((float(v) + 1) / 2)
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "white" if luminance < 0.55 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=text_color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with st.sidebar:
    st.header("Upload")
    uploaded = st.file_uploader("Transactions Excel (.xlsx)", type=["xlsx"])
    password = st.text_input("Password (only if file is encrypted)", type="password")
    include_benchmark = st.checkbox("Include benchmarks (Nifty 50 & Nifty 500)", value=True)
    st.divider()
    st.header("Assumptions")
    rf_annual = st.number_input("Risk-free rate (annual, %)", min_value=0.0, value=0.0, step=0.25) / 100.0
    st.caption(
        "Defaults to 0% — Sharpe, Sortino, and Treynor all subtract this from the return, so at 0% those "
        "ratios show raw return/risk with no risk-free adjustment, which reads high/unusual next to "
        "reports that use a real rate. Set this to your actual reference rate (e.g. a 91-day T-bill or "
        "repo-linked yield) to get comparable ratios."
    )

if not uploaded:
    st.info("Upload the transactions Excel file to begin.")
    st.stop()

try:
    excel_bytes = uploaded.getvalue()
    sheets = load_transactions_excel(excel_bytes, password=password or None)
    # prepare_transactions reads all three sheets — "Initial Holdings" (seeded as
    # opening BUY transactions at each portfolio's StartDate), "Transactions"
    # (the trade log), and "Cashflows" (deposits/withdrawals, kept separate).
    df, cashflows_df = prepare_transactions(sheets)
except TransactionsFormatError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Failed to read the uploaded file: {e}")
    st.stop()

positions = current_positions(df)
tickers = tuple(sorted(positions["ticker_yf"].unique().tolist()))
start_iso = df["date"].min().date().isoformat()

with st.spinner("Downloading market data (yfinance)…"):
    dl = _download_prices_cached(tickers, start_iso, include_benchmark, tuple(BENCHMARKS.values()))

prices = dl.prices.sort_index()
if prices.empty:
    st.error("No prices could be downloaded. Check ticker mappings and internet access on the host.")
    st.stop()

daily_positions, oversold_warnings = build_daily_positions(df, prices.index)
missing_tickers = missing_price_tickers(daily_positions, prices)
# TWR returns come from a full daily ledger (equity + cash combined), not
# stock prices alone — deposits/withdrawals are excluded from the return via
# the External Flow adjustment in compute_ledger_twr, so they still never
# show up as investment gain/loss, but idle cash and cash timing are now
# properly reflected in the portfolio's value and TWR.
ledgers = build_daily_ledger(df, cashflows_df, prices)
twr_series = {}
for _portfolio, _ledger in ledgers.items():
    twr_series[_portfolio] = compute_ledger_twr(_ledger)
twr_df = pd.DataFrame(twr_series)
twr_df.columns = twr_df.columns.astype(str)
portfolios_all = sorted(str(p) for p in daily_positions.keys())

benchmarks = (
    {name: prices[ticker] for name, ticker in BENCHMARKS.items() if ticker in prices.columns}
    if include_benchmark
    else {}
)

overview_tab, client_tab = st.tabs(["Overview", "Client report"])

with overview_tab:
    st.subheader("Key metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clients", f"{len(portfolios_all)}")
    c2.metric("Tickers", f"{len(tickers)}")
    c3.metric("From", f"{df['date'].min().date().isoformat()}")
    c4.metric("To", f"{df['date'].max().date().isoformat()}")

    st.divider()
    st.subheader("Portfolio statistics")

    # Compute metrics here so rf_annual slider changes are reflected immediately
    metrics_rows = []
    for p in portfolios_all:
        twr_col = twr_df[p].dropna() if p in twr_df.columns else pd.Series(dtype=float)
        m = portfolio_metrics(twr_col, benchmarks=benchmarks, rf_annual=rf_annual)
        m.name = str(p)
        metrics_rows.append(m)
    stats = pd.DataFrame(metrics_rows)

    stats_disp = stats.copy()
    for col in stats_disp.columns:
        s = pd.to_numeric(stats_disp[col], errors="coerce")
        stats_disp[col] = s.mask(~pd.Series(np.isfinite(s), index=s.index), np.nan)

    # Show a clean, formatted table instead of lots of "None".
    def _fmt_col_pct(c: pd.Series) -> pd.Series:
        return c.apply(lambda v: "N/A" if pd.isna(v) else f"{v*100:,.2f}%")

    def _fmt_col_float(c: pd.Series) -> pd.Series:
        return c.apply(lambda v: "N/A" if pd.isna(v) else f"{v:,.2f}")

    stats_fmt = pd.DataFrame(index=stats_disp.index.astype(str))
    if "Annualized Return" in stats_disp.columns:
        stats_fmt["Annualized Return"] = _fmt_col_pct(stats_disp["Annualized Return"])
    if "Annualized Volatility" in stats_disp.columns:
        stats_fmt["Annualized Volatility"] = _fmt_col_pct(stats_disp["Annualized Volatility"])
    if "Sharpe" in stats_disp.columns:
        stats_fmt["Sharpe"] = _fmt_col_float(stats_disp["Sharpe"])
    if "Sortino" in stats_disp.columns:
        stats_fmt["Sortino"] = _fmt_col_float(stats_disp["Sortino"])
    # Beta/Jensen Alpha/Treynor exist once per benchmark (e.g. "Beta (Nifty 50)",
    # "Beta (Nifty 500)") — pick them up dynamically rather than hardcoding one.
    for col in stats_disp.columns:
        if col.startswith("Beta (") or col.startswith("Treynor ("):
            stats_fmt[col] = _fmt_col_float(stats_disp[col])
        elif col.startswith("Jensen Alpha ("):
            stats_fmt[col] = _fmt_col_pct(stats_disp[col])
    if "CVaR 99% (Daily)" in stats_disp.columns:
        stats_fmt["CVaR 99% (Daily)"] = _fmt_col_pct(stats_disp["CVaR 99% (Daily)"])
    if "Max Drawdown" in stats_disp.columns:
        stats_fmt["Max Drawdown"] = _fmt_col_pct(stats_disp["Max Drawdown"])

    st.dataframe(stats_fmt, use_container_width=True)

    # ── Realized vs Unrealized P&L ──────────────────────────────────────────
    st.subheader("Realized vs Unrealized P&L (₹)")
    with st.spinner("Computing P&L…"):
        pnl_df = compute_pnl(df, prices)

    if pnl_df.empty:
        st.info("No P&L data available.")
    else:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker

        portfolios_pnl = pnl_df.index.tolist()
        x = np.arange(len(portfolios_pnl))
        width = 0.32

        fig_pnl, ax_pnl = plt.subplots(figsize=(max(7, len(portfolios_pnl) * 1.6), 4.5))
        bars_r = ax_pnl.bar(x - width / 2, pnl_df["realized_pnl"], width,
                            label="Realized P&L",
                            color=["#27ae60" if v >= 0 else "#e74c3c" for v in pnl_df["realized_pnl"]])
        bars_u = ax_pnl.bar(x + width / 2, pnl_df["unrealized_pnl"], width,
                            label="Unrealized P&L",
                            color=["#2980b9" if v >= 0 else "#e67e22" for v in pnl_df["unrealized_pnl"]])

        ax_pnl.set_xticks(x)
        ax_pnl.set_xticklabels(portfolios_pnl, rotation=30, ha="right", fontsize=9)
        ax_pnl.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"₹{v:,.0f}"))
        ax_pnl.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax_pnl.set_title("Realized vs Unrealized P&L per Portfolio", fontsize=11)
        ax_pnl.legend(fontsize=9)
        ax_pnl.grid(True, axis="y", alpha=0.25)

        # Label bars with values
        for bar in list(bars_r) + list(bars_u):
            h = bar.get_height()
            if abs(h) > 0:
                ax_pnl.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + (max(pnl_df[["realized_pnl", "unrealized_pnl"]].abs().max()) * 0.01),
                    f"₹{h:,.0f}",
                    ha="center", va="bottom", fontsize=7, rotation=45,
                )

        fig_pnl.tight_layout()
        st.pyplot(fig_pnl, use_container_width=True)
        plt.close(fig_pnl)

        # Also show as table
        pnl_display = pnl_df.copy()
        for col in pnl_display.columns:
            pnl_display[col] = pnl_display[col].apply(lambda v: f"₹{v:,.2f}")
        pnl_display.columns = ["Realized P&L", "Unrealized P&L", "Total P&L"]
        st.dataframe(pnl_display, use_container_width=True)

    # ── Cash flows ───────────────────────────────────────────────────────────
    st.subheader("Cash flows (capital contributed)")
    if cashflows_df.empty:
        st.info("No cashflow data found.")
    else:
        summary = cashflow_summary(cashflows_df)
        summary_disp = summary.to_frame("Net cash contributed (₹)")
        summary_disp["Net cash contributed (₹)"] = summary_disp["Net cash contributed (₹)"].apply(_fmt_inr)
        st.dataframe(summary_disp, use_container_width=True)
        st.caption(
            "Net cash in/out per portfolio from the Cashflows sheet. TWR below is computed from the full "
            "daily ledger (equity + cash combined) — deposits/withdrawals are still excluded from the return "
            "itself, but idle cash and its timing are reflected in the portfolio's value."
        )

    st.subheader("Cash balance (latest)")
    if ledgers:
        cash_rows = {p: led["cash"].iloc[-1] for p, led in ledgers.items() if not led.empty}
        cash_disp = pd.Series(cash_rows, dtype="float64").sort_values(ascending=False).to_frame("Cash (₹)")
        cash_disp["Cash (₹)"] = cash_disp["Cash (₹)"].apply(_fmt_inr)
        st.dataframe(cash_disp, use_container_width=True)
    else:
        st.info("No ledger data available.")

    st.subheader("Data quality")
    if dl.failed_tickers:
        st.warning(f"Price download failed for {len(dl.failed_tickers)} tickers.")
        st.code("\n".join(dl.failed_tickers))
    if missing_tickers:
        st.warning(f"Holdings contain {len(missing_tickers)} tickers missing from downloaded prices.")
        st.code("\n".join(sorted(missing_tickers)))
    if not oversold_warnings.empty:
        st.warning(
            f"{len(oversold_warnings)} SELL transaction(s) exceed the quantity on record for that ticker "
            "(clamped to zero rather than allowed to go negative). This almost always means the position "
            "was opened before the tracked history starts and is missing from 'Initial Holdings'."
        )
        warn_disp = oversold_warnings.copy()
        warn_disp["date"] = pd.to_datetime(warn_disp["date"]).dt.date.astype(str)
        st.dataframe(warn_disp, use_container_width=True)
    consistency_issues = holdings_consistency_check(daily_positions, positions)
    if not consistency_issues.empty:
        st.error(
            f"{len(consistency_issues)} holding(s) disagree between the two independent position-tracking "
            "methods used in this app. This should never happen — treat any numbers for these tickers as "
            "unreliable until this is investigated."
        )
        st.dataframe(consistency_issues, use_container_width=True)

    negative_cash_rows = []
    for _p, _led in ledgers.items():
        if _led.empty:
            continue
        _min_cash = float(_led["cash"].min())
        if _min_cash < -1.0:
            negative_cash_rows.append(
                {"portfolio": _p, "min_cash": round(_min_cash, 2), "date": _led["cash"].idxmin().date().isoformat()}
            )
    if negative_cash_rows:
        st.error(
            f"{len(negative_cash_rows)} portfolio(s) show negative cash at some point in the daily ledger. "
            "This happens when a BUY isn't matched by a recorded Cashflows deposit — for a portfolio with "
            "'Initial Holdings', that position is treated as already-funded (no cash debit at StartDate), so "
            "any *subsequent* real purchase needs its own deposit on record. A negative cash balance "
            "understates Portfolio Value and will distort TWR, Sharpe, Sortino, and every other ratio "
            "downstream for that portfolio — this is very likely why the ratios look off."
        )
        st.dataframe(pd.DataFrame(negative_cash_rows), use_container_width=True)


with client_tab:
    portfolios = portfolios_all
    if not portfolios:
        st.error("No portfolios found in the uploaded data.")
        st.stop()

    portfolio = st.selectbox("Client / portfolio", portfolios)

    # Transactions filtered to selected client
    st.subheader("Transactions")
    client_txns = df[df["portfolio"].astype(str).str.strip() == str(portfolio).strip()]
    st.dataframe(client_txns.sort_values("date", ascending=False).reset_index(drop=True), use_container_width=True)

    st.subheader("Cash flows")
    client_cash = cashflows_df[cashflows_df["portfolio"].astype(str).str.strip() == str(portfolio).strip()]
    if client_cash.empty:
        st.info("No cashflow records for this client.")
    else:
        cash_disp = client_cash.sort_values("date", ascending=False).reset_index(drop=True).copy()
        cash_disp["amount"] = cash_disp["amount"].apply(_fmt_inr)
        st.dataframe(cash_disp, use_container_width=True)

    pr = twr_df[portfolio].dropna() if portfolio in twr_df.columns else pd.Series(dtype=float)
    corr = None
    client_ledger = ledgers.get(str(portfolio), pd.DataFrame())

    client_metrics = portfolio_metrics(pr, benchmarks=benchmarks, rf_annual=rf_annual)

    v1, v2 = st.columns(2)
    if not client_ledger.empty:
        v1.metric("Current portfolio value", _fmt_inr(client_ledger["portfolio_value"].iloc[-1]))
        v2.metric("Cash balance", _fmt_inr(client_ledger["cash"].iloc[-1]))
    else:
        v1.metric("Current portfolio value", "N/A")
        v2.metric("Cash balance", "N/A")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Annualized return", _fmt_pct(client_metrics.get("Annualized Return")))
    k2.metric("Annualized vol", _fmt_pct(client_metrics.get("Annualized Volatility")))
    k3.metric("Sharpe", _fmt_float(client_metrics.get("Sharpe")))
    k4.metric("Max drawdown", _fmt_pct(client_metrics.get("Max Drawdown")))
    m1, m2 = st.columns(2)
    m1.metric("Sortino", _fmt_float(client_metrics.get("Sortino")))
    m2.metric("99% CVaR (Daily)", _fmt_pct(client_metrics.get("CVaR 99% (Daily)")))

    if benchmarks:
        st.caption("vs. benchmark")
        bench_rows = []
        for name in benchmarks.keys():
            bench_rows.append(
                {
                    "Benchmark": name,
                    "Beta": _fmt_float(client_metrics.get(f"Beta ({name})")),
                    "Jensen Alpha": _fmt_pct(client_metrics.get(f"Jensen Alpha ({name})")),
                    "Treynor": _fmt_float(client_metrics.get(f"Treynor ({name})")),
                }
            )
        st.dataframe(pd.DataFrame(bench_rows).set_index("Benchmark"), use_container_width=True)

    growth = cumulative_growth_from_twr(pr)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Performance vs. Nifty 50 & Nifty 500")
        if growth.empty:
            st.info("Not enough return history for a performance chart.")
        else:
            # Rebase each benchmark to 1.0 on the portfolio's own start date so
            # all lines are directly comparable growth-of-₹1 curves over the
            # same window, regardless of the benchmark's own longer history.
            comparison = pd.DataFrame({str(portfolio): growth})
            for name, bench_prices in benchmarks.items():
                bench_aligned = bench_prices.reindex(growth.index).ffill()
                if bench_aligned.dropna().empty:
                    continue
                first_valid = bench_aligned.dropna().iloc[0]
                if first_valid and first_valid > 0:
                    comparison[name] = bench_aligned / first_valid
            st.line_chart(comparison, height=260)
    with c2:
        st.subheader("Drawdown")
        dd = drawdown_series(growth)
        if dd.empty:
            st.info("Not enough return history for drawdown.")
        else:
            import matplotlib.pyplot as plt
            import matplotlib.ticker as mticker
            fig_dd, ax_dd = plt.subplots(figsize=(6, 2.8))
            ax_dd.fill_between(dd.index, dd.values * 100, 0, color="#C0392B", alpha=0.6)
            ax_dd.plot(dd.index, dd.values * 100, color="#922B21", linewidth=0.8)
            ax_dd.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
            ax_dd.set_ylabel("Drawdown (%)")
            ax_dd.set_title(f"Drawdown — {portfolio}", fontsize=9)
            ax_dd.grid(True, alpha=0.2)
            # Ensure y-axis shows the actual range (negative values)
            dd_min = float(dd.min()) * 100
            ax_dd.set_ylim(min(dd_min * 1.15, dd_min - 1), 1)
            fig_dd.tight_layout()
            st.pyplot(fig_dd, use_container_width=True)
            plt.close(fig_dd)

    st.subheader("Returns (daily)")
    st.line_chart(pr, height=220)

    # Normalised positions for this portfolio — used in both Holdings and Top Movers
    pos_copy = positions.copy()
    pos_copy["portfolio"] = pos_copy["portfolio"].astype(str).str.strip()
    port_pos = pos_copy[pos_copy["portfolio"] == str(portfolio).strip()]

    st.subheader("Holdings (latest)")
    holdings = daily_positions.get(str(portfolio), pd.DataFrame())
    if holdings.empty:
        st.info("No holdings available for this portfolio.")
    else:
        latest = holdings.iloc[-1]
        latest = latest[latest > 0].sort_values(ascending=False).to_frame(name="Qty")
        if "avg_buy_price" in port_pos.columns and not port_pos.empty:
            cur_pos = port_pos[["ticker_yf", "avg_buy_price"]].set_index("ticker_yf")
            cur_pos = cur_pos.rename(columns={"avg_buy_price": "Avg Buy Price (₹)"})
            latest = latest.join(cur_pos, how="left")
        st.dataframe(latest, use_container_width=True)

    st.subheader("Market cap allocation")
    cap_split = pd.Series(dtype=float)
    if not holdings.empty:
        held_tickers = tuple(sorted(latest.index.tolist())) if not latest.empty else ()
        if held_tickers:
            with st.spinner("Fetching market caps…"):
                mkt_caps = _fetch_market_caps_cached(held_tickers)
            cap_split = cap_split_weights(holdings, prices, mkt_caps)
        if cap_split.empty:
            st.info("Market cap data unavailable for this portfolio's holdings.")
        else:
            # matplotlib, not plotly — plotly isn't a guaranteed dependency on
            # every deployment and every other chart in this app already uses
            # matplotlib, so this stays consistent and doesn't add a new
            # import that can break the app if it's missing at deploy time.
            import matplotlib.pyplot as plt

            cap_color_map = {"Large Cap": "#1f77b4", "Mid Cap": "#ff7f0e", "Small Cap": "#2ca02c", "Unknown": "#B0B0B0"}
            colors_ordered = [cap_color_map.get(k, "#B0B0B0") for k in cap_split.index]
            fig_cap, ax_cap = plt.subplots(figsize=(5, 5))
            cap_split.plot.pie(
                ax=ax_cap,
                autopct="%1.1f%%",
                startangle=90,
                colors=colors_ordered,
                textprops={"fontsize": 9},
            )
            ax_cap.set_ylabel("")
            fig_cap.tight_layout()
            left_col, center_col, right_col = st.columns([1, 2, 1])
            with center_col:
                st.pyplot(fig_cap, use_container_width=True)
            plt.close(fig_cap)
    else:
        st.info("No holdings available for this portfolio.")

    st.subheader("Download")
    report_choice = st.selectbox(
        "Choose report",
        [
            "PDF report",
            "Cumulative growth (CSV)",
            "Daily returns (CSV)",
            "Holdings (latest, CSV)",
            "Transactions (this client, CSV)",
        ],
    )

    report_date = datetime.now().date()
    stats_row = client_metrics
    benchmark_returns = {name: s.pct_change() for name, s in benchmarks.items()} if benchmarks else None

    st.subheader("Attribution & risk")
    attrib = None
    rc = None
    winners = None
    laggards = None
    if not holdings.empty:
        attrib = performance_attribution(holdings, prices)
        rc = risk_contribution(holdings, prices)

        st.caption("Contribution to return, since inception — biggest positive to biggest negative contributor")
        if attrib is not None and not attrib.empty:
            contrib_df = attrib.sort_values(ascending=False).to_frame("contribution")
            contrib_df.index = contrib_df.index.str.replace(".NS", "", regex=False).str.replace(".BO", "", regex=False)
            contrib_df.index.name = "Ticker"
            contrib_df["Contribution to Return"] = contrib_df["contribution"].apply(_fmt_pct_signed)
            contrib_df = contrib_df[["Contribution to Return"]]
            st.dataframe(
                contrib_df,
                use_container_width=True,
                height=min(35 * (len(contrib_df) + 1), 480),
            )
        else:
            st.info("Not enough data for attribution.")

        st.caption("Risk contribution — % of total portfolio risk")
        if rc is not None and not rc.empty:
            _bar_chart_from_series(rc, title="Risk contribution (top)", value_label="Risk share", top_n=12)
        else:
            currently_held_ct = int((holdings.iloc[-1] > 1e-9).sum()) if not holdings.empty else 0
            if currently_held_ct < 2:
                st.info(
                    "Only one current holding in this portfolio — risk contribution needs at least two "
                    "to show how risk is split between them."
                )
            else:
                st.info(
                    "Not enough overlapping price history between the current holdings to compute a "
                    "reliable covariance (e.g. a recently-listed stock with a short price history)."
                )

        if not pr.empty:
            recon = attribution_reconciliation(holdings, prices, pr)
            gap = recon.get("gap")
            if pd.notna(gap):
                st.caption(
                    f"Sum of ticker contributions: {_fmt_pct(recon['contribution_total'])} vs. actual cumulative "
                    f"return: {_fmt_pct(recon['actual_cumulative_return'])} (gap: {_fmt_pct(gap)}, "
                    "expected from daily-rebalanced attribution vs. compounded return)."
                )

    st.subheader("Top movers (current holdings only)")
    # Only include tickers that are CURRENTLY held (qty > 0) — not historic positions
    currently_held = set(
        pos_copy.loc[pos_copy["portfolio"] == str(portfolio).strip(), "ticker_yf"].tolist()
    )
    tickers_in_portfolio = [t for t in currently_held if t in prices.columns]
    if tickers_in_portfolio:
        # Return since each stock's first available price in the prices table
        # but only for currently held names — realized stocks excluded
        prc = prices[tickers_in_portfolio].copy()
        stock_rets = {}
        for t in tickers_in_portfolio:
            s = prc[t].dropna()
            if len(s) < 2:
                continue
            stock_rets[t] = float(s.iloc[-1] / s.iloc[0] - 1)
        if stock_rets:
            sr = pd.Series(stock_rets).sort_values(ascending=False)
            winners = sr.head(5)
            laggards = sr.tail(5).sort_values(ascending=True)
            w1, w2 = st.columns(2)
            with w1:
                st.caption("Top 5 performers")
                winners_df = winners.to_frame("Return").copy()
                winners_df["Return"] = winners_df["Return"].apply(_fmt_pct)
                st.dataframe(winners_df, use_container_width=True)
            with w2:
                st.caption("Top 5 laggards")
                laggards_df = laggards.to_frame("Return").copy()
                laggards_df["Return"] = laggards_df["Return"].apply(_fmt_pct)
                st.dataframe(laggards_df, use_container_width=True)
        else:
            st.info("Not enough price history to compute top movers.")
    else:
        st.info("No currently held tickers available for this client.")

    st.subheader("Return distribution (daily)")
    if not pr.empty and len(pr) >= 10:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        cutoff_99 = float(pr.quantile(0.01))
        tail_99   = pr[pr <= cutoff_99]
        cvar99_daily = float(-tail_99.mean()) if not tail_99.empty else float(-cutoff_99)
        fig_hist, ax_hist = plt.subplots(figsize=(9, 3.5))
        ax_hist.hist(pr.dropna().values, bins=40, color="#1F618D", alpha=0.85, edgecolor="white", linewidth=0.3)
        ax_hist.axvline(cutoff_99, color="#C0392B", linewidth=1.5, linestyle="--",
                        label=f"99% CVaR: {cvar99_daily*100:.2f}%  (tail avg beyond {cutoff_99*100:.2f}%)")
        ax_hist.fill_betweenx([0, ax_hist.get_ylim()[1] or 1],
                              pr.min(), cutoff_99,
                              alpha=0.12, color="#C0392B", label="_nolegend_")
        ax_hist.set_xlabel("Daily return")
        ax_hist.set_ylabel("Frequency")
        ax_hist.set_title(f"Distribution of daily returns — {portfolio}", fontsize=10)
        ax_hist.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
        ax_hist.legend(fontsize=8)
        ax_hist.grid(True, alpha=0.2)
        fig_hist.tight_layout()
        st.pyplot(fig_hist, use_container_width=True)
        plt.close(fig_hist)
    else:
        st.info("Not enough return history to display distribution.")

    st.subheader("Stock correlation (this client)")
    if len(tickers_in_portfolio) >= 2:
        stock_rets = prices[tickers_in_portfolio].pct_change().dropna(how="all")
        stock_rets = stock_rets.dropna(axis=1, how="all")
        if stock_rets.shape[1] >= 2:
            corr = stock_rets.corr().round(2)
            st.dataframe(corr, use_container_width=True)
            _heatmap(corr, title=f"Stock correlation heatmap — {portfolio}")
        else:
            st.info("Not enough price history for a stock correlation matrix.")
    else:
        st.info("Need at least 2 stocks in this portfolio to show correlation.")

    if report_choice == "PDF report":
        if holdings.empty or pr.empty:
            st.warning("Not enough data to generate a PDF report for this client.")
        else:
            try:
                pdf_bytes = generate_portfolio_pdf_bytes(
                    portfolio_name=str(portfolio),
                    report_date=report_date,
                    returns=pr,
                    holdings=holdings,
                    prices=prices,
                    stats_row=stats_row,
                    benchmark_returns=benchmark_returns,
                    sector_map=SECTOR_MAP,
                    risk_contrib=rc if isinstance(rc, pd.Series) and not rc.empty else None,
                    attribution=attrib if isinstance(attrib, pd.Series) and not attrib.empty else None,
                    winners=winners if isinstance(winners, pd.Series) and not winners.empty else None,
                    laggards=laggards if isinstance(laggards, pd.Series) and not laggards.empty else None,
                    stock_corr=corr if "corr" in locals() else None,
                    returns_hist=pr if not pr.empty else None,
                    cash_balance=float(client_ledger["cash"].iloc[-1]) if not client_ledger.empty else None,
                    portfolio_value=float(client_ledger["portfolio_value"].iloc[-1]) if not client_ledger.empty else None,
                    cap_split=cap_split if isinstance(cap_split, pd.Series) and not cap_split.empty else None,
                )
                st.download_button(
                    "Download PDF",
                    data=pdf_bytes,
                    file_name=f"Portfolio_{portfolio}.pdf",
                    mime="application/pdf",
                    key=f"dl_pdf_{portfolio}",
                )
            except Exception as e:
                import traceback
                st.error(f"Failed to build PDF: {e}")
                st.code(traceback.format_exc())

    elif report_choice == "Cumulative growth (CSV)":
        out = growth.to_frame(name="growth_of_1_rupee")
        st.download_button(
            "Download growth CSV",
            data=_df_to_csv_bytes(out),
            file_name=f"Growth_{portfolio}.csv",
            mime="text/csv",
            key=f"dl_growth_{portfolio}",
        )

    elif report_choice == "Daily returns (CSV)":
        out = pr.to_frame(name="returns")
        st.download_button(
            "Download returns CSV",
            data=_df_to_csv_bytes(out),
            file_name=f"Returns_{portfolio}.csv",
            mime="text/csv",
            key=f"dl_rets_{portfolio}",
        )

    elif report_choice == "Holdings (latest, CSV)":
        if holdings.empty:
            st.warning("No holdings for this portfolio.")
        else:
            out = holdings.iloc[-1].to_frame(name="qty")
            out = out[out["qty"] > 0].sort_values("qty", ascending=False)
            st.download_button(
                "Download holdings CSV",
                data=_df_to_csv_bytes(out),
                file_name=f"Holdings_{portfolio}.csv",
                mime="text/csv",
                key=f"dl_hold_{portfolio}",
            )

    else:  # Transactions
        out = df[df["portfolio"] == portfolio].copy()
        st.download_button(
            "Download transactions CSV",
            data=_df_to_csv_bytes(out),
            file_name=f"Transactions_{portfolio}.csv",
            mime="text/csv",
            key=f"dl_txn_{portfolio}",
        )
