from __future__ import annotations

import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from portfolio_pdf import generate_portfolio_pdf_bytes
from portfolio_pipeline import (
    BENCHMARK_TICKER,
    SECTOR_MAP,
    TransactionsFormatError,
    build_daily_positions,
    build_nav_df,
    current_positions,
    drawdown_series,
    download_prices,
    load_transactions_excel,
    performance_attribution,
    portfolio_returns,
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


@st.cache_data(show_spinner=False, ttl=60 * 60)
def _download_prices_cached(tickers_ns: tuple[str, ...], start_iso: str, include_benchmark: bool):
    import yfinance as yf

    start_dt = pd.to_datetime(start_iso).to_pydatetime()
    return download_prices(yf, tickers_ns=tickers_ns, start=start_dt, include_benchmark=include_benchmark)


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def _fetch_shares_outstanding_cached(tickers_ns: tuple[str, ...]) -> pd.Series:
    """
    Best-effort shares outstanding via yfinance.
    """
    import yfinance as yf

    shares: dict[str, float] = {}
    for t in tickers_ns:
        try:
            tk = yf.Ticker(t)
            so = None
            try:
                fi = tk.fast_info
                if isinstance(fi, dict):
                    so = fi.get("shares") or fi.get("shares_outstanding") or fi.get("sharesOutstanding")
            except Exception:
                so = None
            if so is None:
                try:
                    inf = tk.info
                    if isinstance(inf, dict):
                        so = inf.get("sharesOutstanding")
                except Exception:
                    so = None
            if so is not None:
                shares[t] = float(so)
        except Exception:
            continue
    return pd.Series(shares, dtype="float64")


logo_path = "assets/vika_logo.png"
try:
    st.image(logo_path, width=160)
except Exception:
    pass

st.title("Portfolio Stats")
st.caption("Upload a transactions Excel file → view portfolio analytics → download reports per client.")


def _bar_chart_from_series(series: pd.Series, *, title: str, value_label: str = "Value", top_n: int = 12):
    import altair as alt

    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        st.info("Not enough data to chart.")
        return
    s = s.sort_values(ascending=False).head(top_n)
    dfc = pd.DataFrame({"Label": s.index.astype(str), value_label: s.values})
    chart = (
        alt.Chart(dfc)
        .mark_bar()
        .encode(
            y=alt.Y("Label:N", sort="-x", title=None),
            x=alt.X(f"{value_label}:Q", title=None),
            tooltip=["Label:N", alt.Tooltip(f"{value_label}:Q", format=",.6f")],
        )
        .properties(height=min(28 * len(dfc), 360), title=title)
    )
    st.altair_chart(chart, use_container_width=True)


def _heatmap(df: pd.DataFrame, *, title: str):
    import matplotlib.pyplot as plt

    if df.empty:
        st.info("Not enough data.")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(df.values, aspect="auto")
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    ax.set_yticklabels(df.index)
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            v = df.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

with st.sidebar:
    st.header("Upload")
    uploaded = st.file_uploader("Transactions Excel (.xlsx)", type=["xlsx"])
    password = st.text_input("Password (only if file is encrypted)", type="password")
    include_benchmark = st.checkbox(f"Include benchmark ({BENCHMARK_TICKER})", value=True)
    st.divider()
    st.header("Assumptions")
    rf_annual = st.number_input("Risk-free rate (annual, %)", min_value=0.0, value=0.0, step=0.25) / 100.0

if not uploaded:
    st.info("Upload the transactions Excel file to begin.")
    st.stop()

try:
    excel_bytes = uploaded.getvalue()
    df_raw = load_transactions_excel(excel_bytes, password=password or None)
    df = prepare_transactions(df_raw)
except TransactionsFormatError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Failed to read the uploaded file: {e}")
    st.stop()

st.subheader("Transactions")
st.dataframe(df.head(50), use_container_width=True)

positions = current_positions(df)
tickers = tuple(sorted(positions["ticker_yf"].unique().tolist()))
start_iso = df["date"].min().date().isoformat()

with st.spinner("Downloading market data (yfinance)…"):
    dl = _download_prices_cached(tickers, start_iso, include_benchmark)

prices = dl.prices.sort_index()
if prices.empty:
    st.error("No prices could be downloaded. Check ticker mappings and internet access on the host.")
    st.stop()

daily_positions = build_daily_positions(df, prices.index)
nav_df, missing_tickers = build_nav_df(daily_positions, prices)
rets = portfolio_returns(nav_df)
nav_df.columns = nav_df.columns.astype(str)
rets.columns = rets.columns.astype(str)

benchmark_nav = prices[BENCHMARK_TICKER] if include_benchmark and BENCHMARK_TICKER in prices.columns else None

# Portfolio-level metrics table (annualized + risk metrics)
metrics_rows = []
for p in nav_df.columns:
    m = portfolio_metrics(nav_df[p], benchmark_nav=benchmark_nav, rf_annual=rf_annual)
    m.name = str(p)
    metrics_rows.append(m)
stats = pd.DataFrame(metrics_rows)

overview_tab, client_tab = st.tabs(["Overview", "Client report"])

with overview_tab:
    st.subheader("Key metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clients", f"{len(nav_df.columns)}")
    c2.metric("Tickers", f"{len(tickers)}")
    c3.metric("From", f"{df['date'].min().date().isoformat()}")
    c4.metric("To", f"{df['date'].max().date().isoformat()}")

    st.divider()
    st.subheader("Portfolio statistics")
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
    if "Beta (Nifty 500)" in stats_disp.columns:
        stats_fmt["Beta (Nifty 500)"] = _fmt_col_float(stats_disp["Beta (Nifty 500)"])
    if "Jensen Alpha" in stats_disp.columns:
        stats_fmt["Jensen Alpha"] = _fmt_col_pct(stats_disp["Jensen Alpha"])
    if "Treynor" in stats_disp.columns:
        stats_fmt["Treynor"] = _fmt_col_float(stats_disp["Treynor"])
    if "VaR 99% (Daily)" in stats_disp.columns:
        stats_fmt["VaR 99% (Daily)"] = _fmt_col_pct(stats_disp["VaR 99% (Daily)"])
    if "Max Drawdown" in stats_disp.columns:
        stats_fmt["Max Drawdown"] = _fmt_col_pct(stats_disp["Max Drawdown"])

    st.dataframe(stats_fmt, use_container_width=True)

    st.subheader("Data quality")
    if dl.failed_tickers:
        st.warning(f"Price download failed for {len(dl.failed_tickers)} tickers.")
        st.code("\n".join(dl.failed_tickers))
    if missing_tickers:
        st.warning(f"Holdings contain {len(missing_tickers)} tickers missing from downloaded prices.")
        st.code("\n".join(sorted(missing_tickers)))
    # Market-cap split intentionally omitted (too unreliable across tickers via Yahoo Finance).

with client_tab:
    portfolios = [c for c in nav_df.columns if c and str(c).strip()]
    if not portfolios:
        st.error("No portfolios found in the uploaded data.")
        st.stop()

    portfolio = st.selectbox("Client / portfolio", portfolios)

    nav = nav_df[portfolio].dropna()
    pr = rets[portfolio].dropna() if portfolio in rets.columns else pd.Series(dtype=float)
    corr = None

    client_metrics = portfolio_metrics(nav, benchmark_nav=benchmark_nav, rf_annual=rf_annual)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Annualized return", _fmt_pct(client_metrics.get("Annualized Return")))
    k2.metric("Annualized vol", _fmt_pct(client_metrics.get("Annualized Volatility")))
    k3.metric("Sharpe", _fmt_float(client_metrics.get("Sharpe")))
    k4.metric("Max drawdown", _fmt_pct(client_metrics.get("Max Drawdown")))

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("NAV")
        st.line_chart(nav, height=260)
    with c2:
        st.subheader("Drawdown")
        dd = drawdown_series(nav)
        st.area_chart(dd, height=260)

    st.subheader("Returns (daily)")
    st.line_chart(pr, height=220)

    st.subheader("Holdings (latest)")
    holdings = daily_positions.get(str(portfolio), pd.DataFrame())
    if holdings.empty:
        st.info("No holdings available for this portfolio.")
    else:
        latest = holdings.iloc[-1]
        latest = latest[latest > 0].sort_values(ascending=False).to_frame(name="qty")
        st.dataframe(latest, use_container_width=True)

    st.subheader("Download")
    report_choice = st.selectbox(
        "Choose report",
        [
            "PDF report",
            "NAV history (CSV)",
            "Daily returns (CSV)",
            "Holdings (latest, CSV)",
            "Transactions (this client, CSV)",
        ],
    )

    report_date = datetime.now().date()
    stats_row = client_metrics
    benchmark_returns = prices[BENCHMARK_TICKER].pct_change() if include_benchmark and BENCHMARK_TICKER in prices.columns else None

    st.subheader("Attribution & risk")
    attrib = None
    rc = None
    winners = None
    laggards = None
    if not holdings.empty:
        attrib = performance_attribution(holdings, prices)
        rc = risk_contribution(holdings, prices)

        a1, a2 = st.columns(2)
        with a1:
            st.caption("Performance attribution (top)")
            if attrib is not None and not attrib.empty:
                _bar_chart_from_series(attrib, title="Attribution (top)", value_label="Contribution", top_n=12)
            else:
                st.info("Not enough data for attribution.")
        with a2:
            st.caption("Risk contribution")
            if rc is not None and not rc.empty:
                _bar_chart_from_series(rc, title="Risk contribution (top)", value_label="Risk", top_n=12)
            else:
                st.info("Not enough data for risk contribution.")

    st.subheader("Top movers (stocks)")
    tickers_in_portfolio = []
    if not holdings.empty:
        tickers_in_portfolio = [t for t in holdings.columns if t in prices.columns]
    if tickers_in_portfolio:
        # Period returns per stock (first/last valid)
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
        st.info("No tickers available for this client.")

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
        if holdings.empty or nav.empty or pr.empty:
            st.warning("Not enough data to generate a PDF report for this client.")
        else:
            try:
                pdf_bytes = generate_portfolio_pdf_bytes(
                    portfolio_name=str(portfolio),
                    report_date=report_date,
                    nav=nav,
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
                )
                st.download_button(
                    "Download PDF",
                    data=pdf_bytes,
                    file_name=f"Portfolio_{portfolio}.pdf",
                    mime="application/pdf",
                    key=f"dl_pdf_{portfolio}",
                )
            except Exception as e:
                st.error(f"Failed to build PDF: {e}")

    elif report_choice == "NAV history (CSV)":
        out = nav.to_frame(name="nav")
        st.download_button(
            "Download NAV CSV",
            data=_df_to_csv_bytes(out),
            file_name=f"NAV_{portfolio}.csv",
            mime="text/csv",
            key=f"dl_nav_{portfolio}",
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
