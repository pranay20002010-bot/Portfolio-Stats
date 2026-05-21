from __future__ import annotations

import io
from datetime import datetime

import pandas as pd
import streamlit as st

from portfolio_pdf import generate_portfolio_pdf_bytes
from portfolio_pipeline import (
    BENCHMARK_TICKER,
    SECTOR_MAP,
    TransactionsFormatError,
    build_daily_positions,
    build_nav_df,
    cap_split_weights,
    current_positions,
    drawdown_series,
    download_prices,
    load_transactions_excel,
    overlap_matrix,
    performance_attribution,
    portfolio_returns,
    prepare_transactions,
    risk_contribution,
    stats_table,
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
def _fetch_market_caps_cached(tickers_ns: tuple[str, ...]) -> pd.Series:
    """
    Best-effort market cap fetch via yfinance. Returns INR market caps when available.
    """
    import yfinance as yf

    caps: dict[str, float] = {}
    for t in tickers_ns:
        try:
            info = yf.Ticker(t).fast_info
            mc = None
            if info and isinstance(info, dict):
                mc = info.get("market_cap")
            if mc is None:
                # fallback (slower)
                mc = yf.Ticker(t).info.get("marketCap")
            if mc is not None:
                caps[t] = float(mc)
        except Exception:
            continue
    return pd.Series(caps, dtype="float64")


st.title("Portfolio Stats")
st.caption("Upload a transactions Excel file → view portfolio analytics → download reports per client.")

with st.sidebar:
    st.header("Upload")
    uploaded = st.file_uploader("Transactions Excel (.xlsx)", type=["xlsx"])
    password = st.text_input("Password (only if file is encrypted)", type="password")
    include_benchmark = st.checkbox(f"Include benchmark ({BENCHMARK_TICKER})", value=True)
    st.divider()
    st.header("Cap buckets")
    large_crore = st.number_input("Large cap threshold (₹ crore)", min_value=1000.0, value=20000.0, step=500.0)
    mid_crore = st.number_input("Mid cap threshold (₹ crore)", min_value=100.0, value=5000.0, step=250.0)
    small_crore = st.number_input("Small cap threshold (₹ crore)", min_value=10.0, value=1000.0, step=50.0)

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
stats = stats_table(rets)
om = overlap_matrix(positions)

market_caps = _fetch_market_caps_cached(tickers) if tickers else pd.Series(dtype="float64")

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
    # Avoid pandas replace edge-cases on some builds by masking non-finite values explicitly.
    stats_disp = stats.copy()
    stats_disp = stats_disp.apply(pd.to_numeric, errors="ignore")
    for col in stats_disp.columns:
        if pd.api.types.is_numeric_dtype(stats_disp[col]):
            s = pd.to_numeric(stats_disp[col], errors="coerce")
            stats_disp[col] = s.mask(~pd.Series(np.isfinite(s), index=s.index), pd.NA)
    st.dataframe(stats_disp, use_container_width=True)

    st.subheader("Correlation (returns)")
    if rets.shape[1] >= 2:
        corr = rets.corr().round(2)
        st.dataframe(corr, use_container_width=True)
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(corr.values, aspect="auto")
        ax.set_xticks(np.arange(len(corr.columns)))
        ax.set_yticks(np.arange(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr.columns)
        for i in range(len(corr.index)):
            for j in range(len(corr.columns)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8, color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Correlation heatmap")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Need at least 2 clients to show correlation.")

    st.subheader("Overlap matrix (holdings)")
    if len(om.columns) >= 2:
        st.dataframe(om, use_container_width=True)
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(om.values, aspect="auto")
        ax.set_xticks(np.arange(len(om.columns)))
        ax.set_yticks(np.arange(len(om.columns)))
        ax.set_xticklabels(om.columns, rotation=45, ha="right")
        ax.set_yticklabels(om.columns)
        for i in range(len(om.index)):
            for j in range(len(om.columns)):
                ax.text(j, i, f"{om.iloc[i, j]:.1f}%", ha="center", va="center", fontsize=8, color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Portfolio overlap heatmap (%)")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Need at least 2 clients to show overlap.")

    st.subheader("Data quality")
    if dl.failed_tickers:
        st.warning(f"Price download failed for {len(dl.failed_tickers)} tickers.")
        st.code("\n".join(dl.failed_tickers))
    if missing_tickers:
        st.warning(f"Holdings contain {len(missing_tickers)} tickers missing from downloaded prices.")
        st.code("\n".join(sorted(missing_tickers)))
    if market_caps.empty:
        st.info("Market-cap data unavailable (cap split will show as Unknown).")

with client_tab:
    portfolios = [c for c in nav_df.columns if c and str(c).strip()]
    if not portfolios:
        st.error("No portfolios found in the uploaded data.")
        st.stop()

    portfolio = st.selectbox("Client / portfolio", portfolios)

    nav = nav_df[portfolio].dropna()
    pr = rets[portfolio].dropna() if portfolio in rets.columns else pd.Series(dtype=float)

    k1, k2, k3, k4 = st.columns(4)
    if portfolio in stats.index:
        row = stats.loc[portfolio]
        k1.metric("Annualized return", _fmt_pct(row.get("Annual Return")))
        k2.metric("Annualized vol", _fmt_pct(row.get("Volatility")))
        k3.metric("Sharpe", _fmt_float(row.get("Sharpe Ratio")))
        k4.metric("Max drawdown", _fmt_pct(row.get("Max Drawdown")))

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
    stats_row = stats.loc[portfolio] if portfolio in stats.index else None
    benchmark_returns = prices[BENCHMARK_TICKER].pct_change() if include_benchmark and BENCHMARK_TICKER in prices.columns else None

    st.subheader("Attribution & risk")
    attrib = None
    rc = None
    cap_split = None
    if not holdings.empty:
        attrib = performance_attribution(holdings, prices)
        rc = risk_contribution(holdings, prices)

        a1, a2 = st.columns(2)
        with a1:
            st.caption("Performance attribution (top/bottom)")
            if not attrib.empty:
                st.bar_chart(attrib.head(10))
            else:
                st.info("Not enough data for attribution.")
        with a2:
            st.caption("Risk contribution")
            if not rc.empty:
                st.bar_chart(rc.head(12))
            else:
                st.info("Not enough data for risk contribution.")

        st.subheader("Large/Mid/Small cap split")
        # market caps are keyed by ticker_yf (e.g. *.NS)
        cap_split = cap_split_weights(
            holdings,
            prices,
            market_caps,
            large_crore=float(large_crore),
            mid_crore=float(mid_crore),
            small_crore=float(small_crore),
        )
        if not cap_split.empty:
            st.bar_chart(cap_split)
        else:
            st.info("Cap split not available (missing market caps or holdings).")

    if report_choice == "PDF report":
        if holdings.empty or nav.empty or pr.empty:
            st.warning("Not enough data to generate a PDF report for this client.")
        else:
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
                cap_split=cap_split if isinstance(cap_split, pd.Series) and not cap_split.empty else None,
                risk_contrib=rc if isinstance(rc, pd.Series) and not rc.empty else None,
                attribution=attrib if isinstance(attrib, pd.Series) and not attrib.empty else None,
            )
            st.download_button(
                "Download PDF",
                data=pdf_bytes,
                file_name=f"Portfolio_{portfolio}.pdf",
                mime="application/pdf",
            )

    elif report_choice == "NAV history (CSV)":
        out = nav.to_frame(name="nav")
        st.download_button(
            "Download NAV CSV",
            data=_df_to_csv_bytes(out),
            file_name=f"NAV_{portfolio}.csv",
            mime="text/csv",
        )

    elif report_choice == "Daily returns (CSV)":
        out = pr.to_frame(name="returns")
        st.download_button(
            "Download returns CSV",
            data=_df_to_csv_bytes(out),
            file_name=f"Returns_{portfolio}.csv",
            mime="text/csv",
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
            )

    else:  # Transactions
        out = df[df["portfolio"] == portfolio].copy()
        st.download_button(
            "Download transactions CSV",
            data=_df_to_csv_bytes(out),
            file_name=f"Transactions_{portfolio}.csv",
            mime="text/csv",
        )
