from __future__ import annotations

import io
from datetime import date, datetime
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pathlib import Path

from portfolio_pipeline import get_corr_cmap


DARK_TEAL = colors.HexColor("#1B4F72")
ACCENT_TEAL = colors.HexColor("#1F618D")
WHITE = colors.white
LIGHT_GRAY = colors.HexColor("#F2F3F4")
MID_GRAY = colors.HexColor("#717D7E")
BLACK = colors.black

PAGE_W, PAGE_H = A4


def _style(name: str, **kw) -> ParagraphStyle:
    return ParagraphStyle(name, **kw)


BASE = getSampleStyleSheet()


def _ensure_unicode_fonts_registered() -> None:
    """
    Register DejaVu fonts (shipped with matplotlib) so unicode symbols like ₹ render correctly.
    """
    try:
        pdfmetrics.getFont("DejaVuSans")
        return
    except Exception:
        pass

    try:
        from matplotlib import font_manager

        regular = font_manager.findfont("DejaVu Sans")
        bold = font_manager.findfont("DejaVu Sans:style=normal:weight=bold")
        pdfmetrics.registerFont(TTFont("DejaVuSans", regular))
        pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", bold))
    except Exception:
        # If registration fails, fall back to core fonts.
        return


_ensure_unicode_fonts_registered()

TITLE_STYLE = _style(
    "ReportTitle",
    fontName="DejaVuSans-Bold" if "DejaVuSans-Bold" in pdfmetrics.getRegisteredFontNames() else "Helvetica-Bold",
    fontSize=24,
    textColor=ACCENT_TEAL,
    alignment=TA_CENTER,
    leading=28,
)
SUBTITLE_STYLE = _style(
    "ReportSubtitle",
    fontName="DejaVuSans" if "DejaVuSans" in pdfmetrics.getRegisteredFontNames() else "Helvetica-Oblique",
    fontSize=11,
    textColor=MID_GRAY,
    alignment=TA_CENTER,
    leading=16,
)
SECTION_STYLE = _style(
    "SectionHead",
    fontName="DejaVuSans-Bold" if "DejaVuSans-Bold" in pdfmetrics.getRegisteredFontNames() else "Helvetica-Bold",
    fontSize=12,
    textColor=ACCENT_TEAL,
    spaceBefore=10,
    spaceAfter=4,
)
BODY_STYLE = _style(
    "Body",
    fontName="DejaVuSans" if "DejaVuSans" in pdfmetrics.getRegisteredFontNames() else "Helvetica",
    fontSize=9,
    textColor=BLACK,
    leading=13,
)


def fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v*100:+.2f}%"


def fmt_inr(val: float) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    # Use unicode rupee sign; DejaVu font registration enables rendering.
    return f"₹{val:,.0f}"


def make_chart_image(fig, width_mm: float = 170, height_mm: float = 70) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    plt.close(fig)
    return Image(buf, width=width_mm * mm, height=height_mm * mm)


def growth_chart(returns: pd.Series, name: str, benchmark_returns: Optional[pd.Series] = None) -> Image:
    fig, ax = plt.subplots(figsize=(9, 3.2))
    cumret = (1 + returns).cumprod()
    ax.plot(cumret.index, cumret.values, color="#1B4F72", linewidth=1.8, label=name)
    if benchmark_returns is not None:
        bench = (1 + benchmark_returns.reindex(returns.index).dropna()).cumprod()
        ax.plot(bench.index, bench.values, color="#E67E22", linewidth=1.2, linestyle="--", label="Benchmark")
    ax.set_title(f"Portfolio Growth — {name}", fontsize=9, color="#1B4F72", fontweight="bold")
    ax.set_ylabel("Growth of ₹1", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return make_chart_image(fig, height_mm=65)


def drawdown_chart(nav: pd.Series, name: str) -> Image:
    rolling_max = nav.cummax()
    dd = nav / rolling_max - 1
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.fill_between(dd.index, dd.values, 0, color="#C0392B", alpha=0.5)
    ax.plot(dd.index, dd.values, color="#C0392B", linewidth=0.8)
    ax.set_title(f"Drawdown — {name}", fontsize=9, color="#1B4F72", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return make_chart_image(fig, height_mm=55)


def top_holdings_table(holdings: pd.DataFrame, prices: pd.DataFrame, n: int = 12) -> Optional[Table]:
    tickers = [t for t in holdings.columns if t in prices.columns]
    if not tickers:
        return None

    latest_h = holdings.iloc[-1][tickers]
    latest_p = prices.loc[prices.index[-1], tickers]
    mv = (latest_h * latest_p).sort_values(ascending=False).head(n)
    total = mv.sum()

    rows = [["Ticker", "Market Value (INR)", "Weight"]]
    for ticker, val in mv.items():
        rows.append([ticker.replace(".NS", "").replace(".BO", ""), fmt_inr(float(val)), f"{val/total*100:.1f}%"])

    t = Table(rows, colWidths=[70 * mm, 65 * mm, 30 * mm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), DARK_TEAL),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("FONTNAME", (0, 0), (-1, 0), TITLE_STYLE.fontName),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("FONTNAME", (0, 1), (-1, -1), BODY_STYLE.fontName),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
                ("GRID", (0, 0), (-1, -1), 0.3, MID_GRAY),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def sector_pie_image(holdings: pd.DataFrame, prices: pd.DataFrame, sector_map: dict[str, str]) -> Optional[Image]:
    tickers = [t for t in holdings.columns if t in prices.columns]
    if not tickers:
        return None

    latest_h = holdings.iloc[-1][tickers]
    latest_p = prices.loc[prices.index[-1], tickers]
    mv = latest_h * latest_p
    sector_df = pd.DataFrame({"Ticker": mv.index, "MV": mv.values})
    sector_df["Sector"] = sector_df["Ticker"].map(sector_map).fillna("Unknown")
    exp = sector_df.groupby("Sector")["MV"].sum().sort_values(ascending=False)
    if exp.empty:
        return None

    fig, ax = plt.subplots(figsize=(5, 5))
    wedge_colors = plt.cm.tab20.colors
    exp.plot.pie(
        ax=ax,
        autopct="%1.1f%%",
        startangle=90,
        colors=wedge_colors[: len(exp)],
        textprops={"fontsize": 7},
    )
    ax.set_title("Sector Exposure", fontsize=9, color="#1B4F72", fontweight="bold")
    ax.set_ylabel("")
    fig.tight_layout()
    return make_chart_image(fig, width_mm=90, height_mm=85)


def bar_series_image(series: pd.Series, title: str, *, top_n: int = 12) -> Optional[Image]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    s = s.head(top_n)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    s.sort_values(ascending=True).plot.barh(ax=ax, color="#1F618D")
    ax.set_title(title, fontsize=9, color="#1B4F72", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return make_chart_image(fig, height_mm=70)


def pie_series_image(series: pd.Series, title: str) -> Optional[Image]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    colorset = plt.cm.Set3.colors
    s.plot.pie(
        ax=ax,
        autopct="%1.0f%%",
        startangle=90,
        colors=colorset[: len(s)],
        textprops={"fontsize": 8},
    )
    ax.set_title(title, fontsize=9, color="#1B4F72", fontweight="bold")
    ax.set_ylabel("")
    fig.tight_layout()
    return make_chart_image(fig, width_mm=95, height_mm=80)


def corr_heatmap_image(corr: pd.DataFrame, title: str) -> Optional[Image]:
    if corr is None or corr.empty:
        return None
    df = corr.copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if df.shape[0] < 2 or df.shape[1] < 2:
        return None

    cmap = get_corr_cmap()

    fig, ax = plt.subplots(figsize=(8.5, 6))
    im = ax.imshow(df.values.astype(float), aspect="auto", cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_xticklabels([str(c).replace(".NS", "") for c in df.columns], rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels([str(i).replace(".NS", "") for i in df.index], fontsize=6)
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            v = df.iat[i, j]
            if pd.isna(v):
                continue
            text_color = "white" if abs(float(v)) > 0.65 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5, color=text_color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=9, color="#1B4F72", fontweight="bold")
    fig.tight_layout()
    return make_chart_image(fig, width_mm=170, height_mm=120)


def returns_histogram_image(returns: pd.Series, title: str) -> Optional[Image]:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) < 10:
        return None
    var99_val = float(r.quantile(0.01))
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.hist(r.values, bins=40, color="#1F618D", alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axvline(var99_val, color="#C0392B", linewidth=1.5, linestyle="--",
               label=f"99% VaR: {var99_val*100:.2f}%")
    ax.set_title(title, fontsize=9, color="#1B4F72", fontweight="bold")
    ax.set_xlabel("Daily return")
    ax.set_ylabel("Frequency")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return make_chart_image(fig, width_mm=170, height_mm=80)


def normalize_cap_split_for_report(cap_split: Optional[pd.Series]) -> Optional[pd.Series]:
    """
    Streamlit app may produce 3-5 buckets; for the PDF keep it clean:
    - Drop Unknown if present
    """
    if cap_split is None:
        return None
    s = pd.to_numeric(cap_split, errors="coerce").dropna()
    if s.empty:
        return None
    s = s.copy()
    if "Unknown" in s.index:
        s = s.drop(index=["Unknown"])
    total = float(s.sum())
    if total > 0:
        s = s / total
    return s.sort_values(ascending=False)


def summary_text(
    *,
    portfolio_name: str,
    stats_row: Optional[pd.Series],
    nav: pd.Series,
    winners: Optional[pd.Series],
    laggards: Optional[pd.Series],
) -> str:
    nav = nav.dropna()
    start = nav.index.min().date().isoformat() if not nav.empty else "N/A"
    end = nav.index.max().date().isoformat() if not nav.empty else "N/A"
    total_ret = (nav.iloc[-1] / nav.iloc[0] - 1) if len(nav) >= 2 else np.nan

    ann = stats_row.get("Annualized Return") if stats_row is not None else np.nan
    vol = stats_row.get("Annualized Volatility") if stats_row is not None else np.nan
    sharpe = stats_row.get("Sharpe") if stats_row is not None else np.nan
    sortino = stats_row.get("Sortino") if stats_row is not None else np.nan
    beta = stats_row.get("Beta (Nifty 500)") if stats_row is not None else np.nan
    alpha = stats_row.get("Jensen Alpha") if stats_row is not None else np.nan
    var99 = stats_row.get("VaR 99% (Daily)") if stats_row is not None else np.nan
    mdd = stats_row.get("Max Drawdown") if stats_row is not None else np.nan

    def _pct(v):
        return "N/A" if v is None or pd.isna(v) or v in (float("inf"), float("-inf")) else f"{v*100:.1f}%"

    def _flt(v):
        return "N/A" if v is None or pd.isna(v) or v in (float("inf"), float("-inf")) else f"{float(v):.2f}"

    w_txt = ""
    if winners is not None and not winners.empty:
        w_txt = ", ".join([f"{str(i).replace('.NS','')}" for i in winners.index[:3]])
    l_txt = ""
    if laggards is not None and not laggards.empty:
        l_txt = ", ".join([f"{str(i).replace('.NS','')}" for i in laggards.index[:3]])

    parts = [
        f"This report summarises {portfolio_name}'s equity portfolio performance from {start} to {end}.",
        f"Over the period, portfolio NAV returned {_pct(total_ret)}.",
        f"Annualised return: {_pct(ann)}; annualised volatility: {_pct(vol)}; Sharpe: {_flt(sharpe)}; Sortino: {_flt(sortino)}; "
        f"Beta (Nifty 500): {_flt(beta)}; Jensen's alpha: {_pct(alpha)}; 99% daily VaR: {_pct(var99)}; max drawdown: {_pct(mdd)}.",
    ]
    if w_txt:
        parts.append(f"Top performers over the period included {w_txt}.")
    if l_txt:
        parts.append(f"Key laggards included {l_txt}.")
    return " ".join(parts)


def generate_portfolio_pdf_bytes(
    *,
    portfolio_name: str,
    report_date: date | datetime,
    nav: pd.Series,
    returns: pd.Series,
    holdings: pd.DataFrame,
    prices: pd.DataFrame,
    stats_row: Optional[pd.Series] = None,
    benchmark_returns: Optional[pd.Series] = None,
    sector_map: Optional[dict[str, str]] = None,
    risk_contrib: Optional[pd.Series] = None,
    attribution: Optional[pd.Series] = None,
    winners: Optional[pd.Series] = None,
    laggards: Optional[pd.Series] = None,
    stock_corr: Optional[pd.DataFrame] = None,
    returns_hist: Optional[pd.Series] = None,
) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title=f"Portfolio Report - {portfolio_name}",
    )

    sector_map = sector_map or {}

    story = []
    # Support both .png and .jpg logo filenames
    _assets = Path(__file__).resolve().parent / "assets"
    logo_file = next(
        (p for p in [
            _assets / "vika_logo.png",
            _assets / "vika_logo.jpg",
            _assets / "Vika_Logo.png",
            _assets / "Vika_Logo.jpg",
        ] if p.exists()),
        _assets / "vika_logo.png",  # fallback path (may not exist — caught below)
    )

    def _page_header(canvas_obj, doc_obj):
        if logo_file.exists():
            try:
                w = 42 * mm
                h = 17 * mm
                # TOP-LEFT corner
                x = doc_obj.leftMargin
                y = PAGE_H - doc_obj.topMargin - h
                canvas_obj.drawImage(
                    str(logo_file),
                    x,
                    y,
                    width=w,
                    height=h,
                    preserveAspectRatio=True,
                    mask="auto",
                )
            except Exception:
                pass

    story.append(Spacer(1, 12 * mm))
    story.append(Paragraph("Portfolio Report", TITLE_STYLE))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(str(portfolio_name), SUBTITLE_STYLE))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(f"Report date: {pd.to_datetime(report_date).date().isoformat()}", SUBTITLE_STYLE))
    story.append(Spacer(1, 8 * mm))
    story.append(HRFlowable(width="100%", color=ACCENT_TEAL, thickness=0.7))
    story.append(Spacer(1, 8 * mm))

    story.append(Paragraph("Executive Summary", SECTION_STYLE))
    story.append(HRFlowable(width="100%", color=ACCENT_TEAL, thickness=0.5))
    story.append(Spacer(1, 4 * mm))
    story.append(
        Paragraph(
            summary_text(
                portfolio_name=str(portfolio_name),
                stats_row=stats_row,
                nav=nav,
                winners=winners,
                laggards=laggards,
            ),
            BODY_STYLE,
        )
    )
    story.append(Spacer(1, 6 * mm))

    story.append(Paragraph("Summary Statistics", SECTION_STYLE))
    story.append(HRFlowable(width="100%", color=ACCENT_TEAL, thickness=0.5))
    story.append(Spacer(1, 4 * mm))

    if stats_row is not None:
        rows = [
            ["Annualized Return", fmt_pct(float(stats_row.get("Annualized Return")))],
            ["Annualized Volatility", fmt_pct(float(stats_row.get("Annualized Volatility")))],
            ["Sharpe", f"{stats_row.get('Sharpe'):.2f}" if pd.notna(stats_row.get("Sharpe")) else "N/A"],
            ["Sortino", f"{stats_row.get('Sortino'):.2f}" if pd.notna(stats_row.get("Sortino")) else "N/A"],
            [
                "Beta (Nifty 500)",
                f"{stats_row.get('Beta (Nifty 500)'):.2f}" if pd.notna(stats_row.get("Beta (Nifty 500)")) else "N/A",
            ],
            ["Jensen Alpha", fmt_pct(float(stats_row.get("Jensen Alpha")))],
            ["Treynor", f"{stats_row.get('Treynor'):.2f}" if pd.notna(stats_row.get("Treynor")) else "N/A"],
            ["VaR 99% (Daily)", fmt_pct(float(stats_row.get("VaR 99% (Daily)")))],
            ["Max Drawdown", fmt_pct(float(stats_row.get("Max Drawdown")))],
        ]
        t = Table([["Metric", "Value"], *rows], colWidths=[70 * mm, 50 * mm])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), DARK_TEAL),
                    ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
                    ("GRID", (0, 0), (-1, -1), 0.3, MID_GRAY),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(t)
    else:
        story.append(Paragraph("Statistics not available.", BODY_STYLE))

    story.append(PageBreak())

    story.append(Paragraph("Performance", SECTION_STYLE))
    story.append(HRFlowable(width="100%", color=ACCENT_TEAL, thickness=0.5))
    story.append(Spacer(1, 4 * mm))
    story.append(growth_chart(returns.dropna(), str(portfolio_name), benchmark_returns=benchmark_returns))
    story.append(Spacer(1, 6 * mm))
    story.append(drawdown_chart(nav.dropna(), str(portfolio_name)))

    story.append(PageBreak())

    story.append(Paragraph("Holdings & Allocation", SECTION_STYLE))
    story.append(HRFlowable(width="100%", color=ACCENT_TEAL, thickness=0.5))
    story.append(Spacer(1, 4 * mm))
    tbl = top_holdings_table(holdings, prices, n=15)
    if tbl is not None:
        story.append(tbl)
    else:
        story.append(Paragraph("No holdings available.", BODY_STYLE))

    story.append(Spacer(1, 10 * mm))
    story.append(Paragraph("Sector Exposure", SECTION_STYLE))
    story.append(HRFlowable(width="100%", color=ACCENT_TEAL, thickness=0.5))
    story.append(Spacer(1, 4 * mm))
    sec = sector_pie_image(holdings, prices, sector_map=sector_map)
    if sec is not None:
        story.append(sec)
    else:
        story.append(Paragraph("Sector map not available.", BODY_STYLE))

    story.append(PageBreak())

    story.append(Paragraph("Risk & Attribution", SECTION_STYLE))
    story.append(HRFlowable(width="100%", color=ACCENT_TEAL, thickness=0.5))
    story.append(Spacer(1, 4 * mm))

    rc_img = bar_series_image(risk_contrib, "Risk contribution (top)", top_n=12) if risk_contrib is not None else None
    if rc_img is not None:
        story.append(rc_img)
    else:
        story.append(Paragraph("Risk contribution not available.", BODY_STYLE))

    story.append(Spacer(1, 8 * mm))
    at_img = (
        bar_series_image(attribution, "Performance attribution (top)", top_n=12)
        if attribution is not None
        else None
    )
    if at_img is not None:
        story.append(at_img)
    else:
        story.append(Paragraph("Performance attribution not available.", BODY_STYLE))

    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph("Return Distribution", SECTION_STYLE))
    story.append(HRFlowable(width="100%", color=ACCENT_TEAL, thickness=0.5))
    story.append(Spacer(1, 4 * mm))
    hist_img = returns_histogram_image(returns_hist, "Histogram of daily returns") if returns_hist is not None else None
    if hist_img is not None:
        story.append(hist_img)
    else:
        story.append(Paragraph("Return distribution not available.", BODY_STYLE))

    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph("Stock Correlation", SECTION_STYLE))
    story.append(HRFlowable(width="100%", color=ACCENT_TEAL, thickness=0.5))
    story.append(Spacer(1, 4 * mm))
    corr_img = corr_heatmap_image(stock_corr, "Correlation heatmap (stocks)") if stock_corr is not None else None
    if corr_img is not None:
        story.append(corr_img)
    else:
        story.append(Paragraph("Correlation heatmap not available.", BODY_STYLE))

    if winners is not None and not winners.empty and laggards is not None and not laggards.empty:
        story.append(Spacer(1, 8 * mm))
        story.append(Paragraph("Top Movers (Stocks)", SECTION_STYLE))
        story.append(HRFlowable(width="100%", color=ACCENT_TEAL, thickness=0.5))
        story.append(Spacer(1, 4 * mm))
        rows = [["Top performers", "Return"], *[[i.replace(".NS", ""), f"{v*100:.1f}%"] for i, v in winners.items()]]
        rows2 = [["Top laggards", "Return"], *[[i.replace(".NS", ""), f"{v*100:.1f}%"] for i, v in laggards.items()]]
        t1 = Table(rows, colWidths=[70 * mm, 30 * mm])
        t2 = Table(rows2, colWidths=[70 * mm, 30 * mm])
        for t in (t1, t2):
            t.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), DARK_TEAL),
                        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                        ("FONTNAME", (0, 0), (-1, 0), TITLE_STYLE.fontName),
                        ("FONTNAME", (0, 1), (-1, -1), BODY_STYLE.fontName),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
                        ("GRID", (0, 0), (-1, -1), 0.3, MID_GRAY),
                        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                    ]
                )
            )
        story.append(t1)
        story.append(Spacer(1, 4 * mm))
        story.append(t2)

    story.append(PageBreak())

    story.append(Paragraph("Disclaimer", SECTION_STYLE))
    story.append(HRFlowable(width="100%", color=ACCENT_TEAL, thickness=0.5))
    story.append(Spacer(1, 4 * mm))
    story.append(
        Paragraph(
            "Equity investments are subject to market risks. Past performance is not indicative of future returns. "
            "This report is for information purposes only and should not be construed as investment advice.",
            BODY_STYLE,
        )
    )

    doc.build(story, onFirstPage=_page_header, onLaterPages=_page_header)
    return buf.getvalue()
