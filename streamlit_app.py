"""
VIKA Wealth — SIF Performance Tracker

Data, refreshed at most every 6 hours:
  1. the saved copy  data/sif_nav_history.csv + data/sif_meta.csv  (your own history, never lost)
  2. sif.tigzig.com  full SIF NAV history since launch (free mirror of AMFI)
  3. AMFI            today's SIF NAVs (portal.amfiindia.com/spages/SIF_NAVAll.txt)
New rows are appended and the saved copy is written back: to the `data` branch of your GitHub
repo when [github] secrets are set (Streamlit Cloud), otherwise to the local data/ folder.
"""
import base64
import html
import io
import os
import re
import urllib.request
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ───────────────────────── config ─────────────────────────
GOLD, GREEN, RED, GREY = "#C9A227", "#3FB950", "#F85149", "#8B949E"
NAV_URL = "https://sif.tigzig.com/sif_nav.csv"
META_URL = "https://sif.tigzig.com/sif_meta.csv"
AMFI_LATEST_URL = "https://portal.amfiindia.com/spages/SIF_NAVAll.txt"
ISSUE_PRICE = 10.0
LOOKBACKS = {"1M": 30, "3M": 91, "6M": 182, "1Y": 365}
MIN_OBS = 20
HEDGE_MIN_OBS = 40  # days needed before a hedge verdict
# SEBI category rules (text only). Each fund's own limits come from its ISID (PROMISES below).
MANDATE = {
    "Equity Long-Short Fund": "≥80% equity · unhedged shorts ≤25% of NAV",
    "Equity Ex-Top 100 Long-Short Fund": "≥65% in stocks outside the top 100 · unhedged shorts ≤25%",
    "Sector Rotation Long-Short Fund": "≥80% equity in ≤4 sectors · unhedged shorts ≤25% (sector level)",
    "Hybrid Long-Short Fund": "≥25% equity and ≥25% debt · unhedged shorts ≤25%",
    "Active Asset Allocator Long-Short Fund": "Dynamic across equity, debt, REITs/InvITs, commodity derivatives · unhedged shorts ≤25%",
}
# Benchmark building blocks. Each is fetched from the best real source available, in this order:
#   1. NSE official TRI (niftyindices.com)
#   2. NAV of an index fund / ETF that tracks it (AMFI data via mfapi.in): the TRI net of a small expense ratio
#   3. Yahoo price index (no dividends; labelled "price only")
# No assumed returns are added anywhere.
COMPONENTS = {
    "Nifty 50": dict(nse="NIFTY 50", mf=118482, mf_name="Bandhan Nifty 50 Index Fund", yahoo="^NSEI"),
    "Nifty 100": dict(nse="NIFTY 100", mf=149868, mf_name="HDFC Nifty 100 Index Fund", yahoo="^CNX100"),
    "Nifty 200": dict(nse="NIFTY 200", mf=153329, mf_name="Groww Nifty 200 ETF", yahoo="^CNX200"),
    "Nifty 500": dict(nse="NIFTY 500", mf=147625, mf_name="Motilal Oswal Nifty 500 Index Fund", yahoo="^CRSLDX"),
    "Nifty MidSmallcap 400": dict(nse="NIFTY MIDSMALLCAP 400", mf=154027,
                                  mf_name="Navi Nifty MidSmallcap 400 Index Fund", yahoo=None),
    "BSE 500": dict(nse=None, mf=151728, mf_name="HDFC BSE 500 Index Fund", yahoo="BSE-500.BO"),
    "BSE 200": dict(nse="NIFTY 200", mf=153329, mf_name="Groww Nifty 200 ETF", yahoo="BSE-200.BO",
                    note="no BSE 200 index fund exists, so Nifty 200 stands in"),
    "Sensex": dict(nse=None, mf=119065, mf_name="HDFC BSE Sensex Index Fund", yahoo="^BSESN"),
    "Debt": dict(nse=None, mf=151365, mf_name="Edelweiss CRISIL IBX 50:50 Gilt+SDL Short Duration Index Fund",
                 yahoo=None),
    "Gold": dict(nse=None, mf=140088, mf_name="Nippon India ETF Gold BeES", yahoo=None),
    "Silver": dict(nse=None, mf=149758, mf_name="Nippon India Silver ETF", yahoo=None),
}
NON_EQUITY = {"Debt", "Gold", "Silver"}


def _equity_component(t: str) -> str:
    if "midsmall" in t:
        return "Nifty MidSmallcap 400"
    if "bse 500" in t:
        return "BSE 500"
    if "bse 200" in t:
        return "BSE 200"
    if "sensex" in t:
        return "Sensex"
    if "500" in t:
        return "Nifty 500"
    if "nifty 200" in t:
        return "Nifty 200"
    if "nifty 100" in t:
        return "Nifty 100"
    return "Nifty 50"


def parse_benchmark(text: str):
    """ISID benchmark text -> list of (component, weight).
    '50% Nifty 500 TRI + 40% Nifty Composite Debt + 7% gold + 3% silver' -> 4 parts.
    Hybrid composites: 50:50 / 50+50 -> 50% Nifty 50 + 50% debt; 85+15 conservative -> 15/85.
    iCOMDEX (multi-commodity) has no fund tracking it; it is split equally between gold and silver."""
    t = str(text).lower()
    parts = [x.strip() for x in t.split("+")]
    if len(parts) > 1 and all(re.match(r"\d+(\.\d+)?%", x) for x in parts):
        comps = []
        for x in parts:
            w = float(re.match(r"(\d+(?:\.\d+)?)%", x).group(1)) / 100
            if "gold" in x:
                comps.append(("Gold", w))
            elif "silver" in x:
                comps.append(("Silver", w))
            elif "icomdex" in x or "commodit" in x:
                comps += [("Gold", w / 2), ("Silver", w / 2)]
            elif any(k in x for k in ("bond", "debt", "short term", "short duration")):
                comps.append(("Debt", w))
            else:
                comps.append((_equity_component(x), w))
        return comps
    if "85+15" in t:
        return [("Nifty 50", 0.15), ("Debt", 0.85)]
    if "50:50" in t or "50+50" in t or "hybrid" in t:
        return [("Nifty 50", 0.5), ("Debt", 0.5)]
    return [(_equity_component(t), 1.0)]


VERDICT_COLOR = {"🟢": GREEN, "🟡": GOLD, "🔴": RED, "⚠️": "#E3842B", "⏳": GREY}
SHORT = {  # short labels for the category picker
    "Equity Long-Short Fund": "Equity L/S",
    "Equity Ex-Top 100 Long-Short Fund": "Ex-Top 100 L/S",
    "Sector Rotation Long-Short Fund": "Sector Rotation L/S",
    "Hybrid Long-Short Fund": "Hybrid L/S",
    "Active Asset Allocator Long-Short Fund": "Active Asset Allocator L/S",
}

st.set_page_config(page_title="SIF Tracker · VIKA Wealth", page_icon="📊", layout="wide")


def plot_layout(fig, title="", h=420):
    fig.update_layout(title=dict(text=title, font=dict(size=15)), height=h,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=45 if title else 10, b=10),
                      legend=dict(orientation="h", y=-0.12))
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.18)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)")
    return fig


# ───────────────────────── data ─────────────────────────
def fund_name(name: str) -> str:
    """'qsif Equity Long Short Fund - Growth Option - Direct Plan' -> 'qsif Equity Long Short Fund'"""
    n = re.sub(r"(?i)\s*[-–]?\s*\b(direct|regular)\b(\s*plan)?", "", name)
    n = re.sub(r"(?i)\s*[-–]?\s*\b(growth|idcw|dividend)\b(\s*option)?", "", n)
    return re.sub(r"[\s\-–]+$", "", " ".join(n.split()))


def parse_amfi_latest(text: str) -> pd.DataFrame:
    rows, cat, amc = [], "", ""
    for line in (l.strip() for l in text.splitlines()):
        if not line:
            continue
        if ";" not in line:
            if "(" in line and "Scheme" in line:
                cat = line[line.find("(") + 1: line.rfind(")")].split(" - ")[-1].strip()
            else:
                amc = line
            continue
        p = [x.strip() for x in line.split(";")]
        if p[0].lower().startswith("scheme code"):
            continue
        rows.append(dict(code=p[0], date=p[7], nav=p[6], scheme_name=p[3], amc=amc, category=cat,
                         plan="Direct" if "direct" in p[4].lower() else "Regular" if p[4] else "",
                         option="IDCW" if "idcw" in p[5].lower() else "Growth"))
    df = pd.DataFrame(rows)
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y", errors="coerce")
    return df.dropna(subset=["nav", "date"])


# ── saved copy: GitHub repo (Streamlit Cloud) or local data/ folder (your Mac) ──
NAV_FILE, META_FILE = "data/sif_nav_history.csv", "data/sif_meta.csv"


def _gh():
    """GitHub settings from Streamlit secrets, or None. Accepts 'owner/repo' or a full GitHub URL."""
    try:
        g = st.secrets["github"]
        repo = str(g["repo"]).strip().rstrip("/")
        repo = re.sub(r"^(https?://)?(www\.)?github\.com/", "", repo)
        repo = re.sub(r"\.git$", "", repo)
        return {"token": str(g["token"]).strip(), "repo": repo, "branch": str(g.get("branch", "data")).strip()}
    except Exception:
        return None


def _gh_api(method, url, cfg, **kw):
    import requests
    h = {"Authorization": f"Bearer {cfg['token']}", "Accept": "application/vnd.github+json",
         "X-GitHub-Api-Version": "2022-11-28"}
    h.update(kw.pop("headers", {}))
    return requests.request(method, f"https://api.github.com/repos/{cfg['repo']}{url}",
                            headers=h, timeout=60, **kw)


def _gh_reason(r, cfg, step):
    """Plain-language explanation of a failed GitHub call."""
    try:
        msg = r.json().get("message", "")
    except Exception:
        msg = r.text[:200]
    hint = {
        401: "the token is invalid or has expired. Create a new one and paste it into Secrets.",
        403: (f"the token can't write to {cfg['repo']}. For a fine-grained token: Repository access → "
              "Only select repositories → this repo, and Permissions → Contents → Read and write. "
              "If the repo belongs to an organisation, the organisation may also need to approve the token."),
        404: (f"GitHub can't find {cfg['repo']} with this token. Check that repo = \"owner/name\" in Secrets "
              "matches exactly, and that the token was given access to this repo."),
        409: "the repository is empty. Add any file (e.g. a README) to the main branch first.",
        422: "GitHub rejected the request.",
    }.get(r.status_code, "unexpected response.")
    return f"{step} failed (HTTP {r.status_code}): {hint} GitHub said: \"{msg}\""


def _ensure_branch(cfg):
    """Create the data branch from the default branch if needed. Returns error text or None."""
    r = _gh_api("GET", f"/git/ref/heads/{cfg['branch']}", cfg)
    if r.status_code == 200:
        return None
    repo = _gh_api("GET", "", cfg)
    if repo.status_code != 200:
        return _gh_reason(repo, cfg, "Opening the repo")
    default = repo.json()["default_branch"]
    ref = _gh_api("GET", f"/git/ref/heads/{default}", cfg)
    if ref.status_code != 200:
        return _gh_reason(ref, cfg, f"Reading the {default} branch")
    made = _gh_api("POST", "/git/refs", cfg, json={"ref": f"refs/heads/{cfg['branch']}",
                                                   "sha": ref.json()["object"]["sha"]})
    if made.status_code not in (200, 201):
        return _gh_reason(made, cfg, f"Creating the '{cfg['branch']}' branch")
    return None


SAVE_LOG = {"read": None, "write": None}  # last error messages, shown in Data status


def read_saved(path):
    cfg = _gh()
    try:
        if cfg:
            r = _gh_api("GET", f"/contents/{path}", cfg, params={"ref": cfg["branch"]},
                        headers={"Accept": "application/vnd.github.raw"})
            if r.status_code == 200:
                return pd.read_csv(io.StringIO(r.text))
            if r.status_code != 404:  # 404 just means nothing saved yet
                SAVE_LOG["read"] = _gh_reason(r, cfg, f"Reading {path}")
            return None
        return pd.read_csv(path) if os.path.exists(path) else None
    except Exception as e:
        SAVE_LOG["read"] = f"Reading {path} failed: {e}"
        return None


def write_saved(path, df, message):
    """Overwrite the saved CSV. Returns (ok, error text or None)."""
    text = df.to_csv(index=False, date_format="%Y-%m-%d")
    cfg = _gh()
    try:
        if not cfg:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "w").write(text)
            return True, None
        err = _ensure_branch(cfg)
        if err:
            return False, err
        cur = _gh_api("GET", f"/contents/{path}", cfg, params={"ref": cfg["branch"]})
        body = {"message": message, "branch": cfg["branch"],
                "content": base64.b64encode(text.encode()).decode()}
        if cur.status_code == 200:
            body["sha"] = cur.json()["sha"]
        elif cur.status_code != 404:
            return False, _gh_reason(cur, cfg, f"Checking {path}")
        put = _gh_api("PUT", f"/contents/{path}", cfg, json=body)
        if put.status_code in (200, 201):
            return True, None
        return False, _gh_reason(put, cfg, f"Saving {path}")
    except Exception as e:
        return False, f"Saving {path} failed: {e}"


def gh_connection_test():
    """Step-by-step check of the GitHub setup. Returns list of (ok, text)."""
    cfg = _gh()
    if not cfg:
        return [(False, "No [github] section in Secrets, so saving goes to the app's temporary disk.")]
    out = [(True, f"Secrets found · repo `{cfg['repo']}` · branch `{cfg['branch']}`")]
    if "/" not in cfg["repo"] or cfg["repo"].count("/") != 1:
        return out + [(False, "repo should look like \"owner/name\", e.g. \"vishnu-vika/sif-tracker\".")]
    try:
        r = _gh_api("GET", "", cfg)
        if r.status_code != 200:
            return out + [(False, _gh_reason(r, cfg, "Opening the repo"))]
        out.append((True, f"Token can see the repo (default branch `{r.json()['default_branch']}`)"))
        err = _ensure_branch(cfg)
        if err:
            return out + [(False, err)]
        out.append((True, f"Branch `{cfg['branch']}` is ready"))
        ok, err = write_saved("data/_connection_test.txt",
                              pd.DataFrame({"checked_at": [pd.Timestamp.now(tz="Asia/Kolkata").isoformat()]}),
                              "Connection test from SIF tracker")
        out.append((ok, "Test file written to the data branch: saving works" if ok else err))
    except Exception as e:
        out.append((False, f"Could not reach GitHub: {e}"))
    return out


def _code(x):
    """'SIF-120' / '120' / 120 -> 120"""
    return pd.to_numeric(pd.Series(x).astype(str).str.replace("SIF-", "", regex=False), errors="coerce")


@st.cache_data(ttl=6 * 3600, show_spinner="Updating SIF NAV history…")
def load_data():
    """Saved copy + whatever the live sources add today; saves back only when there is new data.
    Returns (nav, meta, status dict)."""
    status = {"mirror": False, "amfi": False, "saved_rows": 0, "added": 0, "wrote": None,
              "error": None, "read_error": None}
    SAVE_LOG["read"] = None

    saved_nav = read_saved(NAV_FILE)
    saved_meta = read_saved(META_FILE)
    navs = [saved_nav] if saved_nav is not None else []
    metas = [saved_meta] if saved_meta is not None else []
    status["saved_rows"] = 0 if saved_nav is None else len(saved_nav)
    status["read_error"] = SAVE_LOG["read"]

    try:  # 1) full history from the mirror
        n = pd.read_csv(NAV_URL, usecols=["scheme_code", "date", "nav"])
        m = pd.read_csv(META_URL)
        navs.append(n.rename(columns={"scheme_code": "code"}))
        # the mirror also has its own long 'category' column: keep only what we need, then rename
        m = m[["scheme_code", "scheme_name", "amc", "category_sub", "scheme_plan", "scheme_option"]]
        metas.append(m.rename(columns={"scheme_code": "code", "category_sub": "category",
                                       "scheme_plan": "plan", "scheme_option": "option"}))
        status["mirror"] = True
    except Exception:
        pass
    try:  # 2) today's NAVs straight from AMFI
        req = urllib.request.Request(AMFI_LATEST_URL, headers={"User-Agent": "Mozilla/5.0"})
        live = parse_amfi_latest(urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "ignore"))
        navs.append(live[["code", "date", "nav"]])
        metas.append(live)
        status["amfi"] = True
    except Exception:
        pass
    if not navs:
        st.error("No saved history yet and both live sources are unreachable. Try again later.")
        st.stop()

    nav = pd.concat(navs, ignore_index=True)
    nav["code"] = _code(nav["code"]).values
    nav["date"] = pd.to_datetime(nav["date"], errors="coerce")
    nav["nav"] = pd.to_numeric(nav["nav"], errors="coerce")
    nav = (nav.dropna().astype({"code": int})
              .drop_duplicates(["code", "date"], keep="first")   # saved copy wins on overlap
              .sort_values(["code", "date"]).reset_index(drop=True))

    cols = ["code", "scheme_name", "amc", "category", "plan", "option"]
    meta = pd.concat([m.loc[:, ~m.columns.duplicated()].reindex(columns=cols) for m in metas],
                     ignore_index=True)
    meta["code"] = _code(meta["code"]).values
    meta = meta.dropna(subset=["code"]).astype({"code": int})
    for c in cols[1:]:
        meta[c] = meta[c].fillna("").astype(str)
    # prefer the richest row per scheme (one that has category + plan filled)
    meta["filled"] = meta[["category", "plan", "amc"]].ne("").sum(axis=1)
    meta = (meta.sort_values("filled", ascending=False).drop_duplicates("code")
                .drop(columns="filled").sort_values("code").reset_index(drop=True))

    status["added"] = len(nav) - status["saved_rows"]
    if status["added"] > 0:
        stamp = f"{nav['date'].max():%Y-%m-%d}"
        ok1, err1 = write_saved(NAV_FILE, nav, f"SIF NAVs to {stamp} (+{status['added']} rows)")
        ok2, err2 = (write_saved(META_FILE, meta, f"SIF scheme master {stamp}") if ok1 else (False, None))
        status["wrote"] = ok1 and ok2
        status["error"] = err1 or err2
    meta["fund"] = meta["scheme_name"].map(fund_name)
    return nav, meta, status


@st.cache_data(ttl=6 * 3600, show_spinner="Loading benchmark…")
def load_benchmark(ticker: str, start) -> pd.Series:
    try:
        import yfinance as yf
        s = yf.download(ticker, start=start - timedelta(days=10), progress=False, auto_adjust=True)["Close"]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        return s.dropna()
    except Exception:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))


def _empty():
    return pd.Series(dtype=float, index=pd.DatetimeIndex([]))


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def nse_tri(name: str, start) -> pd.Series:
    """Official NSE total return index from niftyindices.com, fetched in yearly chunks."""
    import json as _json
    import requests
    hdr = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json; charset=UTF-8",
           "Origin": "https://www.niftyindices.com", "Referer": "https://www.niftyindices.com/reports/historical-data"}
    out, a, end = [], pd.Timestamp(start) - timedelta(days=10), pd.Timestamp.today().normalize()
    try:
        while a <= end:
            b = min(a + timedelta(days=360), end)
            cinfo = _json.dumps({"name": name, "startDate": a.strftime("%d-%b-%Y"),
                                 "endDate": b.strftime("%d-%b-%Y"), "indexName": name})
            r = requests.post("https://www.niftyindices.com/Backpage.aspx/getTotalReturnIndexString",
                              json={"cinfo": cinfo}, headers=hdr, timeout=30)
            df = pd.DataFrame(_json.loads(r.json()["d"]))
            if not df.empty:
                dcol = next(c for c in df.columns if "date" in c.lower())
                vcol = next(c for c in df.columns if "return" in c.lower() or "tri" in c.lower())
                out.append(pd.Series(pd.to_numeric(df[vcol].astype(str).str.replace(",", ""), errors="coerce").values,
                                     index=pd.to_datetime(df[dcol], format="mixed", dayfirst=True)))
            a = b + timedelta(days=1)
        s = pd.concat(out).dropna().sort_index()
        return s[~s.index.duplicated()]
    except Exception:
        return _empty()


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def mf_nav(code: int) -> pd.Series:
    """Daily NAV of any Indian mutual fund / ETF from AMFI data (mfapi.in)."""
    try:
        import requests
        d = requests.get(f"https://api.mfapi.in/mf/{code}", timeout=30).json()["data"]
        s = pd.Series(pd.to_numeric([x["nav"] for x in d], errors="coerce"),
                      index=pd.to_datetime([x["date"] for x in d], format="%d-%m-%Y"))
        s = s[s > 0].dropna().sort_index()
        return s[~s.index.duplicated()]
    except Exception:
        return _empty()


@st.cache_data(ttl=12 * 3600, show_spinner="Loading benchmark data…")
def component_series(name: str, start):
    """(series, source label) for one benchmark building block, best real source first."""
    c = COMPONENTS[name]
    since = pd.Timestamp(start) - timedelta(days=10)
    note = f" ({c['note']})" if c.get("note") else ""
    if c["nse"]:
        s = nse_tri(c["nse"], start)
        s = s[s.index >= since]
        if len(s) > MIN_OBS:
            return s, f"NSE {c['nse'].title()} TRI, official{note}"
    if c["mf"]:
        s = mf_nav(c["mf"])
        s = s[s.index >= since]
        if len(s) > MIN_OBS:
            return s, f"{c['mf_name']} NAV (tracks the TRI, net of its expense ratio){note}"
    if c["yahoo"]:
        s = load_benchmark(c["yahoo"], start)
        if len(s) > MIN_OBS:
            return s, f"{name} price index, price only (no dividends)"
    return _empty(), "unavailable"


def pick_variants(meta: pd.DataFrame, plan: str) -> pd.DataFrame:
    """One row per fund: the chosen plan, Growth option where available."""
    m = meta[meta["plan"].str.lower().eq(plan.lower()) | meta["plan"].eq("")].copy()
    m["prio"] = np.where(m["option"].str.lower().str.contains("growth"), 0, 1)
    m["blank_plan"] = m["plan"].eq("")
    return m.sort_values(["prio", "blank_plan"]).drop_duplicates(["amc", "fund"])


# ───────────────────────── benchmarks (from each fund's ISID) ─────────────────────────
def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower().replace("long short", "long-short"))


@st.cache_data
def load_promises() -> pd.DataFrame:
    p = pd.read_csv(io.StringIO(ISID_CSV))
    p["key"] = p["fund"].map(_key)
    return p


def promise_for(fund: str):
    p = load_promises()
    hit = p[p["key"] == _key(fund)]
    return None if hit.empty else hit.iloc[0]


def ceiling_of(pr, category):
    """Max net market exposure the ISID allows (max equity, or a stated max net)."""
    if pr is not None:
        m = re.search(r"net\)? equity (\d+)-(\d+)%", str(pr["net_exposure_stated"]), re.I)
        if m:
            return int(m.group(2)) / 100
        if pd.notna(pr["equity_max"]):
            return float(pr["equity_max"]) / 100
    return 0.75 if category == "Hybrid Long-Short Fund" else 1.0


def own_benchmark(pr, start, category=""):
    """The fund's ISID benchmark rebuilt from real series (see COMPONENTS), rebalanced daily to its weights.
    Returns (benchmark, label, equity component name, equity series, sources)."""
    text = str(pr["benchmark"]) if pr is not None else ""
    if not text or "not stated" in text.lower():
        text = "NIFTY 50 Hybrid Composite Debt 50:50" if "Hybrid" in category else "Nifty 500"
    comps = parse_benchmark(text)
    levels, sources = {}, []
    for name, w in comps:
        ser, src = component_series(name, start)
        sources.append((name, w, src))
        if not ser.empty:
            levels[name] = ser
    eq_name = next((n for n, _ in comps if n not in NON_EQUITY), "Nifty 500")
    eq = levels.get(eq_name, _empty())
    label = (f"{comps[0][0]} TRI" if len(comps) == 1 else
             " + ".join(f"{w:.0%} {n if n != 'Debt' else 'debt'}" for n, w in comps))
    if eq.empty:
        return _empty(), "unavailable", eq_name, eq, sources
    # align every leg to equity trading days, carrying the last NAV over gaps
    df = pd.DataFrame(levels).sort_index().ffill().reindex(eq.index).dropna()
    w = pd.Series({n: 0.0 for n in levels})
    for n, wt in comps:
        if n in levels:
            w[n] += wt
    if w.sum() < 0.999:
        label += " (partial: some legs unavailable)"
    w = w / w.sum()
    r = (df.pct_change().fillna(0) * w).sum(axis=1)
    return (1 + r).cumprod(), label, eq_name, eq, sources


# ───────────────────────── metrics ─────────────────────────
def point_return(s, days):
    past = s[s.index <= s.index[-1] - timedelta(days=days)]
    return s.iloc[-1] / past.iloc[-1] - 1 if len(past) else np.nan


def _ratios(r, rf_d):
    """Sharpe, Sortino, vol from daily returns (annualised)."""
    if len(r) < MIN_OBS or r.std() == 0:
        return np.nan, np.nan, np.nan
    ex = r - rf_d
    downside = np.sqrt((np.minimum(ex, 0) ** 2).mean())
    return (ex.mean() / r.std() * np.sqrt(252),
            ex.mean() / downside * np.sqrt(252) if downside > 0 else np.nan,
            r.std() * np.sqrt(252))


def fund_metrics(s, bm, eq, rf):
    """All metrics for one fund vs its own benchmark. s = NAV series."""
    rf_d = rf / 252
    out = {"Since launch": s.iloc[-1] / ISSUE_PRICE - 1, "Days": 0}
    for k, d in LOOKBACKS.items():
        out[k] = point_return(s, d)
    r = s.pct_change().dropna()
    out["Sharpe"], out["Sortino"], out["Volatility"] = _ratios(r, rf_d)
    out["Max DD"] = (s / s.cummax() - 1).min() if len(s) > 1 else np.nan
    days = (s.index[-1] - s.index[0]).days
    out["CAGR"] = (s.iloc[-1] / s.iloc[0]) ** (365 / days) - 1 if days >= 365 else np.nan
    keys = ["Benchmark return", "vs benchmark", "Beta", "Alpha", "Up capture", "Down capture",
            "Tracking error", "Info ratio", "Down-day wins", "Worst 10: fund", "Worst 10: bmk",
            "Bmk Sharpe", "Bmk Sortino", "Bmk Max DD", "Equity beta"]
    out.update({k: np.nan for k in keys})
    if bm.empty:
        return out
    b = bm[bm.index >= s.index[0]]
    if len(b) > 1:
        b_on = b.reindex(s.index, method="ffill").dropna()
        if len(b_on) > 1:
            out["Benchmark return"] = b_on.iloc[-1] / b_on.iloc[0] - 1
            out["vs benchmark"] = (s.iloc[-1] / s.loc[b_on.index[0]] - 1) - out["Benchmark return"]
    j = pd.concat([r, bm.pct_change()], axis=1, join="inner").dropna()
    j.columns = ["f", "b"]
    out["Days"] = len(j)
    if len(j) < MIN_OBS:
        return out
    beta = j.cov().iloc[0, 1] / j.b.var()
    up, dn = j[j.b > 0], j[j.b < 0]
    diff = j.f - j.b
    te = diff.std() * np.sqrt(252)
    worst = j.nsmallest(min(10, len(dn)), "b")
    bsh, bso, _ = _ratios(j.b, rf_d)
    bcum = (1 + j.b).cumprod()
    out.update({
        "Beta": beta,
        "Alpha": ((j.f.mean() - rf_d) - beta * (j.b.mean() - rf_d)) * 252,
        "Up capture": up.f.mean() / up.b.mean() if len(up) else np.nan,
        "Down capture": dn.f.mean() / dn.b.mean() if len(dn) else np.nan,
        "Tracking error": te, "Info ratio": diff.mean() * 252 / te if te > 0 else np.nan,
        "Down-day wins": (dn.f > dn.b).mean() if len(dn) else np.nan,
        "Worst 10: fund": worst.f.mean(), "Worst 10: bmk": worst.b.mean(),
        "Bmk Sharpe": bsh, "Bmk Sortino": bso, "Bmk Max DD": (bcum / bcum.cummax() - 1).min(),
    })
    if not eq.empty:
        je = pd.concat([r, eq.pct_change()], axis=1, join="inner").dropna()
        if len(je) >= MIN_OBS:
            out["Equity beta"] = je.cov().iloc[0, 1] / je.iloc[:, 1].var()
    return out


VERDICT_EXPLAIN = {
    "⏳ Too early": "Too early to judge: under {n} trading days of history",
    "⚠️ Over ISID limit": "More market exposure than its ISID allows",
    "🔴 No visible hedge": "Moves with its benchmark: no visible hedge",
    "🟢 Real hedge": "Hedging works: falls less than its benchmark and keeps the upside",
    "🟡 Lower exposure only": "Falls less, but gives up just as much upside: lower exposure, not a hedge",
    "🟡 Light hedge": "Light hedging: some protection, not much",
}


def verdict(m, ceiling):
    """Hedge verdict from behaviour vs own benchmark + ISID ceiling on net equity exposure."""
    if m["Days"] < HEDGE_MIN_OBS or pd.isna(m["Down capture"]):
        return "⏳ Too early"
    if ceiling < 0.99 and pd.notna(m["Equity beta"]) and m["Equity beta"] > ceiling + 0.10:
        return "⚠️ Over ISID limit"
    if m["Beta"] >= 0.85 and m["Down capture"] >= 0.90:
        return "🔴 No visible hedge"
    if m["Down capture"] < 0.80 and (m["Up capture"] - m["Down capture"]) >= 0.10:
        return "🟢 Real hedge"
    if m["Down capture"] < 0.80:
        return "🟡 Lower exposure only"
    return "🟡 Light hedge"


@st.cache_data(show_spinner="Calculating metrics…")
def build_table(nav, funds, rf):
    rows, series = [], {}
    start = nav["date"].min()
    for _, m in funds.iterrows():
        s = nav[nav.code == m.code].set_index("date")["nav"]
        if s.empty:
            continue
        pr = promise_for(m.fund)
        bm, bm_label, idx_name, eq, srcs = own_benchmark(pr, start, m.category)
        met = fund_metrics(s, bm, eq, rf)
        ceil_ = ceiling_of(pr, m.category)
        rows.append({"Fund": m.fund, "AMC": m.amc, "Category": m.category, "code": m.code,
                     "Launched": s.index[0].date(), "NAV": s.iloc[-1], **met,
                     "Benchmark": bm_label, "Equity index": idx_name, "Ceiling": ceil_,
                     "Verdict": verdict(met, ceil_)})
        series[m.code] = (bm, eq, srcs)
    return pd.DataFrame(rows), series


# metric -> (higher is better?, format, label shown in peer bars)
PEER_METRICS = {
    "Since launch": (True, "+.1%", "Return since launch"),
    "vs benchmark": (True, "+.1%", "Return vs own benchmark"),
    "Sharpe": (True, ".2f", "Sharpe ratio"),
    "Sortino": (True, ".2f", "Sortino ratio"),
    "Alpha": (True, "+.1%", "Alpha (annualised)"),
    "Max DD": (True, ".1%", "Max drawdown"),          # less negative = higher = better
    "Down capture": (False, ".2f", "Downside capture"),
    "Up capture": (True, ".2f", "Upside capture"),
}


def peer_rank(table, fund_row, col):
    """(rank, n, percentile 0-1 where 1 = best) within the table."""
    hib = PEER_METRICS[col][0]
    vals = table[col].dropna()
    v = fund_row[col]
    if pd.isna(v) or len(vals) < 2:
        return None
    better = (vals > v).sum() if hib else (vals < v).sum()
    rank = int(better) + 1
    return rank, len(vals), 1 - (rank - 1) / (len(vals) - 1)


# ───────────────────────── visual components ─────────────────────────
POS, NEG, NEU = "#2EA043", "#E5534B", "#C9A227"
CSS = """<style>
.block-container {padding-top: 1.2rem;}
.v-grid {display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:6px 0 14px;}
.v-tile {border:1px solid rgba(128,128,128,.22); border-radius:12px; padding:12px 14px;
         background:rgba(128,128,128,.04);}
.v-lab {font-size:.72rem; letter-spacing:.06em; text-transform:uppercase; opacity:.62; margin-bottom:4px;}
.v-val {font-size:1.55rem; font-weight:650; line-height:1.15; font-variant-numeric:tabular-nums;}
.v-sub {font-size:.78rem; opacity:.75; margin-top:4px;}
.v-pill {display:inline-block; padding:1px 8px; border-radius:999px; font-size:.74rem; font-weight:600;}
.v-pos {color:#2EA043; background:rgba(46,160,67,.12);} .v-neg {color:#E5534B; background:rgba(229,83,75,.12);}
.v-neu {color:#B8901C; background:rgba(201,162,39,.14);} .v-mut {opacity:.7; background:rgba(128,128,128,.14);}
.v-card {border:1px solid rgba(128,128,128,.22); border-radius:14px; padding:16px 18px; margin:8px 0 14px;}
.v-h {font-size:1.02rem; font-weight:650; margin-bottom:2px;}
.v-meta {font-size:.78rem; opacity:.65; margin-bottom:12px;}
.v-sec {font-size:.72rem; letter-spacing:.06em; text-transform:uppercase; opacity:.62; margin:12px 0 8px;}
.v-row {display:flex; align-items:center; gap:10px; margin:7px 0; font-size:.86rem;}
.v-row .n {width:150px; flex:none;} .v-row .t {flex:1; position:relative; height:9px;
           background:rgba(128,128,128,.16); border-radius:6px;}
.v-row .f {position:absolute; top:0; bottom:0; border-radius:6px;}
.v-row .r {width:110px; flex:none; text-align:right; font-variant-numeric:tabular-nums; opacity:.85;}
.v-mk {position:absolute; top:-4px; width:3px; height:17px; border-radius:2px; background:currentColor;}
.v-chk {display:flex; gap:10px; padding:8px 0; border-bottom:1px solid rgba(128,128,128,.13); font-size:.88rem;}
.v-chk:last-child {border-bottom:none;} .v-chk .i {font-size:1rem; width:20px; flex:none;}
.v-chk .d {opacity:.7; font-size:.8rem;}
.v-chip {display:inline-block; margin:3px 5px 3px 0; padding:3px 10px; border-radius:999px; font-size:.78rem;
         border:1px solid rgba(128,128,128,.28);}
.v-chip.on {border-color:rgba(201,162,39,.7); background:rgba(201,162,39,.12); font-weight:600;}
.v-verdict {display:inline-block; padding:4px 12px; border-radius:999px; font-size:.85rem; font-weight:600;
            margin:2px 0 6px;}
.p-head {display:flex; align-items:center; justify-content:space-between; gap:8px;}
.p-g {font-size:.7rem; letter-spacing:.07em; text-transform:uppercase; opacity:.55; margin:14px 0 2px;}
.p-row {display:grid; grid-template-columns:112px 1fr 58px 70px; align-items:center; gap:10px; height:30px;}
.p-n {font-size:.84rem;} .p-v {font-size:.84rem; text-align:right; font-variant-numeric:tabular-nums;}
.p-r {text-align:right;}
.p-t {position:relative; height:22px;}
.p-t::before {content:""; position:absolute; left:0; right:0; top:10px; height:2px; border-radius:2px;
              background:linear-gradient(90deg, rgba(229,83,75,.35), rgba(128,128,128,.18) 50%, rgba(46,160,67,.35));}
.p-dot {position:absolute; top:8px; width:6px; height:6px; margin-left:-3px; border-radius:50%;
        background:rgba(128,128,128,.38);}
.p-med {position:absolute; top:3px; width:2px; height:16px; margin-left:-1px; background:rgba(128,128,128,.7);}
.p-me {position:absolute; top:3px; width:16px; height:16px; margin-left:-8px; border-radius:50%;
       border:2.5px solid rgba(255,255,255,.95); box-shadow:0 0 0 1px rgba(0,0,0,.25), 0 1px 4px rgba(0,0,0,.25);}
</style>"""


def esc(x):
    return html.escape(str(x))


def fmt(v, spec):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, spec)


def pill(text, kind):
    return f'<span class="v-pill v-{kind}">{esc(text)}</span>'


def tile(label, value, sub=""):
    return (f'<div class="v-tile"><div class="v-lab">{esc(label)}</div><div class="v-val">{value}</div>'
            f'<div class="v-sub">{sub}</div></div>')


def tiles(items):
    st.markdown('<div class="v-grid">' + "".join(items) + "</div>", unsafe_allow_html=True)


def compare_pill(v, b, spec, higher_better=True):
    """Pill comparing fund value v with benchmark value b."""
    if pd.isna(v) or pd.isna(b):
        return pill("benchmark n/a", "mut")
    good = v > b if higher_better else v < b
    return pill(f"bmk {format(b, spec)}", "pos" if good else "neg")


def verdict_pill(v):
    kind = {"🟢": "pos", "🔴": "neg", "⚠️": "neg", "🟡": "neu"}.get(v.split(" ")[0], "mut")
    text = v.split(" ")[0] + " " + VERDICT_EXPLAIN.get(v, v).format(n=HEDGE_MIN_OBS)
    return f'<span class="v-verdict v-{kind}">{esc(text)}</span>'


PEER_GROUPS = {
    "Returns": ["Since launch", "vs benchmark", "Alpha"],
    "Risk-adjusted": ["Sharpe", "Sortino"],
    "Risk": ["Max DD", "Down capture", "Up capture"],
}
SHORT_LABEL = {"Since launch": "Since launch", "vs benchmark": "vs own benchmark", "Alpha": "Alpha (ann.)",
               "Sharpe": "Sharpe", "Sortino": "Sortino", "Max DD": "Max drawdown",
               "Down capture": "Downside capture", "Up capture": "Upside capture"}


def _ordinal(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def peer_bars(table, row, title):
    """Peer-distribution strip: every fund in the scope as a dot (better to the right), this fund
    highlighted, the median as a tick; grouped, with a plain-language summary on top."""
    group_pct, lines = {}, []
    for group, cols in PEER_GROUPS.items():
        rows_html, pcts = [], []
        for col in cols:
            hib, spec, _ = PEER_METRICS[col]
            pr = peer_rank(table, row, col)
            vals = table[col].astype(float).dropna()
            if pr is None or len(vals) < 2:
                continue
            rank, n, pct = pr
            pcts.append(pct)
            # position: value-based, better to the right, outliers clipped so one fund can't squash the rest
            lo, hi = vals.quantile(0.05), vals.quantile(0.95)
            if hi - lo < 1e-12:
                lo, hi = vals.min() - 1e-9, vals.max() + 1e-9
            pos = lambda v: float(np.clip((v - lo) / (hi - lo), 0, 1)) * (1 if hib else -1) + (0 if hib else 1)
            peers = "".join(f'<span class="p-dot" style="left:{pos(v) * 100:.1f}%"></span>' for v in vals)
            med = f'<span class="p-med" style="left:{pos(vals.median()) * 100:.1f}%"></span>'
            color = POS if pct >= 0.67 else (NEG if pct < 0.34 else NEU)
            me = (f'<span class="p-me" style="left:{pos(row[col]) * 100:.1f}%;background:{color}"></span>')
            badge_kind = "pos" if pct >= 0.67 else ("neg" if pct < 0.34 else "neu")
            rows_html.append(
                f'<div class="p-row"><div class="p-n">{esc(SHORT_LABEL[col])}</div>'
                f'<div class="p-t">{peers}{med}{me}</div>'
                f'<div class="p-v">{fmt(row[col], spec)}</div>'
                f'<div class="p-r"><span class="v-pill v-{badge_kind}">{_ordinal(rank)} of {n}</span></div></div>')
        if rows_html:
            group_pct[group] = float(np.mean(pcts))
            lines.append(f'<div class="p-g">{esc(group)}</div>' + "".join(rows_html))
    if not lines:
        st.markdown(f'<div class="v-card"><div class="v-h">{esc(title)}</div>'
                    '<div class="v-meta">Needs at least two funds with enough history.</div></div>',
                    unsafe_allow_html=True)
        return
    overall = float(np.mean(list(group_pct.values())))
    phrase = {"Returns": "returns", "Risk-adjusted": "risk-adjusted returns", "Risk": "risk control"}
    strong = [phrase[g] for g, v in group_pct.items() if v >= 0.67]
    weak = [phrase[g] for g, v in group_pct.items() if v < 0.34]
    listing = lambda xs: xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]
    summary = []
    if strong:
        summary.append("strong on " + listing(strong))
    if weak:
        summary.append("weak on " + listing(weak))
    top = max(1, round((1 - overall) * 100))
    head_kind = "pos" if overall >= 0.67 else ("neg" if overall < 0.34 else "neu")
    st.markdown(
        f'<div class="v-card"><div class="p-head"><div class="v-h">{esc(title)}</div>'
        f'<span class="v-pill v-{head_kind}">Top {top}% overall</span></div>'
        f'<div class="v-meta">{esc("; ".join(summary).capitalize() or "Around the middle of the pack")}. '
        f'Each dot is a fund, <b>better to the right</b>; the big dot is this fund, the tick is the median.</div>'
        + "".join(lines) + "</div>", unsafe_allow_html=True)


def peer_summary(table, row):
    """(overall percentile 0-1, plain-language summary) from the three metric groups."""
    group_pct = {}
    for group, cols in PEER_GROUPS.items():
        pcts = [pr[2] for pr in (peer_rank(table, row, c) for c in cols) if pr is not None]
        if pcts:
            group_pct[group] = float(np.mean(pcts))
    if not group_pct:
        return None, ""
    phrase = {"Returns": "returns", "Risk-adjusted": "risk-adjusted returns", "Risk": "risk control"}
    listing = lambda xs: xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]
    strong = [phrase[g] for g, v in group_pct.items() if v >= 0.67]
    weak = [phrase[g] for g, v in group_pct.items() if v < 0.34]
    parts = (["strong on " + listing(strong)] if strong else []) + (["weak on " + listing(weak)] if weak else [])
    return float(np.mean(list(group_pct.values()))), ("; ".join(parts).capitalize() or "Around the middle of the pack")


def peer_chart(table, row, title):
    """Option B: every metric on a common 'worst → best' percentile scale with coloured zones."""
    overall, summary = peer_summary(table, row)
    if overall is None:
        st.markdown(f'<div class="v-card"><div class="v-h">{esc(title)}</div>'
                    '<div class="v-meta">Needs at least two funds with enough history.</div></div>',
                    unsafe_allow_html=True)
        return
    kind = "pos" if overall >= 0.67 else ("neg" if overall < 0.34 else "neu")
    rows = []  # (y label, metric col)
    for group, cols in PEER_GROUPS.items():
        for c in cols:
            if peer_rank(table, row, c) is not None:
                rows.append((group, c))
    labels = [f"<b>{SHORT_LABEL[c]}</b>" for _, c in rows]
    ys = list(range(len(rows)))[::-1]  # first metric at the top
    fig = go.Figure()
    for x0, x1, col in [(0, 33.3, "229,83,75"), (33.3, 66.7, "201,162,39"), (66.7, 100, "46,160,67")]:
        fig.add_vrect(x0=x0, x1=x1, fillcolor=f"rgba({col},0.10)", line_width=0, layer="below")
    for x, t in [(16.7, "Bottom third"), (50, "Middle"), (83.3, "Top third")]:
        fig.add_annotation(x=x, y=len(rows) - 0.35, text=t, showarrow=False, font=dict(size=10, color=GREY))
    # median line stops below the zone labels so it doesn't run through "Middle"
    fig.add_shape(type="line", x0=50, x1=50, y0=-0.6, y1=len(rows) - 0.65,
                  line=dict(color=GREY, width=1, dash="dot"), layer="below")
    rng = np.random.default_rng(0)
    prev_group = None
    for (group, c), y in zip(rows, ys):
        hib, spec, _ = PEER_METRICS[c]
        sub = table[["Fund", c]].dropna()
        n = len(sub)
        rk = sub[c].rank(ascending=hib, method="average")
        pct = (rk - 1) / (n - 1) * 100
        mine = sub["Fund"] == row["Fund"]
        jitter = rng.uniform(-0.18, 0.18, n)
        fig.add_scatter(x=pct[~mine], y=y + jitter[~mine.values], mode="markers",
                        marker=dict(size=7, color="rgba(128,128,128,0.45)"), showlegend=False,
                        customdata=np.stack([sub["Fund"][~mine], sub[c][~mine].map(lambda v: format(v, spec))], axis=1),
                        hovertemplate="%{customdata[0]}<br>%{customdata[1]}<extra></extra>")
        rank, total, p = peer_rank(table, row, c)
        colr = POS if p >= 0.67 else (NEG if p < 0.34 else NEU)
        fig.add_scatter(x=[p * 100], y=[y], mode="markers", showlegend=False,
                        marker=dict(size=17, color=colr, line=dict(color="white", width=2.5)),
                        hovertemplate=f"{esc(row['Fund'])}<br>{format(row[c], spec)} · {_ordinal(rank)} of {total}<extra></extra>")
        fig.add_annotation(xref="paper", x=1.02, y=y, xanchor="left", showarrow=False, align="left",
                           text=f"<b>{format(row[c], spec)}</b>", font=dict(size=12))
        fig.add_annotation(xref="paper", x=1.02, y=y - 0.32, xanchor="left", showarrow=False, align="left",
                           text=f"{_ordinal(rank)} of {total}", font=dict(size=10, color=colr))
        if prev_group is not None and group != prev_group:
            fig.add_hline(y=y + 0.5, line=dict(color="rgba(128,128,128,0.35)", width=1))
        prev_group = group
    group_ticks = []
    for g in PEER_GROUPS:
        gy = [y for (gg, _), y in zip(rows, ys) if gg == g]
        if gy:
            fig.add_annotation(x=-1, y=max(gy) + 0.42, xanchor="right", showarrow=False,
                               text=g.upper(), font=dict(size=9, color=GREY))
    fig.update_layout(
        height=70 + 44 * len(rows), margin=dict(l=10, r=78, t=10, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hoverlabel=dict(font_size=12))
    fig.update_xaxes(range=[-2, 102], tickvals=[0, 50, 100], ticktext=["← Worse", "Median", "Better →"],
                     showgrid=False, zeroline=False, fixedrange=True)
    fig.update_yaxes(tickvals=ys, ticktext=labels, range=[-0.6, len(rows) - 0.1], showgrid=False,
                     zeroline=False, fixedrange=True)
    st.markdown(f'<div class="p-head"><div class="v-h">{esc(title)}</div>'
                f'<span class="v-pill v-{kind}">Top {max(1, round((1 - overall) * 100))}% overall</span></div>'
                f'<div class="v-meta">{esc(summary)}. Big dot = this fund; grey dots = peers (hover for names). '
                f'Every metric is on the same scale: further right is better.</div>', unsafe_allow_html=True)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def range_bar(name, lo, hi, scale=100, marker=None, note=""):
    """Allowed range shaded; optional marker for observed value. Values in % of scale."""
    if pd.isna(lo) or pd.isna(hi):
        return (f'<div class="v-row"><div class="n">{esc(name)}</div><div class="t"></div>'
                f'<div class="r">not stated</div></div>')
    l, w = lo / scale * 100, max((hi - lo) / scale * 100, 1)
    mk = ""
    if marker is not None and pd.notna(marker):
        inside = lo - 10 <= marker <= hi + 10
        mk = (f'<div class="v-mk" style="left:calc({min(max(marker / scale * 100, 0), 100):.1f}% - 1px);'
              f'color:{POS if inside else NEG}"></div>')
    return (f'<div class="v-row"><div class="n">{esc(name)}</div><div class="t">'
            f'<div class="f" style="left:{l:.1f}%;width:{w:.1f}%;background:{NEU};opacity:.45"></div>{mk}</div>'
            f'<div class="r">{lo:.0f}–{hi:.0f}%{note}</div></div>')


def check(icon, title, detail):
    return (f'<div class="v-chk"><div class="i">{icon}</div><div><div>{esc(title)}</div>'
            f'<div class="d">{detail}</div></div></div>')


TOOL_TAGS = {"covered_calls": "Covered calls", "protective_puts": "Protective puts", "index_hedging": "Index hedging",
             "pair_trades": "Pair trades", "arbitrage": "Arbitrage"}


def promise_panel(pr, m):
    """Promise vs reality: what the ISID allows vs what the NAV shows."""
    if pr is None:
        st.markdown('<div class="v-card"><div class="v-h">Promise vs reality</div>'
                    '<div class="v-meta">No ISID on file for this fund yet.</div></div>', unsafe_allow_html=True)
        return
    eqb = m["Equity beta"]
    ceil_ = m["Ceiling"]
    # left: what it may hold
    left = ['<div class="v-sec">What it may hold (ISID)</div>',
            range_bar("Equity (incl. hedged)", pr["equity_min"], pr["equity_max"]),
            range_bar("Debt", pr["debt_min"], pr["debt_max"]),
            range_bar("Unhedged shorts", 0, pr["unhedged_short_max"]),
            range_bar("Net exposure (est.)", 0, ceil_ * 100, marker=None if pd.isna(eqb) else eqb * 100,
                      note=f" · now {eqb:.0%}" if pd.notna(eqb) else "")]
    stated = str(pr["net_exposure_stated"])
    if not stated.lower().startswith("not stated"):
        left.append(f'<div class="v-meta" style="margin-top:8px">Stated in ISID: {esc(stated)}</div>')
    elif "(" in stated:
        left.append(f'<div class="v-meta" style="margin-top:8px">{esc(stated[stated.find("(") + 1:stated.rfind(")")])}</div>')
    # right: checks
    chk = ['<div class="v-sec">Is it doing what it said?</div>']
    if pd.isna(eqb):
        chk.append(check("⏳", "Net exposure within its ISID limit", "Needs more NAV history."))
    else:
        ok = ceil_ >= 0.99 or eqb <= ceil_ + 0.10
        chk.append(check("✅" if ok else "⚠️", "Net exposure within its ISID limit",
                         f"Estimated {eqb:.0%} vs max {ceil_:.0%} (beta to {esc(m['Equity index'])})"))
    if pd.notna(m["vs benchmark"]):
        chk.append(check("✅" if m["vs benchmark"] > 0 else "❌", "Beating its own benchmark",
                         f"{m['Since launch']:+.1%} vs {m['Benchmark return']:+.1%} for {esc(m['Benchmark'])}"))
    if pd.notna(m["Down capture"]):
        dc, uc = m["Down capture"], m["Up capture"]
        chk.append(check("✅" if dc < 0.9 else "❌", "Protects on its benchmark's down days",
                         f"Falls {dc:.2f}× as much as its benchmark"))
        chk.append(check("✅" if uc - dc >= 0.10 else "➖", "Keeps more upside than downside",
                         f"Up {uc:.2f}× vs down {dc:.2f}×"))
    if pd.notna(m["Sharpe"]) and pd.notna(m["Bmk Sharpe"]):
        chk.append(check("✅" if m["Sharpe"] > m["Bmk Sharpe"] else "❌", "Better risk-adjusted than its benchmark",
                         f"Sharpe {m['Sharpe']:.2f} vs {m['Bmk Sharpe']:.2f}"))
    chk.append(check("⏳", "Uses the derivative tools it names", "Needs the monthly portfolio files."))
    # tools
    named = [v for k, v in TOOL_TAGS.items() if str(pr[k]).upper() == "Y"]
    chips = "".join(f'<span class="v-chip on">{esc(t)}</span>' for t in named) or '<span class="v-meta">–</span>'
    ow = str(pr["options_writing"])
    ow_txt = {"Y": "Yes", "N": "No"}.get(ow, ow)
    head = (f'<div class="v-h">Promise vs reality</div><div class="v-meta">ISID {esc(pr["isid_date"])} · '
            f'Benchmark {esc(pr["benchmark"])} · Managed by {esc(pr["fund_managers"])} · '
            f'<a href="{esc(pr["source_url"])}" target="_blank">ISID PDF ↗</a></div>')
    box = st.container(border=True)
    box.markdown(head, unsafe_allow_html=True)
    a, b = box.columns([1, 1], gap="large")
    left += [f'<div class="v-sec">Tools it says it may use</div>{chips}'
             f'<span class="v-chip">Option writing: {esc(ow_txt)}</span>']
    a.markdown("".join(left), unsafe_allow_html=True)
    b.markdown("".join(chk), unsafe_allow_html=True)
    with a.expander("Full strategy list from the ISID"):
        items = [x.strip() for x in str(pr["derivative_strategies"]).split(";") if x.strip()]
        st.markdown("".join(f'<span class="v-chip">{esc(x)}</span>' for x in items), unsafe_allow_html=True)
        st.caption("Most ISIDs describe this list as illustrative: it shows what is allowed, not what is used.")


def style_table(df, good_high=(), good_low=(), signed=()):
    """Theme-neutral colouring: green/red tint by rank within the column."""
    def shade(col, high):
        v = col.astype(float)
        if v.notna().sum() < 2:
            return [""] * len(v)
        pct = v.rank(pct=True)
        out = []
        for p in pct:
            if pd.isna(p):
                out.append("")
                continue
            p = p if high else 1 - p
            a = abs(p - 0.5) * 0.5
            out.append(f"background-color: rgba({'46,160,67' if p > 0.5 else '229,83,75'},{a:.2f})")
        return out
    sty = df.style
    for c in good_high:
        if c in df:
            sty = sty.apply(lambda col: shade(col, True), subset=[c])
    for c in good_low:
        if c in df:
            sty = sty.apply(lambda col: shade(col, False), subset=[c])
    for c in signed:
        if c in df:
            sty = sty.map(lambda v: "" if pd.isna(v) else f"color: {POS if v > 0 else NEG}", subset=[c])
    return sty


# ───────────────────────── ISID promise table ─────────────────────────
# Collected from AMFI-hosted ISIDs (portal.amfiindia.com/spages/S-<n>.pdf), Nov 2026.
# Platinum (S-26*) from Mirae's launch press release because its ISID would not load.
ISID_CSV = """isid,fund,amc_brand,mutual_fund,category,isid_date,benchmark,equity_min,equity_max,debt_min,debt_max,unhedged_short_max,hedged_max,long_deriv_max,reits_invits_max,net_exposure_stated,derivative_strategies,covered_calls,protective_puts,index_hedging,pair_trades,arbitrage,options_writing,fund_managers,source_url
S-1,qsif Equity Long Short Fund,qsif SIF,quant Mutual Fund,Equity Long-Short Fund,2025-08-06,NIFTY 500 TRI,80,100,0,20,25,100,50,20,Not stated,"Short futures; synthetic short; long put; bear put/call spreads; short call; put butterfly/calendar/diagonal; ratio put spread; hedged/unhedged long-only and long-short with up to 25% naked short",N,N,Y,N,N,Y,"Sandeep Tandon; Lokesh Garg; Ankit Pande; Sameer Kate; Sanjeev Sharma",https://portal.amfiindia.com/spages/S-1.pdf
S-2,qsif Hybrid Long-Short Fund,qsif SIF,quant Mutual Fund,Hybrid Long-Short Fund,2025-09-10,NIFTY 50 Hybrid Composite Debt 50:50,25,75,25,75,25,100,50,20,Not stated,"Same tentative short list as qsif Equity L/S; partial/full hedging with index/stock futures or options; long-short with up to 25% naked short",N,N,Y,N,N,Y,"Sandeep Tandon; Lokesh Garg; Ankit Pande; Sameer Kate; Sanjeev Sharma",https://portal.amfiindia.com/spages/S-2.pdf
S-3,Altiva Hybrid Long-Short Fund,Altiva SIF,Edelweiss Mutual Fund,Hybrid Long-Short Fund,2025-09-18,NIFTY 50 Hybrid Composite Debt 50:50 TRI,25,75,25,75,25,100,,20,Not stated (net short monitored),"Cash-futures arbitrage; covered calls; protective puts; index/stock futures-options for hedging and beta adjustment; long/short pair trades; straddles/strangles; directional futures/options; IRS/IRF/FRA",Y,Y,Y,Y,Y,Covered calls only (short straddles conditional),"Bhavesh Jain; Bharat Lahoti; Dhawal Dalal; Pranavi Kulkarni; Amit Vora",https://portal.amfiindia.com/spages/S-3.pdf
S-4,Magnum Hybrid Long Short Fund,Magnum SIF,SBI Mutual Fund,Hybrid Long-Short Fund,2025-09-23,NIFTY 50 Hybrid Composite Debt 50:50 TRI,65,75,25,35,25,75,,10,Not stated,"Covered calls; cash-futures, dividend, merger, buyback and calendar arbitrage; index/stock futures hedging; protective puts; selling calls/call spreads; inverse covered call; short futures; long puts/put spreads",Y,Y,Y,N,Y,Y,Gaurav Mehta,https://portal.amfiindia.com/spages/S-4.pdf
S-5,Diviniti Equity Long Short Fund,Diviniti SIF,ITI Mutual Fund,Equity Long-Short Fund,2025-10-23,Nifty 50 TRI,80,100,0,20,25,100,100,20,Not stated,"Cash-futures arbitrage; stock futures; index/stock futures-options hedging; alpha trades (stock vs sector index futures); covered calls; short index/stock futures; puts and put spreads; short calls; bear call spreads; ratio/calendar/butterfly/condor spreads; OIS/FRA/IRF",Y,Y,Y,Y,Y,Y,Vasav Sahgal; Rajesh Bhatia (overseas),https://portal.amfiindia.com/spages/S-5.pdf
S-6,qsif Equity Ex-Top 100 Long-Short Fund,qsif SIF,quant Mutual Fund,Equity Ex-Top 100 Long-Short Fund,2025-08-14,NIFTY 500 TRI,65,100,0,35,25,100,50,20,Not stated (65-100% outside top 100; 0-35% top 100),"Same tentative short list as qsif Equity L/S; partial/full hedging with index/stock futures or options",N,N,Y,N,N,Y,"Sandeep Tandon; Lokesh Garg; Ankit Pande; Sameer Kate; Sanjeev Sharma",https://portal.amfiindia.com/spages/S-6.pdf
S-7,Titanium Hybrid Long-Short Fund,Titanium SIF,Tata Mutual Fund,Hybrid Long-Short Fund,2026-09-30,CRISIL Hybrid 50+50 Moderate TRI,65,75,25,35,25,75,,10,Not stated (gross and net managed dynamically),"Short futures/options; index-futures hedging; covered calls; protective puts; call and put writing; pair trades; cash-futures arbitrage; collars, spreads, straddles, strangles, butterflies, condors (illustrative)",Y,Y,Y,Y,Y,Y,Suraj Nanda; Akhil Mittal; Hasmukh Vishariya (overseas),https://portal.amfiindia.com/spages/S-7.pdf
S-8,iSIF Equity Ex-Top 100 Long-Short Fund,iSIF SIF,ICICI Prudential Mutual Fund,Equity Ex-Top 100 Long-Short Fund,2026-09-30,Nifty 500 TRI,65,100,0,35,25,100,,20,Not stated (65-100% ex-top-100; 0-35% other equity),"Equity arbitrage; covered calls; index put buying; index futures shorting; protective stock puts/calls; short calls; short puts; long calls/puts/futures; short futures; bull put/bear call spreads; short straddles/strangles",Y,Y,Y,N,Y,Y,Sankaran Naren; Manan Tijoriwala; Divya Jain,https://portal.amfiindia.com/spages/S-8.pdf
S-9,iSIF Hybrid Long-Short Fund,iSIF SIF,ICICI Prudential Mutual Fund,Hybrid Long-Short Fund,2026-09-30,CRISIL Hybrid 50+50 Moderate,65,75,25,35,25,100,,10,Not stated,"Equity arbitrage; covered calls; index futures and index put hedging; protective stock puts/calls; short calls/puts; short straddles/strangles; long calls/puts/futures; short futures; bull put/bear call spreads",Y,Y,Y,N,Y,Y,Rajat Chandak; Ayush Shah; Manish Banthia; Akhil Kakkar,https://portal.amfiindia.com/spages/S-9.pdf
S-10,Arudha Hybrid Long-Short Fund,Arudha SIF,Bandhan Mutual Fund,Hybrid Long-Short Fund,2026-01-02,CRISIL Hybrid 85+15 Conservative TRI,35,65,35,65,25,100,50,20,Not stated,"Covered call; protective put; collar; cash-secured put; long calls/puts; bull/bear spreads; calendar; straddle; strangle; butterfly; iron condor/butterfly; synthetic long/short; pair trades; stock-level hedging; IRS/FRA/IRF/CDS",Y,Y,N,Y,N,Y,Kapil Kankonkar; Brijesh Shah; Debraj Lahiri; Nilesh Saha,https://portal.amfiindia.com/spages/S-10.pdf
S-11,DynaSIF Equity Long-Short Fund,DynaSIF SIF,360 ONE Mutual Fund,Equity Long-Short Fund,2026-01-21,BSE 500 TRI,80,100,0,20,25,,25,20,Not stated (sector long-short net ~20% average),"Long-short equity; sector long-short; pairs trading; merger arbitrage; index rebalancing trades; directional index derivatives; index pair trading; covered calls; long/short straddles-strangles; stock options/futures arbitrage; index-derivative hedging",Y,N,Y,Y,Y,Y,Harsh Aggarwal,https://portal.amfiindia.com/spages/S-11.pdf
S-12,Arudha Equity Long-Short Fund,Arudha SIF,Bandhan Mutual Fund,Equity Long-Short Fund,2026-02-06,NIFTY 500 TRI,80,100,0,20,25,100,50,20,Not stated,"Covered call; protective put; collar; cash-secured put; long calls/puts; bull/bear spreads; calendar; straddle; strangle; butterfly; iron condor/butterfly; synthetic long/short; pair trades; stock-level hedging",Y,Y,N,Y,N,Y,Nilesh Saha; Brijesh Shah,https://portal.amfiindia.com/spages/S-12.pdf
S-13,Apex Hybrid Long-Short Fund,Apex SIF,Aditya Birla Sun Life Mutual Fund,Hybrid Long-Short Fund,2026-02-27,NIFTY 50 Hybrid Composite Debt 50:50 TRI,35,65,35,65,25,,,20,Not stated,"Cash-futures arbitrage; covered calls; protective puts; long/short pair trades; index/stock futures-options for hedging or beta adjustment; long/short straddles-strangles; directional futures/options; put-call parity trades; IRS/IRF/FRA",Y,Y,Y,Y,Y,Y,Lovelish Solanki; Mohit Sharma,https://portal.amfiindia.com/spages/S-13.pdf
S-14,DynaSIF Active Asset Allocator Long-Short Fund,DynaSIF SIF,360 ONE Mutual Fund,Active Asset Allocator Long-Short Fund,2026-03-27,25% BSE Sensex TRI + 60% CRISIL Short Term Bond + 15% iCOMDEX,20,50,20,65,25,100,,20,Not stated (commodity derivatives 0-25%),"Stock and sector long-short; pairs trading; merger arbitrage; index rebalancing; directional index trades; index pair trading; covered calls; straddles/strangles; stock options/futures arbitrage; index-derivative hedging; commodity options",Y,N,Y,Y,Y,Y,Harsh Agarwal; Milan Mody (debt); Rahul Khetawat (commodity),https://portal.amfiindia.com/spages/S-14.pdf
S-15,qsif Active Asset Allocator Long-Short Fund,qsif SIF,quant Mutual Fund,Active Asset Allocator Long-Short Fund,2026-03-27,40% NSE 500 TRI + 30% CRISIL Short Term Bond + 30% iCOMDEX,0,100,0,100,25,100,50,20,Not stated (commodity derivatives 0-30%),"Same tentative short list as qsif Equity L/S; derivatives for hedging and rebalancing",N,N,N,N,N,Y,Sandeep Tandon; Jignesh Shah; Ankit Pande; Sameer Kate; Sanjeev Sharma,https://portal.amfiindia.com/spages/S-15.pdf
S-16,Sapphire Equity Long-Short SIF,Sapphire SIF,Franklin Templeton Mutual Fund,Equity Long-Short Fund,2026-03-26,Nifty 500 TRI,80,100,0,20,25,50,,,"Not stated (approach section: gross long 75-100%, short 0-25%; conflicts with 80% floor)","Unhedged shorts via listed futures/options as stock-level bearish picks; covered calls; imperfect hedging",Y,N,N,N,N,Covered calls only,Arihant Jain,https://portal.amfiindia.com/spages/S-16.pdf
S-17,Titanium Equity Long-Short Fund,Titanium SIF,Tata Mutual Fund,Equity Long-Short Fund,2026-04-09,Nifty 500 TRI,80,100,0,20,25,100,,20,Not stated (gross and net managed dynamically),"Long-short via stock and index futures-options; index futures-options hedging; pair trades; cash-futures arbitrage; sector vs index trades; covered calls; call and put writing; protective puts (illustrative)",Y,Y,Y,Y,Y,Y,Suraj Nanda; Amit Somani; Hasmukh Vishariya (overseas),https://portal.amfiindia.com/spages/S-17.pdf
S-18,WSIF Equity Ex-Top 100 Long-Short Fund,WSIF,The Wealth Company Mutual Fund,Equity Ex-Top 100 Long-Short Fund,2026-03-25,NIFTY 500 TRI,65,100,0,35,25,100,,20,Not stated (65-100% ex-top-100; 0-35% top 100),"Tentative short list (short futures; synthetic short; long puts; put/call spreads; short call; ratio put spread); hedged/unhedged long-only and long-short; covered calls; index/stock futures hedging",Y,Y,Y,N,N,Y,Chinmay Sathe,https://portal.amfiindia.com/spages/S-18.pdf
S-19,WSIF Equity Long-Short Fund,WSIF,The Wealth Company Mutual Fund,Equity Long-Short Fund,2026-03-25,NIFTY 500 TRI,80,100,0,20,25,100,,20,Not stated,"Tentative short list (short futures; synthetic short; long puts; put/call spreads; short call; ratio put spread); covered calls; index/stock futures-options hedging up to 100%",Y,Y,Y,N,N,Y,Chinmay Sathe,https://portal.amfiindia.com/spages/S-19.pdf
S-20,Arthaya Equity Long Short Fund,Arthaya SIF,Union Mutual Fund,Equity Long-Short Fund,2026-04-17,NIFTY 200 TRI,80,100,0,20,25,100,100,20,Not stated,"Hedged/unhedged long-only and long-short with up to 25% unhedged short; short futures; synthetic short; long puts; put/call spreads; short call; ratio put spread; covered calls; index futures hedging; unhedged option writing (risk factors)",Y,Y,Y,N,N,Y,Rajesh Aynor; Hiten Bhadra (arbitrage),https://portal.amfiindia.com/spages/S-20.pdf
S-21,qsif Sector Rotation Long-Short Fund,qsif SIF,quant Mutual Fund,Sector Rotation Long-Short Fund,2026-04-22,NIFTY 500 TRI,80,100,0,20,25,100,50,20,Not stated (80-100% in max 4 sectors),"Same tentative short list as qsif Equity L/S; partial/full hedging with index/stock futures or options",N,N,Y,N,N,Y,Sandeep Tandon; Jignesh Shah; Ankit Pande; Sameer Kate; Sanjeev Sharma,https://portal.amfiindia.com/spages/S-21.pdf
S-22,Altiva Equity Ex-Top 100 Long-Short Fund,Altiva SIF,Edelweiss Mutual Fund,Equity Ex-Top 100 Long-Short Fund,2026-04-20,Nifty 500 TRI,65,100,0,35,25,100,,20,Not stated (65-100% ex-top-100; 0-35% other equity),"Equity arbitrage; covered calls; pair trades; index put buying; index futures shorting; protective stock puts/calls; short calls/puts; short straddles/strangles; long/short futures and options; bear put/call spreads",Y,Y,Y,Y,Y,Y,Nikhil Gada; Trideep Bhattacharya; Amit Vora (overseas),https://portal.amfiindia.com/spages/S-22.pdf
S-23,iSIF Active Asset Allocator Long-Short Fund,iSIF SIF,ICICI Prudential Mutual Fund,Active Asset Allocator Long-Short Fund,2026-11-13,50% Nifty 500 TRI + 40% Nifty Composite Debt + 7% gold + 3% silver,35,80,10,55,25,100,,20,Not stated (commodity derivatives allowed),"Equity arbitrage; covered calls; index put and index futures hedging; protective stock puts/calls; short calls/puts; short straddles/strangles; long/short futures and options; bear put/call spreads; commodity arbitrage",Y,Y,Y,N,Y,Y,Sharmila D'silva; Masoomi Jhurmarvala; Manish Banthia; Akhil Kakkar; Gaurav Chikane (commodity),https://portal.amfiindia.com/spages/S-23.pdf
S-24,iSIF Equity Long-Short Fund,iSIF SIF,ICICI Prudential Mutual Fund,Equity Long-Short Fund,2026-05-07,Nifty 500 TRI,80,100,0,20,25,,,20,Not stated,"Equity arbitrage; covered calls; protective stock puts/calls; index futures and index put hedging; short calls/puts; short straddles/strangles; long/short futures and options; bull put/bear call spreads",Y,Y,Y,N,Y,Y,Mittul Kalawadia; Nitya Mishra; Sri Sharma,https://portal.amfiindia.com/spages/S-24.pdf
S-25,RedHex Hybrid Long-Short Fund,RedHex SIF,HSBC Mutual Fund,Hybrid Long-Short Fund,2026-05-07,NIFTY 50 Hybrid Composite Debt 50:50 TRI,25,75,25,75,25,75,,20,Not stated,"Cash-futures and reverse arbitrage; covered calls; protective puts; index futures hedging; pair trades (same-industry); short straddles/strangles; butterflies; directional futures/options; IRS/IRF/FRA; CDS",Y,Y,Y,Y,Y,Y,Shriram Ramanathan; Praveen Ayathan (arbitrage); Venugopal Manghat; Mayank Chaturvedi (overseas),https://portal.amfiindia.com/spages/S-25.pdf
S-27,Prism Hybrid Long-Short Fund,Prism SIF,JioBlackRock Mutual Fund,Hybrid Long-Short Fund,2026-05-15,NIFTY 50 Hybrid Composite Debt 50:50 TRI,35,75,25,65,25,100,100,20,Not stated,"Merger arbitrage; collars; cash-futures arbitrage; IPO and special-situation trades (open offers, buybacks); indicative list of 29 incl. covered calls, protective/cash-secured/long puts, index put and index futures hedging, pair trades, short calls/puts, short straddles/strangles, market neutral, short futures",Y,Y,Y,Y,Y,Y,Arun Ramachandran; Tanvi Kacheria; Siddharth Deb; Virendra Kumar,https://portal.amfiindia.com/spages/S-27.pdf
S-28,DynaSIF Equity Ex-Top 100 Long-Short Fund,DynaSIF SIF,360 ONE Mutual Fund,Equity Ex-Top 100 Long-Short Fund,2026-05-22,BSE 500 TRI,65,100,0,25,25,100,,20,"Not stated (65-100% ex-top-100; 0-35% top 100; internal: net sector ~20%, index trades ~5%)","Stock and sector long-short; pairs trading; merger arbitrage; index rebalancing; directional index trades; index pair trading; covered calls; straddles/strangles; stock options/futures arbitrage; index-derivative hedging",Y,N,Y,Y,Y,Y,Harsh Agarwal; Mayur Patel; Milan Mody; Pranav Mise,https://portal.amfiindia.com/spages/S-28.pdf
S-29,INFINITY Hybrid Long-Short Fund,Infinity SIF,Kotak Mahindra Mutual Fund,Hybrid Long-Short Fund,2026-05-26,NIFTY 50 Hybrid Composite Debt 50:50,35,75,25,65,25,100,50,20,Not stated,"Covered call; protective put; collar; cash-secured put; long calls/puts; bull/bear spreads; calendar; straddle; strangle; butterfly; iron condor/butterfly; synthetic long/short; pair trades; stock-level hedging; index futures hedging; IRS/FRA/IRF/CDS",Y,Y,Y,Y,N,Y,Hiten Shah; Kalpesh Jain; Abhishek Bisen,https://portal.amfiindia.com/spages/S-29.pdf
S-30,Summit Equity Long-Short Fund,Summit SIF,Invesco Mutual Fund,Equity Long-Short Fund,2026-06-15,BSE 500 TRI,80,100,0,20,25,100,25,20,Not stated (tactical shorts described as 20-25% of assets),"Tentative short list (short futures; synthetic short; long puts; put/call spreads; short call; ratio put spread); index and sector index futures shorts; covered calls (risk factors)",Y,Y,N,N,N,Y,Hiten Jain,https://portal.amfiindia.com/spages/S-30.pdf
S-31,Apex Equity Long-Short Fund,Apex SIF,Aditya Birla Sun Life Mutual Fund,Equity Long-Short Fund,2026-07-22,NIFTY 500 TRI,80,100,0,20,25,,,20,Not stated (actively managed),"Covered calls; protective puts; index/stock futures-options hedging; long/short pair trades; cash-futures and corporate-action arbitrage; short calls; bear call/put spreads; collar; straddle; strangle; synthetic short; iron butterfly; put butterfly/calendar/diagonal; ratio put spread",Y,Y,Y,Y,Y,Y,Manish Gupta; Harshil Suvarnkar,https://portal.amfiindia.com/spages/S-31.pdf
S-32,Apex Equity Ex-Top 100 Long-Short Fund,Apex SIF,Aditya Birla Sun Life Mutual Fund,Equity Ex-Top 100 Long-Short Fund,2026-07-22,NIFTY 500 TRI,65,100,0,35,25,,,20,Not stated (65-100% ex-top-100; 0-35% other equity),"Covered calls (against holdings only); protective puts; cash-futures arbitrage; long/short pair trades; index/stock futures-options for hedging or beta adjustment; long/short straddles-strangles; directional futures/options; synthetic options and put-call parity trades",Y,Y,Y,Y,Y,Y,Manish Gupta; Harshil Suvarnkar,https://portal.amfiindia.com/spages/S-32.pdf
S-33,Magnum Equity Ex-Top 100 Long Short Fund,Magnum SIF,SBI Mutual Fund,Equity Ex-Top 100 Long-Short Fund,2026-07-31,BSE 500 TRI,65,100,0,35,25,100,,20,Not stated (65-100% ex-top-100; 0-35% top 100),"Short futures; long puts; put spreads; selling calls/call spreads; inverse covered call (short puts with short futures); covered calls; index and stock futures-options hedging; cash-futures arbitrage; interest rate futures for imperfect hedging (up to 20%)",Y,Y,Y,N,Y,Y,Gaurav Mehta,https://portal.amfiindia.com/spages/S-33.pdf
S-34,Altiva Equity Long-Short Fund,Altiva SIF,Edelweiss Mutual Fund,Equity Long-Short Fund,2026-08-20,Nifty 100 TRI,80,100,0,20,25,100,,20,Not stated,"Equity arbitrage; covered calls; index put and index futures hedging; protective stock puts/calls; short calls/puts; short straddles/strangles; long/short futures and options; bull/bear spreads; pair trading (risk factors)",Y,Y,Y,Y,Y,Y,Bharat Lahoti; Bhavesh Jain; Amit Vora (overseas),https://portal.amfiindia.com/spages/S-34.pdf
S-35,Titanium Active Asset Allocator Long-Short Fund,Titanium SIF,Tata Mutual Fund,Active Asset Allocator Long-Short Fund,2026-09-08,35% BSE 200 TRI + 50% CRISIL Short Term Bond + 15% iCOMDEX,35,100,0,65,25,100,,20,Net equity may go as low as 0% (commodity derivatives 0-30%),"Cash-futures arbitrage; pair trades; sector vs index trades; hedging long stocks with short futures; covered calls; call and put writing; protective puts, collars, spreads, straddles, condors (illustrative); IRS/FRA/IRF",Y,Y,Y,Y,Y,Y,Sailesh Jain; Akhil Mittal; Tapan Patel (commodity); Hasmukh Vishariya,https://portal.amfiindia.com/spages/S-35.pdf
S-36,MSIF Equity Long-Short Fund,MSIF,Mahindra Manulife Mutual Fund,Equity Long-Short Fund,2026-09-10,Nifty 500 TRI,80,100,0,20,25,100,50,10,Not stated,"Short exposure via stock, index and sector index futures and options; hedging/offsetting positions; long/short calls and puts; covered calls; options writing (risk factors)",Y,Y,Y,N,N,Y,Aalap Shah; Abhishek Jaiswal,https://portal.amfiindia.com/spages/S-36.pdf
S-37,Summit Active Asset Allocator Long-Short Fund,Summit SIF,Invesco Mutual Fund,Active Asset Allocator Long-Short Fund,2026-09-28,40% Nifty 500 TRI + 40% CRISIL Short Term Bond + 20% iCOMDEX,65,90,10,35,25,100,,20,Not stated (commodity derivatives 0-25%),"Short futures; long puts; put spreads; selling calls/call spreads; inverse covered call; covered calls; protective puts; index futures hedging; arbitrage",Y,Y,Y,N,Y,Y,Deepak Gupta,https://portal.amfiindia.com/spages/S-37.pdf
S-38,Eminence Hybrid Long-Short Fund,Eminence SIF,AlphaGrep Mutual Fund,Hybrid Long-Short Fund,2026-09-09,NIFTY 50 Hybrid Composite Debt 50:50 TRI,25,75,25,75,25,100,,20,Not stated,"Cash-futures arbitrage; covered calls; protective puts; collars; index/stock futures-options for hedging or beta adjustment; pair trades; merger/share-swap and dividend arbitrage; straddles/strangles; IRS/IRF/FRA",Y,Y,Y,Y,Y,Y,Dwaipayan Ray; Ravneet Singh; Ashish Jain; Ramesh S,https://portal.amfiindia.com/spages/S-38.pdf
S-40,SELECT Equity Ex-Top 100 Long-Short Fund,Select SIF,Sundaram Mutual Fund,Equity Ex-Top 100 Long-Short Fund,2026-09-28,Nifty MidSmallcap 400 TRI,65,100,0,35,25,100,,10,Not stated (65-100% ex-top-100; 0-35% other equity),"Covered calls; protective stock puts/calls; index put and index futures hedging; pair trades; equity arbitrage; short calls/puts; bear call/put spreads; short straddles/strangles; long/short futures and options",Y,Y,Y,Y,Y,Y,Madanagopal Ramu; Sandeep Agarwal; Shalav Saket (overseas),https://portal.amfiindia.com/spages/S-40.pdf
S-41,H-SIF Hybrid Long-Short Fund,H-SIF,HDFC Mutual Fund,Hybrid Long-Short Fund,2026-10-28,NIFTY 50 Hybrid Composite Debt 50:50,25,75,25,75,25,100,,20,Not stated,"Equity arbitrage; covered calls; protective puts; long puts; index put and index futures hedging; short calls/puts; bull put/bear call spreads; short straddles/strangles; protective calls; premium harvesting",Y,Y,Y,N,Y,Y,Anil Bamboli; Ihab Dalwai; Paul S Parampreet; Gopal Agrawal (overseas),https://portal.amfiindia.com/spages/S-41.pdf
S-26*,Platinum Hybrid Long-Short Fund,Platinum SIF,Mirae Asset Mutual Fund,Hybrid Long-Short Fund,2026-05-13,Not stated in source,,,25,35,25,,,,"Controlled (net) equity 5-70% incl. collars and special situations; fully hedged arbitrage 5-70%; accrual debt 25-35%","Collars; special situations; arbitrage; accrual debt",N,N,N,N,Y,Collars,Gaurik Shah (quoted; not confirmed as manager),https://www.miraeassetmf.co.in/downloads/press-release/platinum-sif-brought-to-you-by-mirae-asset-mutual-fund-launches-platinum-hybrid-long-short-fund
S-42,Altiva Active Asset Allocator Long-Short Fund,Altiva SIF,Edelweiss Mutual Fund,Active Asset Allocator Long-Short Fund,2026-11-05,50% Nifty 50 TRI + 35% Nifty Short Duration Debt + 7.5% gold + 7.5% silver,65,100,0,35,25,,100,20,Not stated (commodity derivatives 0-30%),"Equity arbitrage; covered calls; index put and index futures hedging; protective stock puts/calls; short calls/puts; short straddles/strangles; long/short futures and options; bear put/call spreads; put-call parity trades; pair trades (risk factors)",Y,Y,Y,Y,Y,Y,Bharat Lahoti; Bhavesh Jain; Hetul Raval (debt); Amit Vora (commodity/overseas),https://portal.amfiindia.com/spages/S-42.pdf
S-43,Infinity Equity Long-Short Fund,Infinity SIF,Kotak Mahindra Mutual Fund,Equity Long-Short Fund,2026-10-27,NIFTY 100 TRI,80,100,0,20,25,100,50,20,Not stated,"Covered calls; protective puts; collars; cash-secured puts; pair trades; stock-level hedging; index futures hedging; short straddles/strangles",Y,Y,Y,Y,N,Y,Kalpesh Jain; Rohit Tandon; Abhishek Bisen,https://portal.amfiindia.com/spages/S-43.pdf
S-44,MSIF Hybrid Long-Short Fund,MSIF,Mahindra Manulife Mutual Fund,Hybrid Long-Short Fund,2026-11-09,CRISIL Hybrid 50+50 Moderate,65,75,25,35,25,100,50,10,Not stated,"Covered calls; cash-futures, dividend, merger, buyback and calendar arbitrage; short calls and long puts against longs; short puts and long calls against shorts; index futures-options hedging; interest rate futures",Y,Y,Y,N,Y,Y,Aalap Shah; Abhishek Jaiswal; Amit Garg; Kush Sonigara,https://portal.amfiindia.com/spages/S-44.pdf
"""

# ───────────────────────── fund report (shared by both views) ─────────────────────────
def fund_report(table, series, nav, pick_label, scope):
    m = table.iloc[(table["Fund"] + " · " + table["AMC"]).tolist().index(pick_label)]
    s = nav[nav.code == m.code].set_index("date")["nav"]
    bm, eq, srcs = series.get(m.code, (_empty(), _empty(), []))
    pr = promise_for(m.Fund)

    st.markdown(f"### {esc(m.Fund)}", unsafe_allow_html=True)
    st.markdown(f'<div class="v-meta">{esc(m.AMC)} · {esc(SHORT.get(m.Category, m.Category))} · launched '
                f'{m.Launched:%d %b %Y} · benchmark: {esc(m.Benchmark)}</div>{verdict_pill(m.Verdict)}',
                unsafe_allow_html=True)

    ann_note = f"annualised from {int(m['Days'])} days" if 0 < m["Days"] < 252 else ""
    tiles([
        tile("Return since launch", f"{m['Since launch']:+.1%}",
             compare_pill(m["Since launch"], m["Benchmark return"], "+.1%")),
        tile("Sharpe ratio", fmt(m["Sharpe"], ".2f"), compare_pill(m["Sharpe"], m["Bmk Sharpe"], ".2f")),
        tile("Sortino ratio", fmt(m["Sortino"], ".2f"), compare_pill(m["Sortino"], m["Bmk Sortino"], ".2f")),
        tile("Max drawdown", fmt(m["Max DD"], ".1%"), compare_pill(m["Max DD"], m["Bmk Max DD"], ".1%")),
        tile("Alpha (annualised)", fmt(m["Alpha"], "+.1%"),
             f"beta {fmt(m['Beta'], '.2f')}" + (f" · {ann_note}" if ann_note else "")),
        tile("Up / down capture", f"{fmt(m['Up capture'], '.2f')} / {fmt(m['Down capture'], '.2f')}",
             pill("asymmetric", "pos") if pd.notna(m["Up capture"]) and m["Up capture"] - m["Down capture"] >= .10
             else pill("symmetric", "mut")),
    ])

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        if PEER_VIEW.startswith("B"):
            with st.container(border=True):
                peer_chart(table, m, f"Standing among {scope}")
        else:
            peer_bars(table, m, f"Standing among {scope}")
    with right:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
        fig.add_scatter(x=s.index, y=s / s.iloc[0] * 100, name="Fund", line=dict(color=GOLD, width=2.4), row=1, col=1)
        fig.add_scatter(x=s.index, y=s / s.cummax() - 1, fill="tozeroy", line=dict(color=GOLD, width=1),
                        showlegend=False, row=2, col=1)
        b = bm[bm.index >= s.index[0]] if not bm.empty else bm
        if not b.empty:
            fig.add_scatter(x=b.index, y=b / b.iloc[0] * 100, name=m.Benchmark,
                            line=dict(color=GREY, width=1.6, dash="dot"), row=1, col=1)
            fig.add_scatter(x=b.index, y=b / b.cummax() - 1, line=dict(color=GREY, width=1, dash="dot"),
                            showlegend=False, row=2, col=1)
        fig.update_yaxes(title="Growth of 100", row=1, col=1)
        fig.update_yaxes(title="Drawdown", tickformat=".0%", row=2, col=1)
        st.plotly_chart(plot_layout(fig, "Fund vs its own benchmark", 430), width="stretch")
        if srcs:
            st.markdown('<div class="v-meta">Benchmark data: ' + " · ".join(
                f"{w:.0%} {esc(n)} → {esc(src)}" for n, w, src in srcs) + "</div>", unsafe_allow_html=True)

    promise_panel(pr, m)

    with st.expander("More detail: period returns, risk, worst days, rolling beta"):
        rows = []
        for k, d in LOOKBACKS.items():
            br = point_return(b, d) if len(b) > 1 else np.nan
            rows.append({"Period": k, "Fund": m[k], "Benchmark": br,
                         "Difference": m[k] - br if pd.notna(m[k]) and pd.notna(br) else np.nan})
        pr_df = pd.DataFrame(rows).set_index("Period")
        c1, c2 = st.columns([1, 1.4], gap="large")
        c1.dataframe(style_table(pr_df, signed=["Difference"]).format("{:+.1%}", na_rep="–"), width="stretch")
        stat = lambda n, v: (f'<div class="v-row"><div class="n">{n}</div>'
                             f'<div class="r" style="width:auto">{v}</div></div>')
        c1.markdown("".join([
            stat("Volatility", fmt(m["Volatility"], ".1%")),
            stat("Tracking error", fmt(m["Tracking error"], ".1%")),
            stat("Information ratio", fmt(m["Info ratio"], ".2f")),
            stat("Down-day wins", fmt(m["Down-day wins"], ".0%")),
            stat("CAGR", fmt(m["CAGR"], "+.1%") + (" (after 1 year)" if pd.isna(m["CAGR"]) else "")),
        ]), unsafe_allow_html=True)
        j = pd.concat([s.pct_change(), bm.pct_change()], axis=1, join="inner").dropna()
        j.columns = ["f", "b"]
        if len(j) >= MIN_OBS:
            w = j.nsmallest(10, "b").sort_index()
            lab = [d.strftime("%d %b %y") for d in w.index]
            fig = go.Figure()
            fig.add_bar(x=lab, y=w.b, name="Benchmark", marker_color=GREY)
            fig.add_bar(x=lab, y=w.f, name="Fund", marker_color=GOLD)
            fig.update_yaxes(tickformat=".1%")
            c2.plotly_chart(plot_layout(fig, "Benchmark's 10 worst days since launch", 300), width="stretch")
        if len(j) >= 50:
            rb = (j.f.rolling(40).cov(j.b) / j.b.rolling(40).var()).dropna()
            fig = go.Figure(go.Scatter(x=rb.index, y=rb, line=dict(color=GOLD, width=2), name="40-day beta"))
            fig.add_hline(y=1, line=dict(color=GREY, dash="dot"))
            st.plotly_chart(plot_layout(fig, "Rolling 40-day beta to its own benchmark", 280), width="stretch")


def pick_fund(table, event, key):
    st.markdown('<div class="v-sec">Fund report · pick a fund here or click a row above</div>',
                unsafe_allow_html=True)
    labels_f = (table["Fund"] + " · " + table["AMC"]).tolist()
    sel = event.selection.rows if event is not None and event.selection else []
    if sel:
        st.session_state[key] = labels_f[sel[0]]
    return st.selectbox("Fund", labels_f, key=key, label_visibility="collapsed")


# ───────────────────────── UI ─────────────────────────
st.markdown(CSS, unsafe_allow_html=True)
nav, meta, status = load_data()

with st.sidebar:
    st.markdown(f"### <span style='color:{GOLD}'>VIKA</span> · SIF Tracker", unsafe_allow_html=True)
    plan = st.radio("Plan", ["Direct", "Regular"], horizontal=True)
    rf = st.number_input("Risk-free rate (% a year)", 0.0, 15.0, 6.5, 0.25,
                         help="Used for Sharpe, Sortino and alpha") / 100
    PEER_VIEW = st.radio("Peer view (pick one)", ["A · Dot strip", "B · Percentile chart"],
                         help="Two designs for the 'Standing among peers' panel. Tell Claude which to keep.")
    with st.expander("Data status", expanded=status["wrote"] is False or bool(status["read_error"])):
        tick = lambda ok: "✅" if ok else "❌"
        where = "GitHub `data` branch" if _gh() else "the app's temporary disk (no GitHub secrets set)"
        save_line = ("Nothing new to save" if status["wrote"] is None else
                     f"Saved to {where} " + tick(status["wrote"]))
        st.markdown(f"{tick(status['mirror'])} History mirror (sif.tigzig.com)  \n"
                    f"{tick(status['amfi'])} AMFI latest NAVs  \n"
                    f"💾 Saved copy: {len(nav):,} rows ({status['saved_rows']:,} before this refresh)  \n"
                    + save_line)
        for err in {status["read_error"], status["error"]} - {None}:
            st.error(err)
        c1, c2 = st.columns(2)
        if c1.button("Refresh now", width="stretch"):
            load_data.clear()
            st.rerun()
        if c2.button("Test GitHub", width="stretch", help="Checks the token, repo and branch, "
                     "and writes a small test file to the data branch"):
            for ok, text in gh_connection_test():
                (st.success if ok else st.error)(text)
    with st.expander("How the numbers work"):
        st.markdown(
            "- Every fund is measured against **the benchmark in its own ISID**, rebuilt from real series.\n"
            "- Each part comes from the best source available: **NSE's official TRI** first; if NSE blocks the "
            "request, the **NAV of an index fund tracking that index** (AMFI data), which is the TRI less a "
            "small expense ratio; Yahoo's price index only as a last resort, labelled \"price only\".\n"
            "- Debt legs use a CRISIL-IBX short-duration gilt + SDL index fund; gold and silver use ETFs. "
            "iCOMDEX has no fund tracking it, so it is split between gold and silver.\n"
            "- Each fund's report lists exactly which source was used for every part of its benchmark.\n"
            f"- Ratios are annualised from daily returns and need {MIN_OBS}+ days; hedge verdicts need "
            f"{HEDGE_MIN_OBS}+. Under a year of history, treat them as indicative.\n"
            "- Net exposure is estimated as beta to the equity part of the fund's benchmark.")
    st.caption("Internal research. Past performance is not indicative of future returns.")

st.title("SIF Performance Tracker")
st.caption(f"NAVs {nav['date'].min():%d %b %Y} → {nav['date'].max():%d %b %Y} · {plan} plan, Growth · "
           "each fund vs the benchmark in its own ISID")
if not status["mirror"] and not status["amfi"]:
    st.warning("Live sources are down. Showing the saved history; it will catch up on the next refresh.")

cats = [c for c in SHORT if c in set(meta["category"])] + \
       sorted(c for c in set(meta["category"]) - set(SHORT) if c)
labels = ["All"] + [SHORT.get(c, c) for c in cats]
choice = st.segmented_control("Category", labels, default="All", label_visibility="collapsed") or "All"
category = None if choice == "All" else cats[labels.index(choice) - 1]
view = st.segmented_control("View", ["🏁 Leaderboard", "🛡️ Hedge check"], default="🏁 Leaderboard",
                            label_visibility="collapsed") or "🏁 Leaderboard"

funds = pick_variants(meta, plan)
if category:
    funds = funds[funds["category"] == category]
table, series = build_table(nav, funds, rf)
if table.empty:
    st.info("No funds in this category yet.")
    st.stop()
scope = f"{choice} funds" if category else "all SIFs"
if category and category in MANDATE:
    st.caption(f"SEBI rule for this category: {MANDATE[category]}")

if view == "🏁 Leaderboard":
    beat = int((table["vs benchmark"] > 0).sum())
    known = int(table["vs benchmark"].notna().sum())
    tiles([
        tile("Funds", str(len(table)), f"{(table['Since launch'] > 0).sum()} above ₹10 issue price"),
        tile("Beating own benchmark", f"{beat} of {known}",
             pill(f"{beat / known:.0%}" if known else "–", "pos" if known and beat / known >= .5 else "neg")),
        tile("Median Sharpe", fmt(table["Sharpe"].median(), ".2f"),
             f"their benchmarks {fmt(table['Bmk Sharpe'].median(), '.2f')}"),
        tile("Median max drawdown", fmt(table["Max DD"].median(), ".1%"),
             f"their benchmarks {fmt(table['Bmk Max DD'].median(), '.1%')}"),
        tile("Median down capture", fmt(table["Down capture"].median(), ".2f"), "below 1 = protects"),
    ])
    board = table.sort_values("vs benchmark", ascending=False, na_position="last").reset_index(drop=True)
    show = board[["Fund", "Category", "Since launch", "vs benchmark", "Sharpe", "Sortino", "Alpha",
                  "Max DD", "Up capture", "Down capture", "Benchmark"]].copy()
    show.insert(0, "#", range(1, len(show) + 1))
    show["Category"] = show["Category"].map(lambda c: SHORT.get(c, c))
    if category:
        show = show.drop(columns="Category")
    st.markdown('<div class="v-sec">Leaderboard · ranked by return over its own benchmark · '
                'click a row for the full report</div>', unsafe_allow_html=True)
    sty = style_table(show, good_high=["Sharpe", "Sortino", "Max DD"],
                      good_low=["Down capture"], signed=["Since launch", "vs benchmark", "Alpha"])
    sty = sty.format({"Since launch": "{:+.1%}", "vs benchmark": "{:+.1%}", "Alpha": "{:+.1%}",
                      "Sharpe": "{:.2f}", "Sortino": "{:.2f}", "Max DD": "{:.1%}",
                      "Up capture": "{:.2f}", "Down capture": "{:.2f}"}, na_rep="–")
    event = st.dataframe(sty, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                         key=f"board_{choice}_{plan}", column_config={
                             "#": st.column_config.NumberColumn(width="small"),
                             "Fund": st.column_config.Column(width="medium"),
                             "vs benchmark": st.column_config.Column("vs own bmk"),
                             "Benchmark": st.column_config.Column("Own benchmark (ISID)")})
    st.divider()
    pick = pick_fund(board, event, f"pick_{choice}_{plan}")
    fund_report(board, series, nav, pick, scope)

else:  # hedge check
    counts = table["Verdict"].str.split(" ").str[0].value_counts()
    tiles([
        tile("🟢 Hedging works", str(int(counts.get("🟢", 0))), "falls less, keeps upside"),
        tile("🟡 Partial", str(int(counts.get("🟡", 0))), "protects, but symmetric or light"),
        tile("🔴 / ⚠️ Not as promised", str(int(counts.get("🔴", 0) + counts.get("⚠️", 0))),
             "no visible hedge, or over its ISID limit"),
        tile("⏳ Too early", str(int(counts.get("⏳", 0))), f"under {HEDGE_MIN_OBS} trading days"),
    ])
    order = {"🟢": 0, "🟡": 1, "⚠️": 2, "🔴": 3, "⏳": 4}
    hc = table.assign(_o=table["Verdict"].str.split(" ").str[0].map(order)) \
              .sort_values(["_o", "Down capture"]).reset_index(drop=True)
    show = hc[["Fund", "Verdict", "Down capture", "Up capture", "Down-day wins", "Worst 10: fund",
               "Worst 10: bmk", "Equity beta", "Ceiling", "Benchmark"]].copy()
    st.markdown('<div class="v-sec">Hedge check · each fund vs its own benchmark · click a row for the full report'
                '</div>', unsafe_allow_html=True)
    sty = style_table(show, good_high=["Down-day wins"], good_low=["Down capture"],
                      signed=["Worst 10: fund", "Worst 10: bmk"])
    sty = sty.format({"Down capture": "{:.2f}", "Up capture": "{:.2f}", "Down-day wins": "{:.0%}",
                      "Worst 10: fund": "{:+.2%}", "Worst 10: bmk": "{:+.2%}", "Equity beta": "{:.2f}",
                      "Ceiling": "{:.2f}"}, na_rep="–")
    event = st.dataframe(sty, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                         key=f"hedge_{choice}_{plan}", column_config={
                             "Fund": st.column_config.Column(width="medium"),
                             "Verdict": st.column_config.Column(width="medium"),
                             "Equity beta": st.column_config.Column("Net exposure (est.)"),
                             "Ceiling": st.column_config.Column("ISID max"),
                             "Worst 10: fund": st.column_config.Column("Worst days: fund",
                                 help="Average move on its benchmark's 10 worst days since launch"),
                             "Worst 10: bmk": st.column_config.Column("Worst days: bmk"),
                             "Benchmark": st.column_config.Column("Own benchmark (ISID)")})
    r = hc.dropna(subset=["Down capture", "Up capture"])
    if len(r):
        colors = [{"🟢": POS, "🟡": NEU, "🔴": NEG, "⚠️": "#E3842B"}.get(v.split(" ")[0], GREY) for v in r.Verdict]
        fig = go.Figure(go.Scatter(
            x=r["Down capture"], y=r["Up capture"], mode="markers+text", text=r.AMC, textposition="top center",
            textfont=dict(color=GREY, size=11), marker=dict(size=13, color=colors),
            customdata=np.stack([r.Fund, r.Verdict], axis=1),
            hovertemplate="%{customdata[0]}<br>%{customdata[1]}<br>Down %{x:.2f} · Up %{y:.2f}<extra></extra>"))
        hi = max(1.2, float(r[["Down capture", "Up capture"]].max().max()) + 0.1)
        lo = min(0.0, float(r[["Down capture", "Up capture"]].min().min()) - 0.1)
        fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(color=GREY, dash="dot"))
        fig.add_annotation(x=lo + 0.04, y=hi - 0.04, text="Real hedge: falls less, keeps upside", showarrow=False,
                           xanchor="left", font=dict(color=POS, size=11))
        fig.add_annotation(x=hi - 0.04, y=lo + 0.04, text="Worst of both", showarrow=False, xanchor="right",
                           font=dict(color=NEG, size=11))
        fig.update_xaxes(title="Down capture vs own benchmark (lower is better)", range=[lo, hi])
        fig.update_yaxes(title="Up capture vs own benchmark (higher is better)", range=[lo, hi])
        with st.expander("Capture map", expanded=True):
            st.plotly_chart(plot_layout(fig, "", 420), width="stretch")
    st.divider()
    pick = pick_fund(hc, event, f"hpick_{choice}_{plan}")
    fund_report(hc, series, nav, pick, scope)

with st.sidebar:
    st.download_button("Download table (CSV)", table.drop(columns=["code"]).to_csv(index=False),
                       "sif_funds.csv", width="stretch")
    with st.expander("Benchmark data sources in use"):
        used = sorted({(n, src) for v in series.values() for n, _, src in v[2]})
        st.markdown("\n".join(f"- **{n}**: {src}" for n, src in used) or "–")
