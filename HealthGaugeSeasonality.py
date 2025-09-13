#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Multi-Asset Health Gauge – Streamlit application
# -----------------------------------------------------------------------------
import json
import time
import calendar
import concurrent.futures
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sodapy import Socrata
from yahooquery import Ticker

# ── APP CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Multi-Asset Health Gauge", layout="wide")

START_DATE = "2013-01-01"
END_DATE   = datetime.today().strftime("%Y-%m-%d")

COT_PAGE_SIZE = 15_000
COT_SLEEP     = 0.35            # seconds between paged COT calls
YH_SLEEP      = 0.20            # seconds between Yahoo sub-calls

# ── MAPPINGS ──────────────────────────────────────────────────────────────────
ASSET_LEADERS: Dict[str, List[str]] = {
    "Indices":       ["^GSPC"],
    "Forex":         ["EURUSD=X", "USDJPY=X"],
    "Agricultural":  ["ZS=F"],
    "Energy":        ["CL=F"],
    "Metals":        ["GC=F"],
}

TICKER_TO_NAME = {
    "^GSPC":   "S&P 500",
    "EURUSD=X":"EUR/USD",
    "USDJPY=X":"USD/JPY",
    "ZS=F":    "Soybeans",
    "CL=F":    "WTI Crude",
    "GC=F":    "Gold",
}

COT_ASSET_NAMES = {
    "^GSPC":    "S&P 500 -- Chicago Mercantile Exchange",
    "^GDAXI":   None,
    "EURUSD=X": "EURO FX -- Chicago Mercantile Exchange",
    "USDJPY=X": "JAPANESE YEN -- Chicago Mercantile Exchange",
    "ZS=F":     "SOYBEANS -- Chicago Board of Trade",
    "CL=F":     "WTI-PHYSICAL -- New York Mercantile Exchange",
    "GC=F":     "GOLD -- Commodity Exchange Inc."
}

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
category_selected = st.sidebar.selectbox(
    "Choose Asset Category", list(ASSET_LEADERS.keys())
)
rvol_window = st.sidebar.number_input(
    "RVol Rolling Window (days)", min_value=5, max_value=60, value=20
)
normalize_monthly_checkbox = st.sidebar.checkbox(
    "Normalize monthly distribution counts", value=True
)

# ── CACHING HELPERS ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _yahoo_session() -> Ticker:
    """A singleton YahooFinance session object (cached)."""
    return Ticker([])

@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_price_history(ticker: str) -> pd.DataFrame:
    """Download daily OHLCV price data from Yahoo Finance."""
    yq = _yahoo_session()
    yq.symbols = [ticker]
    df = yq.history(start=START_DATE, end=END_DATE)

    if df.empty:
        return pd.DataFrame()

    df = df.reset_index()
    df["date"]   = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    df["return_pct"] = df["close"].pct_change() * 100
    time.sleep(YH_SLEEP)
    return df

@st.cache_data(show_spinner=False, ttl=24 * 3600)
def fetch_cot_data(cot_name: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch Commitment-of-Traders weekly report from the CFTC Socrata endpoint.

    A defensive layer is added so the function safely returns an empty DataFrame
    whenever expected key columns are missing in the API response.
    """
    if not cot_name:
        return pd.DataFrame()

    client = Socrata("publicreporting.cftc.gov", None, timeout=30)
    where = (
        f"market_and_exchange_names='{cot_name}' AND "
        f"report_date_as_yyyy_mm_dd >= '{start_date}' AND "
        f"report_date_as_yyyy_mm_dd <= '{end_date}'"
    )

    rows: List[dict] = []
    offset = 0
    while True:
        batch = client.get(
            "6dca-aqww",
            where=where,
            order="report_date_as_yyyy_mm_dd",
            limit=COT_PAGE_SIZE,
            offset=offset,
        )
        if not batch:
            break
        rows.extend(batch)
        offset += COT_PAGE_SIZE
        if len(batch) < COT_PAGE_SIZE:
            break
        time.sleep(COT_SLEEP)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(rows)

    # ── Robustness guard – make sure key column exists ────────────────────────
    if "report_date_as_yyyy_mm_dd" not in df.columns:
        return pd.DataFrame()

    keep_cols = [
        "report_date_as_yyyy_mm_dd",
        "open_interest_all",
        "commercial_long_all",
        "commercial_short_all",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    if "report_date_as_yyyy_mm_dd" not in keep_cols:
        return pd.DataFrame()

    df = df[keep_cols]
    df = df.apply(pd.to_numeric, errors="ignore")
    df["report_date_as_yyyy_mm_dd"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])

    if {"commercial_long_all", "commercial_short_all"} <= set(df.columns):
        df["commercial_net"] = df["commercial_long_all"] - df["commercial_short_all"]

    return df.sort_values("report_date_as_yyyy_mm_dd").reset_index(drop=True)

# ── METRIC BUILDERS ───────────────────────────────────────────────────────────
def add_rvol(df: pd.DataFrame, window: int) -> pd.DataFrame:
    out = df.copy()
    rolling_mean = out["volume"].rolling(window).mean().replace(0, np.nan)
    out["rvol"] = out["volume"] / rolling_mean
    return out

def merge_cot_price(cot: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    if price.empty:
        return pd.DataFrame()

    cot = cot.copy()
    # Align Friday COT date to corresponding trading day
    cot["cot_date"] = cot["report_date_as_yyyy_mm_dd"] - pd.to_timedelta(
        cot["report_date_as_yyyy_mm_dd"].dt.weekday - 4, unit="D"
    )
    cot = cot[["cot_date", "commercial_long_all", "commercial_short_all"]]

    merged = pd.merge_asof(
        price.sort_values("date"),
        cot.sort_values("cot_date"),
        left_on="date",
        right_on="cot_date",
        direction="backward",
    ).drop(columns="cot_date")

    total = (merged["commercial_long_all"] + merged["commercial_short_all"]).replace(0, np.nan)
    merged["cot_long_norm"]  = merged["commercial_long_all"]  / total
    merged["cot_short_norm"] = merged["commercial_short_all"] / total
    return merged

def add_health_gauge(
    df: pd.DataFrame,
    weights: Dict[str, float] = {"rvol": 0.5, "cot_long": 0.3, "cot_short": 0.2},
) -> pd.DataFrame:
    out = df.copy()
    out["health_gauge"] = (
        weights["rvol"]      * out["rvol"].fillna(1) +
        weights["cot_long"]  * out["cot_long_norm"].fillna(0) -
        weights["cot_short"] * out["cot_short_norm"].fillna(0)
    )
    return out

# ── FETCH DATA ────────────────────────────────────────────────────────────────
tickers = ASSET_LEADERS[category_selected]

with st.spinner("Downloading & crunching data …"):
    # Yahoo prices – parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futs = {pool.submit(fetch_price_history, t): t for t in tickers}
        price_data = {t: f.result() for f, t in ((f, futs[f]) for f in futs)}

    # COT – sequential (API has its own rate limit)
    cot_data = {
        t: fetch_cot_data(COT_ASSET_NAMES.get(t, ""), START_DATE, END_DATE)
        for t in tickers
    }

    # Merge + indicators
    merged_data = {}
    for t in tickers:
        price = add_rvol(price_data[t], rvol_window)
        merged = merge_cot_price(cot_data[t], price)
        merged = add_health_gauge(merged)
        merged_data[t] = merged

# ── VISUALISATIONS ────────────────────────────────────────────────────────────
st.markdown("## Rolling Volatility")
for t in tickers:
    df = merged_data[t]
    if df.empty:
        st.warning(f"No data for {TICKER_TO_NAME[t]}")
        continue
    fig = px.line(
        df,
        x="date",
        y="rvol",
        title=f"{TICKER_TO_NAME[t]} — RVol ({rvol_window}-day)",
        template="plotly_white",
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

# ── RETURN DISTRIBUTION ───────────────────────────────────────────────────────
st.markdown("## Return Distribution")
df_results = pd.concat(
    [
        d.loc[d["return_pct"].notna(), ["return_pct"]].assign(asset=TICKER_TO_NAME[t])
        for t, d in merged_data.items()
        if not d.empty
    ],
    ignore_index=True,
)

if df_results.empty:
    st.info("No return data available.")
else:
    fig = px.histogram(
        df_results,
        x="return_pct",
        nbins=20,
        color_discrete_sequence=["#3366CC"],
        title="Daily % Return Distribution",
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

# ── HEALTH GAUGE OVER TIME ────────────────────────────────────────────────────
st.markdown("## Health Gauge")
for t in tickers:
    df = merged_data[t]
    if df.empty:
        continue
    fig = px.line(
        df,
        x="date",
        y="health_gauge",
        title=f"{TICKER_TO_NAME[t]} — Health Gauge",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

# ── PIP & RETURN DISTRIBUTION TREE ────────────────────────────────────────────
def _category_for_ticker(ticker: str) -> tuple[str, str]:
    for cat, _tickers in ASSET_LEADERS.items():
        if ticker in _tickers:
            return cat, TICKER_TO_NAME.get(ticker, ticker)
    return "", ticker

def pip_and_return_tree_daily_pips(
    data: Dict[str, pd.DataFrame],
    category: str,
    normalize: bool = True,
) -> Dict[str, Dict]:
    tree: Dict[str, Dict] = {}
    for tkr, df in data.items():
        cat, asset_name = _category_for_ticker(tkr)
        if cat != category or df.empty:
            continue

        tmp = df.copy()
        tmp["date"] = pd.to_datetime(tmp["date"])
        tmp = tmp.sort_values("date")
        tmp["month_num"] = tmp["date"].dt.month
        tmp["daily_pip"] = tmp["high"] - tmp["low"]
        tmp["return"] = tmp["close"].pct_change() * 100

        tmp_returns = tmp.dropna(subset=["return"])
        tmp_pips    = tmp.dropna(subset=["daily_pip"])
        month_dict: Dict[str, Dict] = {}

        for m in range(1, 13):
            pip_month = tmp_pips[tmp_pips["month_num"] == m]
            ret_month = tmp_returns[tmp_returns["month_num"] == m]

            if pip_month.empty and ret_month.empty:
                continue

            pip_stats = {
                "min":   round(pip_month["daily_pip"].min(),  4) if not pip_month.empty else None,
                "max":   round(pip_month["daily_pip"].max(),  4) if not pip_month.empty else None,
                "mean":  round(pip_month["daily_pip"].mean(), 4) if not pip_month.empty else None,
                "count": int(pip_month["daily_pip"].count()),
                "normalized": (pip_month["daily_pip"].count() / len(tmp_pips))
                              if normalize and not pip_month.empty else 0,
            }
            ret_stats = {
                "min":   round(ret_month["return"].min(),  4) if not ret_month.empty else None,
                "max":   round(ret_month["return"].max(),  4) if not ret_month.empty else None,
                "mean":  round(ret_month["return"].mean(), 4) if not ret_month.empty else None,
                "count": int(ret_month["return"].count()),
                "normalized": (ret_month["return"].count() / len(tmp_returns))
                              if normalize and not ret_month.empty else 0,
            }

            month_dict[calendar.month_abbr[m]] = {
                "daily_pip": pip_stats,
                "return":    ret_stats,
            }

        # Aggregate node
        month_dict["Aggregate"] = {
            "daily_pip": {
                "min":   round(tmp_pips["daily_pip"].min(),  4),
                "max":   round(tmp_pips["daily_pip"].max(),  4),
                "mean":  round(tmp_pips["daily_pip"].mean(), 4),
                "count": int(tmp_pips["daily_pip"].count()),
                "normalized": 1.0 if normalize else None,
            },
            "return": {
                "min":   round(tmp_returns["return"].min(),  4),
                "max":   round(tmp_returns["return"].max(),  4),
                "mean":  round(tmp_returns["return"].mean(), 4),
                "count": int(tmp_returns["return"].count()),
                "normalized": 1.0 if normalize else None,
            },
        }

        tree[asset_name] = month_dict
    return tree

# ── JSON EXPORTS ──────────────────────────────────────────────────────────────
st.markdown("## Download Merged Data & Distribution")
export_cols = [
    "date", "open", "high", "low", "close", "volume",
    "return_pct", "rvol",
    "commercial_long_all", "commercial_short_all",
    "cot_long_norm", "cot_short_norm", "health_gauge",
]

payload = {
    t: d[export_cols].round(6).fillna(None).to_dict(orient="records")
    for t, d in merged_data.items()
    if not d.empty
}

st.download_button(
    "Download JSON (Merged Data)",
    data=json.dumps(payload, indent=2, default=str),
    file_name=f"health_gauge_merged_{category_selected.lower()}.json",
    mime="application/json",
)

# Daily pip & return distribution tree
pip_return_tree = pip_and_return_tree_daily_pips(
    merged_data, category_selected, normalize=normalize_monthly_checkbox
)

st.download_button(
    "Download JSON (Daily Pip & Return Distribution)",
    data=json.dumps(pip_return_tree, indent=2, default=str),
    file_name=f"daily_pip_return_{category_selected.lower()}.json",
    mime="application/json",
)
