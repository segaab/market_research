# app.py
import time
import json
import calendar
import concurrent.futures
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sodapy import Socrata
from yahooquery import Ticker

# ── APP CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Multi-Asset Health Gauge", layout="wide")
START_DATE = "2013-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
COT_PAGE_SIZE = 15000       # Socrata max page size
COT_SLEEP    = 0.35        # polite delay between pages (sec)
YH_SLEEP     = 0.2         # polite delay between tickers (sec)

# ── MAPPINGS ──────────────────────────────────────────────────────────────────
ASSET_LEADERS: Dict[str, List[str]] = {
    "Indices":       ["^GSPC"],
    "Forex":         ["EURUSD=X", "JPY=X"],
    "Agricultural":  ["ZS=F"],
    "Energy":        ["CL=F"],
    "Metals":        ["GC=F"],
}

TICKER_TO_NAME = {
    "^GSPC":   "S&P 500",
    "EURUSD=X":"EUR/USD",
    "JPY=X":   "USD/JPY",
    "ZS=F":    "Soybeans",
    "CL=F":    "WTI Crude",
    "GC=F":    "Gold",
}

COT_NAMES = {
    "EURUSD=X": "EURO FX -- Chicago Mercantile Exchange",
    "JPY=X":     "JAPANESE YEN -- Chicago Mercantile Exchange",
    "ZS=F":      "SOYBEANS -- Chicago Board of Trade",
    "CL=F":      "WTI-PHYSICAL -- New York Mercantile Exchange",
    "GC=F":      "GOLD -- Commodity Exchange Inc.",
}

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
category_selected = st.sidebar.selectbox(
    "Choose Asset Category", list(ASSET_LEADERS.keys())
)
rvol_window = st.sidebar.number_input(
    "RVol Rolling Window (days)", min_value=5, max_value=60, value=20
)

# ── CACHING HELPERS ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _yahoo_session() -> Ticker:
    "One shared YahooQuery client (re-uses HTTP session & cookies)."
    return Ticker([])          # empty now; symbols added per request

@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_price_history(ticker: str) -> pd.DataFrame:
    "Daily OHLCV from Yahoo; cached for 1 h."
    yq = _yahoo_session()
    yq.symbols = [ticker]      # attach symbol dynamically
    df = yq.history(start=START_DATE, end=END_DATE, auto_adjust=False)
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    df["return_pct"] = df["close"].pct_change() * 100
    time.sleep(YH_SLEEP)       # polite spacing
    return df

@st.cache_data(show_spinner=False, ttl=24 * 3600)
def fetch_cot_data(cot_name: str, start_date: str, end_date: str) -> pd.DataFrame:
    "Weekly COT report; full-day cache"
    client = Socrata("publicreporting.cftc.gov", None, timeout=30)
    where = (
        f"market_and_exchange_names='{cot_name}' AND "
        f"report_date_as_yyyy_mm_dd >= '{start_date}' AND "
        f"report_date_as_yyyy_mm_dd <= '{end_date}'"
    )
    rows: List[dict] = []
    offset = 0
    while True:
        batch = client.get("6dca-aqww", where=where,
                           order="report_date_as_yyyy_mm_dd",
                           limit=COT_PAGE_SIZE, offset=offset)
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
    keep = [
        "report_date_as_yyyy_mm_dd",
        "open_interest_all",
        "commercial_long_all",
        "commercial_short_all",
    ]
    df = df[keep]
    df = df.apply(pd.to_numeric, errors="ignore")
    df["report_date_as_yyyy_mm_dd"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
    df["commercial_net"] = df["commercial_long_all"] - df["commercial_short_all"]
    return df.sort_values("report_date_as_yyyy_mm_dd").reset_index(drop=True)

# ── METRICS ───────────────────────────────────────────────────────────────────
def add_rvol(df: pd.DataFrame, window: int) -> pd.DataFrame:
    out = df.copy()
    rolling_mean = out["volume"].rolling(window).mean().replace(0, np.nan)
    out["rvol"] = out["volume"] / rolling_mean
    return out

def merge_cot_price(cot: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    if price.empty:
        return pd.DataFrame()
    cot = cot.copy()
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
    )
    merged = merged.drop(columns="cot_date")
    total = (merged["commercial_long_all"] + merged["commercial_short_all"]).replace(0, np.nan)
    merged["cot_long_norm"]  = merged["commercial_long_all"]  / total
    merged["cot_short_norm"] = merged["commercial_short_all"] / total
    return merged

def add_health_gauge(df: pd.DataFrame,
                     weights: Dict[str, float] = {"rvol": .5, "cot_long": .3, "cot_short": .2}
                    ) -> pd.DataFrame:
    out = df.copy()
    out["health_gauge"] = (
        weights["rvol"]      * out["rvol"].fillna(1) +
        weights["cot_long"]  * out["cot_long_norm"].fillna(0) -
        weights["cot_short"] * out["cot_short_norm"].fillna(0)
    )
    return out

def _category_for_ticker(tkr: str) -> tuple[str | None, str | None]:
    for cat, lst in ASSET_LEADERS.items():
        if tkr in lst:
            return cat, TICKER_TO_NAME[tkr]
    return None, None

# ── FETCH ALL DATA ────────────────────────────────────────────────────────────
tickers = ASSET_LEADERS[category_selected]
with st.spinner("Downloading & crunching …"):
    # Yahoo prices in parallel --------------------------------------------------
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        price_futs = {ex.submit(fetch_price_history, t): t for t in tickers}
        price_data = {price_futs[f]: f.result() for f in concurrent.futures.as_completed(price_futs)}

    # COT sequential (cached & throttled) --------------------------------------
    cot_data = {t: fetch_cot_data(COT_NAMES.get(t, ""), START_DATE, END_DATE) for t in tickers}

    # Merge, rvol, health ------------------------------------------------------
    merged_data = {}
    for t in tickers:
        price = add_rvol(price_data[t], rvol_window)
        merged = merge_cot_price(cot_data[t], price)
        merged = add_health_gauge(merged)
        merged_data[t] = merged

# ── PIP DISTRIBUTION ──────────────────────────────────────────────────────────
def pip_distribution_tree(data: dict[str, pd.DataFrame], category: str):
    tree: dict[str, dict] = {}
    month_list = [calendar.month_abbr[m] for m in range(1, 13)]

    for tkr, df in data.items():
        cat, asset_name = _category_for_ticker(tkr)
        if cat != category or df.empty:
            continue
        tmp = df.copy()
        tmp["month_num"] = pd.to_datetime(tmp["date"]).dt.month

        monthly = (
            tmp.groupby("month_num")["close"]
            .agg(["min", "max", "mean", "count"])
            .sort_index()
        )
        monthly["normalized"] = monthly["count"] / monthly["count"].sum()

        month_dict: dict[str, dict] = {}
        for m, row in monthly.round(4).iterrows():
            month_dict[calendar.month_abbr[m]] = row.to_dict()

        agg_row = {
            "min": round(tmp["close"].min(), 4),
            "max": round(tmp["close"].max(), 4),
            "mean": round(tmp["close"].mean(), 4),
            "count": int(tmp["close"].count()),
            "normalized": 1.0,
        }
        month_dict["Aggregate"] = agg_row
        tree[asset_name] = month