#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Multi-Asset Health Gauge – Streamlit application
# -----------------------------------------------------------------------------
import json
import time
import calendar
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sodapy import Socrata
from yahooquery import Ticker

# ── APP CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Multi-Asset Health Gauge", layout="wide")

LATEST_DATE = datetime.today()
START_DATE = (LATEST_DATE - timedelta(days=365*10)).strftime("%Y-%m-%d")
END_DATE = LATEST_DATE.strftime("%Y-%m-%d")

COT_PAGE_SIZE = 15000
COT_SLEEP = 0.35
YH_SLEEP = 0.20

# ── ASSET CONFIG ───────────────────────────────────────────────────────────
ASSET_LEADERS: Dict[str, List[str]] = {
    "Indices": ["^GSPC"],
    "Forex": ["EURUSD=X", "USDJPY=X"],
    "Agricultural": ["ZS=F"],
    "Energy": ["CL=F"],
    "Metals": ["GC=F"],
}

TICKER_TO_NAME = {
    "^GSPC": "S&P 500",
    "EURUSD=X": "EUR/USD",
    "USDJPY=X": "USD/JPY",
    "ZS=F": "Soybeans",
    "CL=F": "WTI Crude",
    "GC=F": "Gold",
}

COT_ASSET_NAMES = {
    "^GSPC": "S&P 500 -- Chicago Mercantile Exchange",
    "EURUSD=X": "EURO FX -- Chicago Mercantile Exchange",
    "USDJPY=X": "JAPANESE YEN -- Chicago Mercantile Exchange",
    "ZS=F": "SOYBEANS -- Chicago Board of Trade",
    "CL=F": "WTI-PHYSICAL -- New York Mercantile Exchange",
    "GC=F": "GOLD -- Commodity Exchange Inc."
}

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
category_selected = st.sidebar.selectbox(
    "Choose Asset Category", list(ASSET_LEADERS.keys())
)
rvol_window = st.sidebar.number_input(
    "RVol Rolling Window (days)", 5, 60, 20
)
normalize_monthly_checkbox = st.sidebar.checkbox(
    "Normalize monthly distribution counts", value=True
)

# ── CACHING & FETCHING HELPERS ───────────────────────────────────────────────
SODAPY_APP_TOKEN = "WSCaavlIcDgtLVZbJA1FKkq40"

@st.cache_resource(show_spinner=False)
def _yahoo_session() -> Ticker:
    return Ticker([])

@st.cache_data(show_spinner=False, ttl=60*60)
def fetch_price_history(ticker: str) -> pd.DataFrame:
    yq = _yahoo_session()
    yq.symbols = [ticker]
    df = yq.history(start=START_DATE, end=END_DATE)
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    df["return_pct"] = df["close"].pct_change() * 100
    time.sleep(YH_SLEEP)
    return df

@st.cache_data(show_spinner=False, ttl=24*3600)
def fetch_cot_data(cot_name: str, start_date: str, end_date: str) -> pd.DataFrame:
    if not cot_name:
        return pd.DataFrame()
    client = Socrata("publicreporting.cftc.gov", SODAPY_APP_TOKEN, timeout=120)
    where = (
        f"market_and_exchange_names='{cot_name}' AND "
        f"report_date_as_yyyy_mm_dd >= '{start_date}' AND "
        f"report_date_as_yyyy_mm_dd <= '{end_date}'"
    )
    rows, offset = [], 0
    while True:
        batch = client.get(
            "6dca-aqww", where=where,
            order="report_date_as_yyyy_mm_dd",
            limit=COT_PAGE_SIZE, offset=offset
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
    if "report_date_as_yyyy_mm_dd" not in df.columns:
        return pd.DataFrame()
    keep = [c for c in ["report_date_as_yyyy_mm_dd","commercial_long_all","commercial_short_all","open_interest_all"] if c in df.columns]
    df = df[keep]
    df["timestamp"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
    if {"commercial_long_all","commercial_short_all"} <= set(df.columns):
        df["commercial_net"] = df["commercial_long_all"] - df["commercial_short_all"]
    df_daily = (
        df.set_index("timestamp")
          .sort_index()
          .reindex(pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="D"))
          .ffill()
          .reset_index()
          .rename(columns={"index": "timestamp"})
    )
    return df_daily.sort_values("timestamp").reset_index(drop=True)

# ── METRIC HELPERS ─────────────────────────────────────────────────────────
def add_rvol(df: pd.DataFrame, window: int) -> pd.DataFrame:
    out = df.copy()
    if "volume" in out.columns:
        out["rvol"] = out["volume"] / out["volume"].rolling(window).mean().replace(0,np.nan)
    else:
        out["rvol"] = np.nan
    return out

def merge_cot_price(cot: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    if price.empty or cot.empty or "timestamp" not in cot.columns:
        return pd.DataFrame()
    
    cot_copy = cot.copy()
    cot_copy["cot_date"] = cot_copy["timestamp"] - pd.to_timedelta(cot_copy["timestamp"].dt.weekday - 4, unit="D")
    cot_copy = cot_copy[["cot_date","commercial_long_all","commercial_short_all"]]
    
    merged = pd.merge_asof(
        price.sort_values("timestamp"),
        cot_copy.sort_values("cot_date"),
        left_on="timestamp", right_on="cot_date", direction="backward"
    ).drop(columns="cot_date")
    
    tot = (merged.get("commercial_long_all",0) + merged.get("commercial_short_all",0)).replace(0,np.nan)
    merged["cot_long_norm"] = merged.get("commercial_long_all",0) / tot
    merged["cot_short_norm"] = merged.get("commercial_short_all",0) / tot
    
    return merged

def add_health_gauge(df: pd.DataFrame, weights: Dict[str,float] = {"rvol":0.5,"cot_long":0.3,"cot_short":0.2}) -> pd.DataFrame:
    out = df.copy()
    for col in ["rvol","cot_long_norm","cot_short_norm"]:
        if col not in out.columns:
            out[col] = 0
    out["health_gauge"] = (
        weights["rvol"]*out["rvol"].fillna(1) +
        weights["cot_long"]*out["cot_long_norm"].fillna(0) -
        weights["cot_short"]*out["cot_short_norm"].fillna(0)
    )
    return out

def compute_daily_pip_range(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not {"high","low"}.issubset(df.columns):
        return pd.DataFrame()
    df["pip_range"] = df["high"] - df["low"]
    df["month"] = df["timestamp"].dt.to_period("M").dt.month
    monthly_pip = df.groupby("month")["pip_range"].mean().reset_index()
    return monthly_pip

# ── FETCH + PROCESS ASSET ──────────────────────────────────────────────────
def fetch_process_asset(ticker: str):
    price = fetch_price_history(ticker)
    cot = fetch_cot_data(COT_ASSET_NAMES.get(ticker,""), START_DATE, END_DATE)
    if price.empty or cot.empty:
        return None, None
    price = add_rvol(price, rvol_window)
    merged = merge_cot_price(cot, price)
    health_df = add_health_gauge(merged)
    
    # Build daily JSON tree
    daily_tree = health_df[[
        "timestamp","close","rvol","cot_long_norm","cot_short_norm","health_gauge"
    ]].rename(columns={"timestamp":"date"}).to_dict(orient="records")
    
    # Build monthly JSON tree
    monthly = health_df.copy()
    monthly["month"] = monthly["date"].dt.month
    monthly_stats = monthly.groupby("month")["health_gauge"].mean().reset_index()
    monthly_stats = monthly_stats.rename(columns={"health_gauge":"mean_health"})
    monthly_tree = monthly_stats.to_dict(orient="records")
    
    return daily_tree, monthly_tree

def process_category(category: str):
    tickers = ASSET_LEADERS.get(category, [])
    daily_json, monthly_json = {}, {}

    for t in tickers:
        st.info(f"Processing {TICKER_TO_NAME[t]}...")

        # Fetch and process asset
        daily_tree, monthly_tree = fetch_process_asset(t)
        if daily_tree is None or monthly_tree is None:
            st.warning(f"No data available for {TICKER_TO_NAME[t]}")
            continue

        # Store JSON trees per ticker
        daily_json[t] = daily_tree
        monthly_json[t] = monthly_tree

        # Visualize health gauge
        df_plot = pd.DataFrame(daily_tree)
        if not df_plot.empty:
            fig = px.line(df_plot, x="date", y="health_gauge", title=f"Health Gauge - {TICKER_TO_NAME[t]}")
            st.plotly_chart(fig, use_container_width=True)

        # Visualize monthly stats
        df_monthly = pd.DataFrame(monthly_tree)
        if not df_monthly.empty:
            fig2 = px.bar(
                df_monthly,
                x="month",
                y="mean_health",
                title=f"Monthly Mean Health Gauge - {TICKER_TO_NAME[t]}"
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── JSON EXPORT ─────────────────────────────────────────────────────────
    st.download_button(
        label="Download Daily JSON Tree",
        data=json.dumps(daily_json, indent=2, default=str),
        file_name=f"daily_health_json_{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

    st.download_button(
        label="Download Monthly JSON Tree",
        data=json.dumps(monthly_json, indent=2, default=str),
        file_name=f"monthly_health_json_{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

    st.success(f"Processing complete for category: {category}")