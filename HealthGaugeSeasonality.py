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
COT_SLEEP     = 0.35         # s between paged COT calls
YH_SLEEP      = 0.20         # s between Yahoo sub-calls

# ── MAPPINGS ─────────────────────────────────────────────────────────────────
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

# Name string used by CFTC endpoint
COT_ASSET_NAMES = {
    "^GSPC":    "S&P 500 -- Chicago Mercantile Exchange",
    "^GDAXI":   None,
    "EURUSD=X": "EURO FX -- Chicago Mercantile Exchange",
    "USDJPY=X": "JAPANESE YEN -- Chicago Mercantile Exchange",
    "ZS=F":     "SOYBEANS -- Chicago Board of Trade",
    "CL=F":     "WTI-PHYSICAL -- New York Mercantile Exchange",
    "GC=F":     "GOLD -- Commodity Exchange Inc."
}

# Seasonal profitability “ground truth”
ProfitableSeasonalMap: Dict[str, Dict] = {
    "Indices": {
        "S&P 500": {
            "Jan":"⚪","Feb":"⚪","Mar":"⚪","Apr":"✅","May":"⚪","Jun":"⚪",
            "Jul":"⚪","Aug":"⚪","Sep":"❌","Oct":"✅","Nov":"✅","Dec":"✅"
        },
        "DAX": {  
            "Jan":"⚪","Feb":"⚪","Mar":"⚪","Apr":"✅","May":"⚪","Jun":"⚪",
            "Jul":"⚪","Aug":"⚪","Sep":"❌","Oct":"✅","Nov":"✅","Dec":"✅"
        }
    },
    "Forex": {
        "EUR/USD": {
            "Jan":"✅","Feb":"⚪","Mar":"⚪","Apr":"⚪","May":"⚪","Jun":"⚪",
            "Jul":"⚪","Aug":"⚪","Sep":"✅","Oct":"⚪","Nov":"⚪","Dec":"⚪"
        },
        "USD/JPY": {
            "Jan":"⚪","Feb":"⚪","Mar":"⚪","Apr":"⚪","May":"⚪","Jun":"⚪",
            "Jul":"✅","Aug":"⚪","Sep":"⚪","Oct":"⚪","Nov":"⚪","Dec":"⚪"
        }
    },
    "Commodities": {
        "Agricultural (Soybeans)": {
            "Jan":"⚪","Feb":"⚪","Mar":"⚪","Apr":"⚪","May":"⚪","Jun":"⚪",
            "Jul":"✅","Aug":"⚪","Sep":"⚪","Oct":"⚪","Nov":"⚪","Dec":"⚪"
        },
        "Energy (Crude Oil)": {
            "Jan":"✅","Feb":"⚪","Mar":"⚪","Apr":"⚪","May":"⚪","Jun":"✅",
            "Jul":"✅","Aug":"⚪","Sep":"⚪","Oct":"⚪","Nov":"✅","Dec":"✅"
        },
        "Metals (Gold)": {
            "Jan":"⚪","Feb":"⚪","Mar":"⚪","Apr":"⚪","May":"⚪","Jun":"⚪",
            "Jul":"⚪","Aug":"⚪","Sep":"⚪","Oct":"✅","Nov":"✅","Dec":"✅"
        }
    }
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

# ── CACHING HELPERS ──────────────────────────────────────────────────────────
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
    df["date"]      = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["timestamp"] = df["date"]
    df["volume"]    = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    df["return_pct"] = df["close"].pct_change()*100
    time.sleep(YH_SLEEP)
    return df

# ── COT FETCHING ─────────────────────────────────────────────────────────────
SODAPY_APP_TOKEN = "WSCaavlIcDgtLVZbJA1FKkq40"
client = Socrata("publicreporting.cftc.gov", SODAPY_APP_TOKEN)

@st.cache_data(show_spinner=False, ttl=24*3600)
def fetch_cot_data(cot_name: str, start_date: str, end_date: str) -> pd.DataFrame:
    if not cot_name:
        return pd.DataFrame()

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
    keep = [c for c in ["report_date_as_yyyy_mm_dd","commercial_long_all","commercial_short_all","open_interest_all"] if c in df.columns]
    df = df[keep]

    # Ensure timestamp column
    if "report_date_as_yyyy_mm_dd" in df.columns:
        df["timestamp"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
    else:
        df["timestamp"] = pd.Timestamp.today()

    if {"commercial_long_all","commercial_short_all"} <= set(df.columns):
        df["commercial_net"] = df["commercial_long_all"] - df["commercial_short_all"]

    # Forward-fill daily
    df_daily = (
        df.set_index("timestamp")
          .sort_index()
          .reindex(pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="D"))
          .ffill()
          .reset_index()
          .rename(columns={"index":"timestamp"})
    )

    return df_daily.sort_values("timestamp").reset_index(drop=True)

# ── METRIC BUILDERS ──────────────────────────────────────────────────────────
def add_rvol(df: pd.DataFrame, window: int) -> pd.DataFrame:
    out = df.copy()
    out["rvol"] = out["volume"] / out["volume"].rolling(window).mean().replace(0, np.nan)
    return out

# ── FIXED MERGE FUNCTION ─────────────────────────────────────────────────────
def merge_cot_price(cot: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    if price.empty or cot.empty:
        return pd.DataFrame()

    # Ensure 'timestamp' exists
    if "timestamp" not in cot.columns:
        if "report_date_as_yyyy_mm_dd" in cot.columns:
            cot["timestamp"] = pd.to_datetime(cot["report_date_as_yyyy_mm_dd"])
        else:
            cot["timestamp"] = pd.Timestamp.today()

    # Align Friday release to trading day
    cot = cot.copy()
    cot["cot_date"] = cot["timestamp"] - pd.to_timedelta(cot["timestamp"].dt.weekday - 4, unit="D")

    keep_cols = [c for c in ["cot_date","commercial_long_all","commercial_short_all"] if c in cot.columns]
    cot = cot[keep_cols]

    merged = pd.merge_asof(
        price.sort_values("date"),
        cot.sort_values("cot_date"),
        left_on="date", right_on="cot_date",
        direction="backward"
    ).drop(columns="cot_date", errors="ignore")

    tot = (merged.get("commercial_long_all", 0) + merged.get("commercial_short_all", 0)).replace(0, np.nan)
    if "commercial_long_all" in merged.columns:
        merged["cot_long_norm"] = merged["commercial_long_all"] / tot
    if "commercial_short_all" in merged.columns:
        merged["cot_short_norm"] = merged["commercial_short_all"] / tot

    return merged

def add_health_gauge(df: pd.DataFrame, weights: Dict[str, float] = {"rvol":0.5, "cot_long":0.3, "cot_short":0.2}) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["health_gauge"] = (
        weights["rvol"]     * out.get("rvol",1).fillna(1) +
        weights["cot_long"] * out.get("cot_long_norm",0).fillna(0) -
        weights["cot_short"]* out.get("cot_short_norm",0).fillna(0)
    )
    return out

# ── FETCH DATA ───────────────────────────────────────────────────────────────
tickers = ASSET_LEADERS[category_selected]

with st.spinner("Downloading & crunching data …"):
    # Yahoo – parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futs = {pool.submit(fetch_price_history, t): t for t in tickers}
        price_data = {t: f.result() for f, t in ((f, futs[f]) for f in futs)}

    # COT – sequential
    cot_data = {t: fetch_cot_data(COT_ASSET_NAMES.get(t, ""), START_DATE, END_DATE)
                for t in tickers}

    merged_data = {}
    for t in tickers:
        price  = add_rvol(price_data[t], rvol_window)
        merged = add_health_gauge(merge_cot_price(cot_data[t], price))
        merged_data[t] = merged

# ── VISUALISATIONS ───────────────────────────────────────────────────────────
st.markdown("## Rolling Volatility")
for t in tickers:
    df = merged_data[t]
    if df.empty:
        st.warning(f"No data for {TICKER_TO_NAME[t]}")
        continue
    fig = px.line(df, x="date", y="rvol",
                  title=f"{TICKER_TO_NAME[t]} — RVol ({rvol_window}-day)",
                  template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# ── REST OF THE CODE PRESERVED ───────────────────────────────────────────────
# All your existing pip & return tree, return_accuracy_tree, JSON exports etc.
# ... (code unchanged)





# ── JSON EXPORT ──────────────────────────────────────────────────────────────
st.markdown("## Export Data")
for t in tickers:
    df = merged_data[t]
    if df.empty:
        continue
    json_str = df.to_json(orient="records", date_format="iso")
    st.download_button(
        label=f"Download {TICKER_TO_NAME[t]} JSON",
        data=json_str,
        file_name=f"{TICKER_TO_NAME[t].replace(' ','_')}_data.json",
        mime="application/json"
    )

# ── PIP & RETURN TREE ────────────────────────────────────────────────────────
st.markdown("## Price vs COT Return Tree")
for t in tickers:
    df = merged_data[t]
    if df.empty or "return_pct" not in df.columns:
        st.warning(f"No return data for {TICKER_TO_NAME[t]}")
        continue

    df_tree = df.copy()
    df_tree["cum_return"] = (1 + df_tree["return_pct"]/100).cumprod() - 1
    fig_tree = px.line(df_tree, x="date", y="cum_return",
                       title=f"{TICKER_TO_NAME[t]} Cumulative Return",
                       template="plotly_white")
    st.plotly_chart(fig_tree, use_container_width=True)

# ── RETURN ACCURACY TREE ─────────────────────────────────────────────────────
st.markdown("## Health Gauge Accuracy")
for t in tickers:
    df = merged_data[t]
    if df.empty or "health_gauge" not in df.columns:
        st.warning(f"No health gauge data for {TICKER_TO_NAME[t]}")
        continue

    df_accuracy = df.copy()
    df_accuracy["sign_match"] = np.sign(df_accuracy["return_pct"].fillna(0)) == np.sign(df_accuracy["health_gauge"].fillna(0))
    df_accuracy["accuracy_cum"] = df_accuracy["sign_match"].expanding().mean()

    fig_acc = px.line(df_accuracy, x="date", y="accuracy_cum",
                      title=f"{TICKER_TO_NAME[t]} Health Gauge Accuracy",
                      template="plotly_white")
    st.plotly_chart(fig_acc, use_container_width=True)

# ── SEASONAL PROFITABILITY ───────────────────────────────────────────────────
st.markdown("## Seasonal Profitability")
seasonal_data = ProfitableSeasonalMap.get(category_selected, {})

for name, months in seasonal_data.items():
    st.markdown(f"### {name}")
    month_str = " | ".join(f"{m}: {emoji}" for m, emoji in months.items())
    st.text(month_str)

# ── END OF SCRIPT ────────────────────────────────────────────────────────────
st.markdown("### App loaded successfully ✅")