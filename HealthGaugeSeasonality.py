#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Multi-Asset Health Gauge  –  Streamlit application
# -----------------------------------------------------------------------------
import json
import time
import calendar
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
COT_SLEEP     = 0.35         # seconds between paged COT calls
YH_SLEEP      = 0.20         # seconds between Yahoo sub-calls

SODAPY_APP_TOKEN = "WSCaavlIcDgtLVZbJA1FKkq40"

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
    """Download OHLCV; add timestamp column."""
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

@st.cache_data(show_spinner=False, ttl=24*3600)
def fetch_cot_data(cot_name: str, start_date: str = START_DATE, end_date: str = END_DATE) -> pd.DataFrame:
    """Weekly COT data → forward-filled daily; timestamp column added."""
    if not cot_name:
        return pd.DataFrame()

    client = Socrata(
        "publicreporting.cftc.gov",
        SODAPY_APP_TOKEN,
        timeout=30
    )

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
    keep = [
        "report_date_as_yyyy_mm_dd",
        "commercial_long_all",
        "commercial_short_all",
        "open_interest_all",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep]
    df["report_date_as_yyyy_mm_dd"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
    df["timestamp"] = df["report_date_as_yyyy_mm_dd"]

    if {"commercial_long_all", "commercial_short_all"} <= set(df.columns):
        df["commercial_net"] = df["commercial_long_all"] - df["commercial_short_all"]

    # ── forward-fill to daily frequency ───────────────────────────────────────
    df_daily = (
        df.set_index("timestamp")
          .sort_index()
          .reindex(pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="D"))
          .ffill()
          .reset_index()
          .rename(columns={"index": "timestamp"})
    )

    return df_daily.sort_values("timestamp").reset_index(drop=True)

# ── METRIC BUILDERS ──────────────────────────────────────────────────────────
def add_rvol(df: pd.DataFrame, window: int) -> pd.DataFrame:
    out = df.copy()
    out["rvol"] = out["volume"] / out["volume"].rolling(window).mean().replace(0, np.nan)
    return out

def merge_cot_price(cot: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    if price.empty:
        return pd.DataFrame()

    cot = cot.copy()
    cot["cot_date"] = cot["timestamp"] - pd.to_timedelta(cot["timestamp"].dt.weekday - 4, unit="D")
    cot = cot[["cot_date", "commercial_long_all", "commercial_short_all"]]

    merged = pd.merge_asof(
        price.sort_values("date"),
        cot.sort_values("cot_date"),
        left_on="date", right_on="cot_date", direction="backward"
    ).drop(columns="cot_date")

    tot = (merged["commercial_long_all"] + merged["commercial_short_all"]).replace(0, np.nan)
    merged["cot_long_norm"]  = merged["commercial_long_all"]  / tot
    merged["cot_short_norm"] = merged["commercial_short_all"] / tot
    return merged

def add_health_gauge(df: pd.DataFrame, w: Dict[str,float] = {"rvol":0.5,"cot_long":0.3,"cot_short":0.2}) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["health_gauge"] = (
        w["rvol"]     * out["rvol"].fillna(1) +
        w["cot_long"] * out["cot_long_norm"].fillna(0) -
        w["cot_short"]* out["cot_short_norm"].fillna(0)
    )
    return out

# ── FETCH DATA ───────────────────────────────────────────────────────────────
tickers = ASSET_LEADERS[category_selected]

with st.spinner("Downloading & crunching data …"):
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futs = {pool.submit(fetch_price_history, t): t for t in tickers}
        price_data = {t: f.result() for f, t in ((f, futs[f]) for f in futs)}

    cot_data = {t: fetch_cot_data(COT_ASSET_NAMES.get(t, ""), START_DATE, END_DATE) for t in tickers}

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

st.markdown("## Return Distribution")
df_results = pd.concat(
    [d.loc[d["return_pct"].notna(), ["return_pct"]].assign(asset=TICKER_TO_NAME[t])
     for t, d in merged_data.items() if not d.empty],
    ignore_index=True
)
if df_results.empty:
    st.info("No return data available.")
else:
    fig = px.histogram(df_results, x="return_pct", nbins=20,
                       color_discrete_sequence=["#3366CC"],
                       title="Daily % Return Distribution")
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("## Health Gauge")
for t in tickers:
    df = merged_data[t]
    if df.empty:
        continue
    fig = px.line(df, x="date", y="health_gauge",
                  title=f"{TICKER_TO_NAME[t]} — Health Gauge",
                  template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# ── JSON EXPORTS ─────────────────────────────────────────────────────────────
export_cols = [
    "date","open","high","low","close","volume",
    "return_pct","rvol",
    "commercial_long_all","commercial_short_all",
    "cot_long_norm","cot_short_norm","health_gauge"
]

payload_merged = {
    t: d[export_cols].round(6).fillna(None).to_dict(orient="records")
    for t,d in merged_data.items() if not d.empty
}
st.download_button(
    "Download JSON (Merged Data)",
    json.dumps(payload_merged, indent=2, default=str),
    file_name=f"health_gauge_merged_{category_selected.lower()}.json",
    mime="application/json"
)