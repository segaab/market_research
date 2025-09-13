#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Multi-Asset Health Gauge – Streamlit application
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
import concurrent.futures

# ── APP CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Multi-Asset Health Gauge", layout="wide")

START_DATE = "2013-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

COT_PAGE_SIZE = 15_000
COT_SLEEP = 0.35
YH_SLEEP = 0.20

# ── MAPPINGS ─────────────────────────────────────────────────────────────────
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

ProfitableSeasonalMap: Dict[str, Dict] = {
    "Indices": {
        "S&P 500": {
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
    df["timestamp"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    df["return_pct"] = df["close"].pct_change()*100
    time.sleep(YH_SLEEP)
    return df

# ── COT FETCH ───────────────────────────────────────────────────────────────
SODAPY_APP_TOKEN = "WSCaavlIcDgtLVZbJA1FKkq40"
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
    keep = ["report_date_as_yyyy_mm_dd", "commercial_long_all", "commercial_short_all", "open_interest_all"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep]
    df["timestamp"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
    if {"commercial_long_all", "commercial_short_all"} <= set(df.columns):
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

# ── METRICS ────────────────────────────────────────────────────────────────
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
        price.sort_values("timestamp"),
        cot.sort_values("cot_date"),
        left_on="timestamp", right_on="cot_date", direction="backward"
    ).drop(columns="cot_date")
    tot = (merged["commercial_long_all"] + merged["commercial_short_all"]).replace(0, np.nan)
    merged["cot_long_norm"] = merged["commercial_long_all"]/tot
    merged["cot_short_norm"] = merged["commercial_short_all"]/tot
    return merged

def add_health_gauge(df: pd.DataFrame, weights: Dict[str,float] = {"rvol":0.5, "cot_long":0.3, "cot_short":0.2}) -> pd.DataFrame:
    out = df.copy()
    out["health_gauge"] = (
        weights["rvol"]*out["rvol"].fillna(1) +
        weights["cot_long"]*out["cot_long_norm"].fillna(0) -
        weights["cot_short"]*out["cot_short_norm"].fillna(0)
    )
    return out

# ── MONTHLY RETURNS & PROFITABILITY ──────────────────────────────────────────
def compute_monthly_returns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df["month"] = df["timestamp"].dt.to_period("M")
    monthly_returns = df.groupby("month")["return_pct"].sum().reset_index()
    monthly_returns["month"] = monthly_returns["month"].dt.to_timestamp()
    return monthly_returns

def assign_profit_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    q33, q77 = df["return_pct"].quantile([0.33, 0.77])
    df["profit_label"] = np.where(df["return_pct"]>=q77, "Profitable",
                          np.where(df["return_pct"]<=q33, "Unprofitable","Average"))
    return df

def calculate_accuracy(df: pd.DataFrame, asset_name: str, category: str) -> float:
    if df.empty:
        return 0.0
    top_key = "Commodities" if category in ("Agricultural","Energy","Metals") else category
    monthly_map = ProfitableSeasonalMap.get(top_key, {}).get(asset_name, {})
    df["expected"] = df["month"].dt.month.map(lambda m: monthly_map.get(calendar.month_abbr[m],"⚪"))
    df["actual_hit"] = df.apply(lambda x: 1 if ((x["expected"]=="✅" and x["profit_label"]=="Profitable") or (x["expected"]=="❌" and x["profit_label"]=="Unprofitable")) else 0, axis=1)
    return round(df["actual_hit"].mean()*100,2)

# ── FETCH, MERGE & CALCULATE ───────────────────────────────────────────────
tickers = ASSET_LEADERS[category_selected]
merged_data = {}
monthly_distribution = {}
accuracy_stats = {}

for t in tickers:
    price = fetch_price_history(t)
    cot = fetch_cot_data(COT_ASSET_NAMES.get(t,""), START_DATE, END_DATE)
    price = add_rvol(price, rvol_window)
    merged = add_health_gauge(merge_cot_price(cot, price))
    merged_data[t] = merged

    monthly = compute_monthly_returns(merged)
    monthly = assign_profit_labels(monthly)
    monthly_distribution[t] = monthly
    accuracy_stats[t] = calculate_accuracy(monthly, TICKER_TO_NAME[t], category_selected)




# ── JSON EXPORT ─────────────────────────────────────────────────────────────
export_cols = ["timestamp","close","rvol","cot_long_norm","cot_short_norm","health_gauge","return_pct","profit_label"]
payload_merged = {
    t: d[export_cols].round(6).fillna("").to_dict(orient="records")
    for t,d in merged_data.items() if not d.empty
}

monthly_payload = {
    t: monthly_distribution[t].round(6).fillna("").to_dict(orient="records")
    for t in monthly_distribution if not monthly_distribution[t].empty
}

accuracy_payload = {t: {"accuracy_pct": accuracy_stats[t]} for t in accuracy_stats}

full_json_payload = {
    "merged_data": payload_merged,
    "monthly_returns": monthly_payload,
    "accuracy_distribution": accuracy_payload
}

st.download_button(
    label="Download Health Gauge JSON",
    data=json.dumps(full_json_payload, indent=2, default=str),
    file_name=f"health_gauge_data_{datetime.now().strftime('%Y%m%d')}.json",
    mime="application/json"
)

# ── VISUALIZATION ──────────────────────────────────────────────────────────
for t in tickers:
    if t not in merged_data or merged_data[t].empty:
        continue
    df_plot = merged_data[t]
    fig = px.line(df_plot, x="timestamp", y="health_gauge", title=f"Health Gauge - {TICKER_TO_NAME[t]}")
    st.plotly_chart(fig, use_container_width=True)

    monthly_df = monthly_distribution[t]
    if not monthly_df.empty:
        fig2 = px.bar(
            monthly_df,
            x="month",
            y="return_pct",
            color="profit_label",
            title=f"Monthly Return Distribution - {TICKER_TO_NAME[t]}"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.write(f"Profitability Accuracy: {accuracy_stats[t]}%")

st.write("### Script Version: Monthly Return Distribution + Pip Stats + JSON Export")