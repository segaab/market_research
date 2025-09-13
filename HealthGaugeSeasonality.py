#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Multi-Asset Health Gauge – Streamlit application
# -----------------------------------------------------------------------------
import json
import time
import calendar
from datetime import datetime, timedelta
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor

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
    df["return_pct"] = df["close"].pct_change() * 100
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

# ── METRICS & ACC/DIST ───────────────────────────────────────────────────────
def add_rvol(df: pd.DataFrame, window: int) -> pd.DataFrame:
    out = df.copy()
    if "volume" in out.columns:
        out["rvol"] = out["volume"] / out["volume"].rolling(window).mean().replace(0,np.nan)
    else:
        out["rvol"] = np.nan
    return out

def calculate_acc_dist_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return"] = df["close"].pct_change().fillna(0)
    df["acc_dist"] = 3  # neutral
    if "rvol" not in df.columns:
        df["rvol"] = 1
    rvol_75 = df["rvol"].quantile(0.75)
    df.loc[(df["return"] >= 0.02) & (df["rvol"] >= rvol_75), "acc_dist"] = 5
    df.loc[(df["return"] >= 0.01) & (df["return"] < 0.02) & (df["rvol"] >= rvol_75*0.7), "acc_dist"] = 4
    df.loc[(df["return"] <= -0.02) & (df["rvol"] >= rvol_75), "acc_dist"] = 0
    df.loc[(df["return"] <= -0.01) & (df["return"] > -0.02) & (df["rvol"] >= rvol_75*0.7), "acc_dist"] = 1
    return df

def calculate_daily_health_gauge(price_df: pd.DataFrame, cot_df: pd.DataFrame) -> float:
    if price_df.empty or cot_df.empty:
        return np.nan
    oi_series = price_df.get("open_interest_all", pd.Series()).dropna()
    oi_score = 0.0 if oi_series.empty else float((oi_series.iloc[-1] - oi_series.min()) / (oi_series.max() - oi_series.min() + 1e-9))
    st_score = lt_score = 0.0
    if "commercial_net" in cot_df.columns and not cot_df.empty:
        st_score = float((cot_df["commercial_net"].iloc[-1] - cot_df["commercial_net"].min()) /
                         (cot_df["commercial_net"].max() - cot_df["commercial_net"].min() + 1e-9))
    if "non_commercial_net" in cot_df.columns and not cot_df.empty:
        lt_score = float((cot_df["non_commercial_net"].iloc[-1] - cot_df["non_commercial_net"].min()) /
                         (cot_df["non_commercial_net"].max() - cot_df["non_commercial_net"].min() + 1e-9))
    cot_score = 0.4 * st_score + 0.6 * lt_score
    acc_mean = price_df["acc_dist"].mean() if "acc_dist" in price_df.columns else 0
    return (0.25*oi_score + 0.35*cot_score + 0.4*acc_mean)*10

def process_daily_tree(price_df: pd.DataFrame, cot_df: pd.DataFrame) -> list:
    daily_list = []
    for day, group in price_df.groupby(price_df["timestamp"].dt.date):
        pip_range_24h = (group["high"] - group["low"]).tolist()
        sd_pip_24h = group["high"].sub(group["low"]).rolling(1).std().fillna(0).tolist()
        daily_mean = (group["high"] - group["low"]).mean()
        daily_median = (group["high"] - group["low"]).median()
        daily_health = calculate_daily_health_gauge(group, cot_df)
        daily_list.append({
            "date": str(day),
            "month": day.month,
            "pip_range_24h": pip_range_24h,
            "daily_mean_pip": daily_mean,
            "daily_median_pip": daily_median,
            "sd_pip_24h": sd_pip_24h,
            "daily_health_gauge": daily_health
        })
    return daily_list

def process_monthly_tree(daily_list: list) -> list:
    df = pd.DataFrame(daily_list)
    if df.empty:
        return []
    df["date"] = pd.to_datetime(df["date"])
    monthly_list = []
    for month, group in df.groupby(df["date"].dt.month):
        daily_returns = group["daily_mean_pip"].pct_change().fillna(0)
        monthly_list.append({
            "month": month,
            "mean_return": daily_returns.mean(),
            "median_return": daily_returns.median(),
            "sd_return": daily_returns.std(),
            "min_return": daily_returns.min(),
            "max_return": daily_returns.max(),
            "accuracy_percentage": (daily_returns>0).mean()*100
        })
    return monthly_list

def fetch_process_asset(asset_name: str):
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_price = executor.submit(fetch_price_history, asset_name)
        future_cot = executor.submit(fetch_cot_data, COT_ASSET_NAMES.get(asset_name,""), START_DATE, END_DATE)
        price_df = future_price.result()
        cot_df = future_cot.result()
    
    if price_df.empty or cot_df.empty:
        return None, None

    price_df = add_rvol(price_df, rvol_window)
    price_df = calculate_acc_dist_score(price_df)

    daily_tree = process_daily_tree(price_df, cot_df)
    monthly_tree = process_monthly_tree(daily_tree)
    return daily_tree, monthly_tree

# ── PROCESS CATEGORY ───────────────────────────────────────────────────────
def process_category(category: str):
    tickers = ASSET_LEADERS.get(category, [])
    daily_json, monthly_json = {}, {}

    for t in tickers:
        st.info(f"Processing {TICKER_TO_NAME[t]}...")
        daily_tree, monthly_tree = fetch_process_asset(t)

        if daily_tree is None or monthly_tree is None:
            st.warning(f"No data for {TICKER_TO_NAME[t]}")
            continue

        daily_json[t] = daily_tree
        monthly_json[t] = monthly_tree

        # ── VISUALIZATION ──────────────────────────────────────────────
        df_daily = pd.DataFrame(daily_tree)
        if not df_daily.empty:
            fig = px.line(
                df_daily,
                x="date",
                y="daily_health_gauge",
                title=f"Health Gauge - {TICKER_TO_NAME[t]}"
            )
            st.plotly_chart(fig, use_container_width=True)

        df_monthly = pd.DataFrame(monthly_tree)
        if not df_monthly.empty:
            fig2 = px.bar(
                df_monthly,
                x="month",
                y="mean_return",
                title=f"Monthly Returns - {TICKER_TO_NAME[t]}"
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── JSON EXPORT ───────────────────────────────────────────────────────
    if daily_json:
        st.download_button(
            label="Download Daily JSON Tree",
            data=json.dumps({category: daily_json}, indent=2, default=str),
            file_name=f"daily_tree_{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    if monthly_json:
        st.download_button(
            label="Download Monthly JSON Tree",
            data=json.dumps({category: monthly_json}, indent=2, default=str),
            file_name=f"monthly_tree_{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )