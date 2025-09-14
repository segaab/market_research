#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Multi-Asset Health Gauge -- Streamlit application (Daily & Monthly JSON export)
# -----------------------------------------------------------------------------
import json
import time
import calendar
from datetime import datetime, timedelta
from typing import Dict, List, DefaultDict

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sodapy import Socrata
from yahooquery import Ticker
from collections import defaultdict

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

# ── FETCHING HELPERS ─────────────────────────────────────────────────────────
SODAPY_APP_TOKEN = "WSCaavlIcDgtLVZbJA1FKkq40"

@st.cache_resource(show_spinner=False)
def _yahoo_session(symbols: list) -> Ticker:
    return Ticker(symbols)

@st.cache_resource(show_spinner=False)
def _cot_client() -> Socrata:
    return Socrata("publicreporting.cftc.gov", SODAPY_APP_TOKEN, timeout=120)

@st.cache_data(show_spinner=False, ttl=60*60)
def fetch_price_history(ticker: str) -> pd.DataFrame:
    yq = _yahoo_session([ticker])
    df = yq.history(start=START_DATE, end=END_DATE)
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    df["return_pct"] = df["close"].pct_change() * 100
    df["pip_range"] = df["high"] - df["low"]      # ← used later for pip stats
    time.sleep(YH_SLEEP)
    return df

@st.cache_data(show_spinner=False, ttl=24*3600)
def fetch_cot_data(cot_name: str, start_date: str, end_date: str) -> pd.DataFrame:
    if not cot_name:
        return pd.DataFrame()
    client = _cot_client()
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

# ── METRICS (unchanged) ──────────────────────────────────────────────────────
def add_rvol(df: pd.DataFrame, window: int) -> pd.DataFrame:
    out = df.copy()
    if "volume" in out.columns:
        out["rvol"] = out["volume"] / out["volume"].rolling(window).mean().replace(0, np.nan)
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
    
    tot = (merged.get("commercial_long_all",0) + merged.get("commercial_short_all",0)).replace(0, np.nan)
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

# ── MONTHLY RETURNS & PROFITABILITY (unchanged helpers) ──────────────────────
def compute_monthly_returns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df["month"] = df["timestamp"].dt.to_period("M")
    monthly_returns = df.groupby(["month"])["return_pct"].sum().reset_index()
    monthly_returns["month_name"] = monthly_returns["month"].dt.strftime("%B")
    return monthly_returns

def assign_profit_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    q33,q77 = df["return_pct"].quantile([0.33,0.77])
    df["profit_label"] = np.where(df["return_pct"]>=q77,"Profitable",
                          np.where(df["return_pct"]<=q33,"Unprofitable","Average"))
    return df

def calculate_accuracy(df: pd.DataFrame, asset_name: str, category: str) -> float:
    if df.empty:
        return 0.0
    top_key = "Commodities" if category in ("Agricultural","Energy","Metals") else category
    monthly_map = ProfitableSeasonalMap.get(top_key,{}).get(asset_name,{})
    df["expected"] = df["month"].dt.month.map(lambda m: monthly_map.get(calendar.month_abbr[m],"⚪"))
    df["actual_hit"] = df.apply(lambda x: 1 if ((x["expected"]=="✅" and x["profit_label"]=="Profitable") or (x["expected"]=="❌" and x["profit_label"]=="Unprofitable")) else 0, axis=1)
    return round(df["actual_hit"].mean()*100,2)



# ── PROCESS CATEGORY ─────────────────────────────────────────────────────────
def process_category(category: str):
    tickers = ASSET_LEADERS.get(category, [])
    merged_data            : Dict[str, pd.DataFrame] = {}
    monthly_distribution   : Dict[str, pd.DataFrame] = {}
    accuracy_stats         : Dict[str, float]        = {}

    # ── helpers used to build new JSON trees ────────────────────────────────
    daily_pip_registry : DefaultDict[pd.Timestamp, List[float]] = defaultdict(list)
    daily_hg_registry  : DefaultDict[pd.Timestamp, List[float]] = defaultdict(list)
    monthly_return_registry : DefaultDict[int, List[float]]     = defaultdict(list)
    monthly_accuracy_registry: DefaultDict[int, List[float]]    = defaultdict(list)

    # ── LOOP THROUGH TICKERS ────────────────────────────────────────────────
    for t in tickers:
        price = fetch_price_history(t)
        cot   = fetch_cot_data(COT_ASSET_NAMES.get(t, ""), START_DATE, END_DATE)

        price      = add_rvol(price, rvol_window)
        health_df  = add_health_gauge(merge_cot_price(cot, price))
        merged_data[t] = health_df

        # --------------- DAILY AGGREGATION ----------------------------------
        if not health_df.empty:
            for idx, row in health_df.iterrows():
                ts = row["timestamp"].normalize()
                # pip_range already present in price DF – align by timestamp
                pr = row.get("pip_range", np.nan)
                if not np.isnan(pr):
                    daily_pip_registry[ts].append(round(pr, 4))
                daily_hg_registry[ts].append(round(row["health_gauge"], 4))

        # --------------- MONTHLY AGGREGATION --------------------------------
        monthly_df = compute_monthly_returns(health_df)
        monthly_df = assign_profit_labels(monthly_df)
        monthly_distribution[t] = monthly_df
        accuracy_stats[t] = calculate_accuracy(monthly_df, TICKER_TO_NAME[t], category)

        if not monthly_df.empty:
            for _, r in monthly_df.iterrows():
                m_num = r["month"].month          # 1-12
                monthly_return_registry[m_num].append(round(r["return_pct"], 4))
                monthly_accuracy_registry[m_num].append(accuracy_stats[t])

    # ── BUILD DAILY JSON TREE ────────────────────────────────────────────────
    daily_json_tree = []
    for dt in sorted(daily_pip_registry.keys()):
        pip_list = daily_pip_registry[dt]
        hg_list  = daily_hg_registry[dt]
        if not pip_list or not hg_list:
            continue
        record = {
            "date"            : dt.strftime("%Y-%m-%d"),
            "month"           : dt.month,
            "pip_range_24h"   : pip_list,                                       # raw list
            "daily_mean_pip"  : round(float(np.mean(pip_list)), 4),
            "daily_median_pip": round(float(np.median(pip_list)), 4),
            "sd_pip_24h"      : round(float(np.std(pip_list, ddof=0)), 4),
            "daily_health_gauge": round(float(np.mean(hg_list)), 4)             # central measure
        }
        daily_json_tree.append(record)

    # ── BUILD MONTHLY JSON TREE ──────────────────────────────────────────────
    monthly_json_tree = []
    for m in range(1, 13):
        if m not in monthly_return_registry:
            continue
        ret_list = monthly_return_registry[m]
        acc_list = monthly_accuracy_registry[m]
        monthly_json_tree.append({
            "month"          : calendar.month_name[m],
            "monthly_return" : ret_list,
            "accuracy_pct"   : round(float(np.mean(acc_list)), 2) if acc_list else 0.0,
            "mean_return"    : round(float(np.mean(ret_list)), 4),
            "median_return"  : round(float(np.median(ret_list)), 4),
            "sd_return"      : round(float(np.std(ret_list, ddof=0)), 4)
        })

    # ── DOWNLOAD BUTTONS ─────────────────────────────────────────────────────
    st.download_button(
        label = "Download Daily Health Gauge + Pip Stats (JSON)",
        data  = json.dumps({category: daily_json_tree}, indent=2),
        file_name = f"daily_health_pip_{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

    st.download_button(
        label = "Download Monthly Stats (JSON)",
        data  = json.dumps({category: monthly_json_tree}, indent=2),
        file_name = f"monthly_stats_{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

    # ── (existing visualisations – unchanged) ───────────────────────────────
    for t in tickers:
        df_plot = merged_data.get(t, pd.DataFrame())
        if df_plot.empty:
            continue
        fig = px.line(df_plot, x="timestamp", y="health_gauge",
                      title=f"Health Gauge - {TICKER_TO_NAME[t]}")
        st.plotly_chart(fig, use_container_width=True)

        m_df = monthly_distribution[t]
        if not m_df.empty:
            fig2 = px.bar(m_df, x=m_df["month"].dt.strftime("%Y-%m"),
                          y="return_pct", color="profit_label",
                          title=f"Monthly Return Distribution - {TICKER_TO_NAME[t]}")
            st.plotly_chart(fig2, use_container_width=True)
        st.write(f"Profitability Accuracy ({TICKER_TO_NAME[t]}): {accuracy_stats[t]}%")

# ── RUN ──────────────────────────────────────────────────────────────────────
process_category(category_selected)
st.write("### Script Version: Daily & Monthly JSON export enabled")
