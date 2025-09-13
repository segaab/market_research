import pandas as pd
import numpy as np
import streamlit as st
import datetime
from concurrent.futures import ThreadPoolExecutor
from yahooquery import Ticker
from sodapy import Socrata
from typing import Dict, List   # ✅ Fix for NameError
import json

# ----------------------------------
# Configuration
# ----------------------------------
START_DATE = "2000-01-01"
END_DATE = datetime.date.today().strftime("%Y-%m-%d")

COT_ASSET_NAMES = {
    # Indices
    "^GSPC": "S&P 500 – Chicago Mercantile Exchange",   # S&P 500
    "^GDAXI": None,  # DAX not directly reported in CFTC COT

    # Forex
    "EURUSD=X": "EURO FX – Chicago Mercantile Exchange",
    "USDJPY=X": "JAPANESE YEN – Chicago Mercantile Exchange",

    # Commodities
    "ZS=F": "SOYBEANS – Chicago Board of Trade",       
    "CL=F": "WTI-PHYSICAL – New York Mercantile Exchange",  
    "GC=F": "GOLD – Commodity Exchange Inc."          
}

MONTHS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

# ----------------------------------
# Socrata client (CFTC)
# ----------------------------------
client = Socrata("publicreporting.cftc.gov", None)

# ----------------------------------
# Price Fetching
# ----------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_history(ticker: str) -> pd.DataFrame:
    """Fetch OHLCV history from Yahoo Finance via yahooquery."""
    try:
        yq = Ticker(ticker)
        df = yq.history(start=START_DATE, end=END_DATE)
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df.rename(columns={"symbol": "ticker"}, inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching price data for {ticker}: {e}")
        return pd.DataFrame()

# ----------------------------------
# COT Fetching
# ----------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_cot_data(market_name: str) -> pd.DataFrame:
    """Fetch COT data for a given market name."""
    try:
        if not market_name:
            return pd.DataFrame()
        result = client.get("6dca-aqww", where=f'market_and_exchange_names="{market_name}"')
        df = pd.DataFrame.from_records(result)
        if df.empty:
            return pd.DataFrame()
        df["report_date_as_yyyy_mm_dd"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
        return df
    except Exception as e:
        st.error(f"Error fetching COT data for {market_name}: {e}")
        return pd.DataFrame()


# ----------------------------------
# Merge COT + Price
# ----------------------------------
def merge_cot_price(cot: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    """Align COT data with price history by nearest Friday reporting."""
    if cot.empty or price.empty:
        return pd.DataFrame()

    cot = cot.copy()
    cot["cot_date"] = cot["report_date_as_yyyy_mm_dd"] - pd.to_timedelta(
        cot["report_date_as_yyyy_mm_dd"].dt.weekday - 4, unit="D"
    )
    cot = cot[[
        "cot_date",
        "commercial_long_all", "commercial_short_all",
        "noncomm_long_all", "noncomm_short_all"
    ]]

    price = price.copy()
    price["date"] = pd.to_datetime(price["date"])

    out = pd.merge_asof(
        price.sort_values("date"),
        cot.sort_values("cot_date"),
        left_on="date",
        right_on="cot_date",
        direction="backward"
    )
    return out

# ----------------------------------
# Health Gauge
# ----------------------------------
def add_health_gauge(df: pd.DataFrame,
                     weights: Dict[str, float] = {"rvol": .5, "cot_long": .25, "cot_short": .25}) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # Rolling volatility (30d default)
    df["rvol"] = df["close"].pct_change().rolling(30).std() * (252**0.5)

    # Net positions (normalized)
    df["cot_long"] = df["commercial_long_all"].astype(float) - df["commercial_short_all"].astype(float)
    df["cot_short"] = df["noncomm_long_all"].astype(float) - df["noncomm_short_all"].astype(float)

    # Health score (weighted sum)
    df["health_gauge"] = (
        weights["rvol"] * df["rvol"].fillna(0) +
        weights["cot_long"] * df["cot_long"].fillna(0) -
        weights["cot_short"] * df["cot_short"].fillna(0)
    )

    return df

# ----------------------------------
# Pip & Return Tree
# ----------------------------------
def pip_and_return_tree_daily_pips(
    data: Dict[str, pd.DataFrame], category: str, normalize: bool = True
) -> Dict[str, Dict]:
    import calendar

    tree: Dict[str, Dict] = {}

    for tkr, df in data.items():
        cat, asset_name = _category_for_ticker(tkr)
        if cat != category or df.empty:
            continue

        tmp = df.copy()
        tmp["date"] = pd.to_datetime(tmp["date"])
        tmp = tmp.sort_values("date")
        tmp["month_num"] = tmp["date"].dt.month

        # Daily pip range
        tmp["daily_pip"] = tmp["high"] - tmp["low"]

        # Daily returns (%)
        tmp["return"] = tmp["close"].pct_change() * 100
        tmp_returns = tmp.dropna(subset=["return"])
        tmp_pips = tmp.dropna(subset=["daily_pip"])

        month_dict: Dict[str, Dict] = {}

        for m in range(1, 13):
            month_data = tmp_pips[tmp_pips["month_num"] == m]
            if month_data.empty:
                continue

            pip_stats = {
                "min": round(month_data["daily_pip"].min(), 4),
                "max": round(month_data["daily_pip"].max(), 4),
                "mean": round(month_data["daily_pip"].mean(), 4),
                "count": int(month_data["daily_pip"].count()),
                "normalized": 0
            }
            if normalize:
                pip_stats["normalized"] = pip_stats["count"] / len(tmp_pips)

            return_data = tmp_returns[tmp_returns["month_num"] == m]
            return_stats = {
                "min": round(return_data["return"].min(), 4),
                "max": round(return_data["return"].max(), 4),
                "mean": round(return_data["return"].mean(), 4),
                "count": int(return_data["return"].count()),
                "normalized": 0
            }
            if normalize:
                return_stats["normalized"] = return_stats["count"] / len(tmp_returns)

            month_name = calendar.month_abbr[m]
            month_dict[month_name] = {"pip": pip_stats, "return": return_stats}

        # Aggregates (raw, no normalization unless user wants)
        agg_pip = {
            "min": round(tmp_pips["daily_pip"].min(), 4),
            "max": round(tmp_pips["daily_pip"].max(), 4),
            "mean": round(tmp_pips["daily_pip"].mean(), 4),
            "count": int(tmp_pips["daily_pip"].count())
        }
        agg_return = {
            "min": round(tmp_returns["return"].min(), 4),
            "max": round(tmp_returns["return"].max(), 4),
            "mean": round(tmp_returns["return"].mean(), 4),
            "count": int(tmp_returns["return"].count())
        }
        month_dict["Aggregate"] = {"pip": agg_pip, "return": agg_return}

        tree[asset_name] = month_dict

    return tree

# ----------------------------------
# Category Helper
# ----------------------------------
def _category_for_ticker(tkr: str):
    if tkr in ["^GSPC", "^GDAXI"]:
        return "Indices", tkr
    if tkr in ["EURUSD=X", "USDJPY=X"]:
        return "Forex", tkr
    if tkr in ["ZS=F", "CL=F", "GC=F"]:
        return "Commodities", tkr
    return None, tkr

# ----------------------------------
# Streamlit App
# ----------------------------------
st.title("📊 Market Intelligence Dashboard")

normalize_checkbox = st.sidebar.checkbox("Normalize monthly stats", value=True)

assets_data = {}
for tkr, cot_name in COT_ASSET_NAMES.items():
    price = fetch_price_history(tkr)
    cot = fetch_cot_data(cot_name) if cot_name else pd.DataFrame()
    merged = merge_cot_price(cot, price)
    hg = add_health_gauge(merged)
    assets_data[tkr] = hg

category_choice = st.sidebar.selectbox("Select Category", ["Indices", "Forex", "Commodities"])
tree = pip_and_return_tree_daily_pips(assets_data, category_choice, normalize=normalize_checkbox)

st.subheader("Granular Pip & Return Distributions")
st.json(tree)