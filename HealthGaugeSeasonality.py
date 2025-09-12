# app.py - Chunk 1
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import concurrent.futures
from typing import Dict, Any, List
from yahooquery import Ticker
from sodapy import Socrata
import matplotlib.pyplot as plt
import plotly.express as px
import json

# ────────────────────────────────────────────────────────────────────────────────
# 1. CONSTANTS
# ────────────────────────────────────────────────────────────────────────────────
START_DATE = (datetime.datetime.now() - datetime.timedelta(days=365 * 10)).strftime("%Y-%m-%d")
END_DATE   = datetime.datetime.now().strftime("%Y-%m-%d")

ASSET_LEADERS = {
    "Indices": ["^GSPC", "^GDAXI"],
    "Forex": ["EURUSD=X", "USDJPY=X"],
    "Commodities": {
        "Agricultural": ["ZS=F"],
        "Energy": ["CL=F"],
        "Metals": ["GC=F"]
    }
}

COT_MARKET_NAMES = {
    "^GSPC": "S&P 500 Consolidated -- Chicago Mercantile Exchange",
    "EURUSD=X": "EURO FX -- Chicago Mercantile Exchange",
    "USDJPY=X": "JAPANESE YEN -- Chicago Mercantile Exchange",
    "ZS=F": "SOYBEANS -- Chicago Board of Trade",
    "CL=F": "WTI-PHYSICAL -- New York Mercantile Exchange",
    "GC=F": "GOLD -- Commodity Exchange Inc."
}

TICKER_TO_NAME = {
    "^GSPC": "S&P500",
    "^GDAXI": "DAX",
    "EURUSD=X": "EUR/USD",
    "USDJPY=X": "USD/JPY",
    "ZS=F": "Agricultural",
    "CL=F": "Energy",
    "GC=F": "Metals",
}

MONTH_MAP = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

# Color-coded seasonal map
ProfitableSeasonalMap = {
    "Indices": {
        "S&P500": {"Jan": "Yellow", "Feb": "Yellow", "Mar": "Yellow", "Apr": "Green",
                   "May": "Yellow", "Jun": "Yellow", "Jul": "Yellow", "Aug": "Yellow",
                   "Sep": "Red", "Oct": "Green", "Nov": "Green", "Dec": "Green"},
        "DAX":    {"Jan": "Yellow", "Feb": "Yellow", "Mar": "Yellow", "Apr": "Green",
                   "May": "Yellow", "Jun": "Yellow", "Jul": "Yellow", "Aug": "Yellow",
                   "Sep": "Red", "Oct": "Green", "Nov": "Green", "Dec": "Green"},
    },
    "Forex": {
        "EUR/USD": {"Jan": "Green", "Feb": "Yellow", "Mar": "Yellow", "Apr": "Yellow",
                    "May": "Yellow", "Jun": "Yellow", "Jul": "Yellow", "Aug": "Yellow",
                    "Sep": "Green", "Oct": "Yellow", "Nov": "Yellow", "Dec": "Yellow"},
        "USD/JPY": {"Jan": "Yellow", "Feb": "Yellow", "Mar": "Yellow", "Apr": "Yellow",
                    "May": "Yellow", "Jun": "Yellow", "Jul": "Green", "Aug": "Yellow",
                    "Sep": "Yellow", "Oct": "Yellow", "Nov": "Yellow", "Dec": "Yellow"},
    },
    "Commodities": {
        "Agricultural": {"Jan": "Yellow", "Feb": "Yellow", "Mar": "Yellow", "Apr": "Yellow",
                         "May": "Yellow", "Jun": "Yellow", "Jul": "Green", "Aug": "Yellow",
                         "Sep": "Yellow", "Oct": "Yellow", "Nov": "Yellow", "Dec": "Yellow"},
        "Energy":       {"Jan": "Green", "Feb": "Yellow", "Mar": "Yellow", "Apr": "Yellow",
                         "May": "Yellow", "Jun": "Green", "Jul": "Green", "Aug": "Green",
                         "Sep": "Yellow", "Oct": "Yellow", "Nov": "Green", "Dec": "Green"},
        "Metals":       {"Jan": "Yellow", "Feb": "Yellow", "Mar": "Yellow", "Apr": "Yellow",
                         "May": "Yellow", "Jun": "Yellow", "Jul": "Yellow", "Aug": "Yellow",
                         "Sep": "Yellow", "Oct": "Green", "Nov": "Green", "Dec": "Green"},
    },
}

# ────────────────────────────────────────────────────────────────────────────────
# 2. BACK-END HELPERS
# ────────────────────────────────────────────────────────────────────────────────
def fetch_price_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    t = Ticker(symbol)
    df = t.history(start=start, end=end).reset_index()
    if df.empty:
        raise ValueError(f"No data for {symbol}")
    df.rename(columns={"adjclose": "close"}, inplace=True)
    return df[["symbol", "date", "close", "volume"]]

def calculate_rvol(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df["rvol"] = df["volume"] / df["volume"].rolling(window).mean()
    return df

# --- COT Fetching function (integrated) ---
from sodapy import Socrata

client = Socrata("publicreporting.cftc.gov", None)

def fetch_cot_data(ticker: str, start_date: str = START_DATE, end_date: str = END_DATE) -> pd.DataFrame:
    market_name = COT_MARKET_NAMES.get(ticker)
    if not market_name:
        return pd.DataFrame()  # no COT data
    
    query = (f"market_and_exchange_names='{market_name}' "
             f"AND report_date_as_yyyy_mm_dd >= '{start_date}' "
             f"AND report_date_as_yyyy_mm_dd <= '{end_date}'")
    results = client.get("6dca-aqww", where=query, limit=5000)
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame.from_records(results)
    keep_cols = ["report_date_as_yyyy_mm_dd", "market_and_exchange_names",
                 "open_interest_all", "commercial_long_all", "commercial_short_all",
                 "non_commercial_long_all", "non_commercial_short_all"]
    df = df[keep_cols]
    for col in keep_cols[2:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["commercial_net"] = df["commercial_long_all"] - df["commercial_short_all"]
    df["non_commercial_net"] = df["non_commercial_long_all"] - df["non_commercial_short_all"]
    df["report_date_as_yyyy_mm_dd"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
    return df.sort_values("report_date_as_yyyy_mm_dd").reset_index(drop=True)

def calculate_health_gauge(df: pd.DataFrame,
                           ticker: str,
                           weights: Dict[str, float] = None) -> pd.DataFrame:
    if weights is None:
        weights = {"rvol": 0.5, "cot_long": 0.3, "cot_short": 0.2}
    
    cot_df = fetch_cot_data(ticker)
    if not cot_df.empty:
        df = df.copy()
        df = df.iloc[-len(cot_df):]  # align lengths
        denom = cot_df["commercial_long_all"] + cot_df["commercial_short_all"]
        df["cot_long_norm"] = cot_df["commercial_long_all"] / denom
        df["cot_short_norm"] = cot_df["commercial_short_all"] / denom
        df["health_gauge"] = (weights["rvol"] * df["rvol"].fillna(1) +
                              weights["cot_long"] * df["cot_long_norm"] -
                              weights["cot_short"] * df["cot_short_norm"])
    else:
        df["health_gauge"] = df["rvol"].fillna(1)
    return df

def _all_tickers(asset_dict: Dict[str, Any]) -> List[str]:
    tickers = []
    for v in asset_dict.values():
        if isinstance(v, list):
            tickers.extend(v)
        else:
            for lst in v.values():
                tickers.extend(lst)
    return tickers

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_all_asset_data(asset_dict: Dict[str, Any],
                         start: str,
                         end: str,
                         rvol_window: int = 20) -> Dict[str, pd.DataFrame]:
    tickers = _all_tickers(asset_dict)
    results: Dict[str, pd.DataFrame] = {}
    prog = st.progress(0.0, text="Downloading price history …")
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(fetch_price_data, t, start, end): t for t in tickers}
        for i, fut in enumerate(concurrent.futures.as_completed(futures)):
            tkr = futures[fut]
            try:
                df = fut.result()
                df = calculate_rvol(df, window=rvol_window)
                df = calculate_health_gauge(df, tkr)
                results[tkr] = df
            except Exception as e:
                st.warning(f"{tkr}: {e}")
            prog.progress((i + 1) / len(futures))
    return results



# app.py - Chunk 2
# ────────────────────────────────────────────────────────────────────────────────
# 3. HELPER FUNCTIONS
# ────────────────────────────────────────────────────────────────────────────────
def _category_for_ticker(tkr: str):
    for cat, subdict in ProfitableSeasonalMap.items():
        for sub, months in subdict.items():
            if TICKER_TO_NAME.get(tkr) == sub:
                return cat, sub
    return None, None

def pip_distribution_tree(data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    tree: Dict[str, Any] = {}
    for tkr, df in data.items():
        cat, sub = _category_for_ticker(tkr)
        if cat is None:
            continue
        asset_name = TICKER_TO_NAME[tkr]
        df["month_num"] = pd.to_datetime(df["date"]).dt.month
        df["phase"] = df["month_num"].map(lambda m: ProfitableSeasonalMap[cat][sub].get(MONTH_MAP[m], "Yellow"))
        # Group by phase and calculate pip ranges
        pip_dist = df.groupby("phase")["close"].agg(["min", "max", "mean"]).to_dict(orient="index")
        if cat not in tree:
            tree[cat] = {}
        tree[cat][asset_name] = pip_dist
    return tree

def display_distribution(df: pd.DataFrame, ticker: str):
    df["month_num"] = pd.to_datetime(df["date"]).dt.month
    df["phase"] = df["month_num"].map(lambda m: ProfitableSeasonalMap[_category_for_ticker(ticker)[0]][_category_for_ticker(ticker)[1]].get(MONTH_MAP[m], "Yellow"))
    fig = px.box(df, x="phase", y="close", color="phase", title=f"Price Distribution by Seasonal Phase ({ticker})",
                 color_discrete_map={"Green":"green", "Yellow":"gold", "Red":"red"})
    st.plotly_chart(fig, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────────
# 4. STREAMLIT LAYOUT
# ────────────────────────────────────────────────────────────────────────────────
st.title("📊 Market Research Dashboard")

category_selected = st.sidebar.selectbox("Choose Asset Category", ["Indices", "Forex", "Commodities", "All"])
rvol_window = st.sidebar.number_input("RVol Rolling Window (days)", min_value=5, max_value=100, value=20)

# Fetch & process data
with st.spinner("Crunching the numbers…"):
    data = fetch_all_asset_data(ASSET_LEADERS, START_DATE, END_DATE, rvol_window)
    dist_tree = pip_distribution_tree(data)

# Display distribution for each ticker
if category_selected != "All":
    tickers_to_show = [t for t in data.keys() if _category_for_ticker(t)[0] == category_selected]
else:
    tickers_to_show = list(data.keys())

for tkr in tickers_to_show:
    st.subheader(TICKER_TO_NAME[tkr])
    df = data[tkr]
    st.dataframe(df.tail(5))
    display_distribution(df, tkr)

# Display JSON tree
st.subheader("📁 Pip Distribution Tree")
st.json(dist_tree)

# Download JSON tree
json_str = json.dumps(dist_tree, indent=2)
filename = f"pip_distribution_{category_selected.lower()}.json"
st.download_button(
    label="Download JSON",
    data=json_str,
    file_name=filename,
    mime="application/json"
)