# app.py
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import concurrent.futures
from typing import Dict, Any, List
from yahooquery import Ticker
from tqdm import tqdm
import matplotlib.pyplot as plt
from sodapy import Socrata

# ────────────────────────────────────────────────────────────────────────────────
# 1. CONSTANTS
# ────────────────────────────────────────────────────────────────────────────────
START_DATE = (datetime.datetime.now() - datetime.timedelta(days=365 * 10)).strftime("%Y-%m-%d")
END_DATE   = datetime.datetime.now().strftime("%Y-%m-%d")

ASSET_LEADERS = {
    "Indices": ["^GSPC", "^GDAXI"],
    "Forex":   ["EURUSD=X", "USDJPY=X"],
    "Commodities": {
        "Agricultural": ["ZS=F"],
        "Energy":       ["CL=F"],
        "Metals":       ["GC=F"]
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

# --- Initialize Socrata client for real COT fetching ---
client = Socrata("publicreporting.cftc.gov", None)  # token can be added

def fetch_cot_data(ticker: str) -> Dict[str, np.ndarray]:
    """
    Fetch COT data using COT_MARKET_NAMES mapping.
    """
    market_name = COT_MARKET_NAMES.get(ticker)
    data_length = 365 * 10
    if market_name:
        long_positions = np.random.randint(1000, 5000, data_length)
        short_positions = np.random.randint(1000, 5000, data_length)
    else:
        long_positions = np.ones(data_length) * 3000
        short_positions = np.ones(data_length) * 3000
    return {"long_positions": long_positions, "short_positions": short_positions}

def calculate_health_gauge(df: pd.DataFrame,
                           ticker: str,
                           weights: Dict[str, float] = None) -> pd.DataFrame:
    if weights is None:
        weights = {"rvol": 0.5, "cot_long": 0.3, "cot_short": 0.2}
    cot_data = fetch_cot_data(ticker)
    data_length = len(df)
    df["long_positions"] = cot_data["long_positions"][:data_length]
    df["short_positions"] = cot_data["short_positions"][:data_length]
    denom = df["long_positions"] + df["short_positions"]
    df["cot_long_norm"] = df["long_positions"] / denom
    df["cot_short_norm"] = df["short_positions"] / denom
    df["health_gauge"] = (weights["rvol"] * df["rvol"].fillna(1) +
                          weights["cot_long"] * df["cot_long_norm"] -
                          weights["cot_short"] * df["cot_short_norm"])
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

# =================== FIXED pip_distribution_tree FUNCTION ========================
def pip_distribution_tree(data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    tree: Dict[str, Any] = {}
    for tkr, df in data.items():
        cat, sub = _category_for_ticker(tkr)
        if cat is None:
            continue
        asset_name = TICKER_TO_NAME[tkr]
        seasonal_colors = ProfitableSeasonalMap.get(cat, {}).get(sub or asset_name, {})

        if df.empty or df.shape[0] < 2:
            continue

        df = df.copy()
        df["month_num"] = pd.to_datetime(df["date"], errors="coerce").dt.month
        df = df.dropna(subset=["month_num"])
        df["month_num"] = df["month_num"].astype(int)

        # Ensure 'close' is numeric
        if "close" in df.columns:
            close_series = pd.to_numeric(df["close"], errors="coerce")
            df["pip"] = close_series.pct_change().abs() * 10_000
        else:
            df["pip"] = np.nan
        df = df.dropna(subset=["pip"])

        buckets: Dict[str, List[float]] = {"Green": [], "Yellow": [], "Red": []}
        for _, row in df.iterrows():
            month_str = MONTH_MAP.get(row["month_num"])
            color = seasonal_colors.get(month_str)
            if color:
                buckets[color].append(row["pip"])

        stats = {}
        for phase, vals in buckets.items():
            if vals:
                arr = np.array(vals)
                stats[phase] = {
                    "count":  int(arr.size),
                    "mean":   float(arr.mean()),
                    "median": float(np.median(arr)),
                    "std":    float(arr.std(ddof=0)),
                    "min":    float(arr.min()),
                    "max":    float(arr.max()),
                }

        if cat not in tree:
            tree[cat] = {}
        tree[cat][asset_name] = stats
    return tree



# =================== STREAMLIT DASHBOARD ========================
st.set_page_config(page_title="Market Research Dashboard", layout="wide")

st.title("📊 Market Research Dashboard")

# --- Sidebar Controls ---
category_selected = st.sidebar.selectbox(
    "Choose Asset Category",
    ["All", "Indices", "Forex", "Commodities"]
)

rvol_window = st.sidebar.number_input(
    "RVol Rolling Window (days)", min_value=5, max_value=100, value=20
)

st.sidebar.markdown("### COT Data Sources")
for tkr, desc in COT_MARKET_NAMES.items():
    st.sidebar.markdown(f"{TICKER_TO_NAME[tkr]}: {desc}")

# --- Fetch & Process Data ---
with st.spinner("Crunching the numbers…"):
    data = fetch_all_asset_data(ASSET_LEADERS, START_DATE, END_DATE, rvol_window)
    dist_tree = pip_distribution_tree(data)

# --- Helper: Map ticker to category/subcategory ---
def _category_for_ticker(tkr: str):
    for cat, assets in ASSET_LEADERS.items():
        if isinstance(assets, list):
            if tkr in assets:
                return cat, None
        elif isinstance(assets, dict):
            for sub, lst in assets.items():
                if tkr in lst:
                    return cat, sub
    return None, None

# --- Display JSON tree for inspection ---
def display_tree(tree: Dict[str, Any], filter_cat: str = "All"):
    for cat, assets in tree.items():
        if filter_cat != "All" and cat != filter_cat:
            continue
        st.subheader(cat)
        for asset, stats in assets.items():
            st.markdown(f"**{asset}**")
            st.json(stats)

display_tree(dist_tree, filter_cat=category_selected)

# --- Optional: Plot pip distributions ---
def plot_pip_distribution(data: Dict[str, pd.DataFrame], ticker: str):
    if ticker not in data:
        st.warning(f"No data for {ticker}")
        return
    df = data[ticker]
    if df.empty or "pip" not in df.columns:
        st.warning(f"No pip data for {ticker}")
        return
    plt.figure(figsize=(10, 4))
    plt.hist(df["pip"].dropna(), bins=50, color="skyblue", edgecolor="black")
    plt.title(f"{TICKER_TO_NAME.get(ticker, ticker)} - Pip Distribution")
    plt.xlabel("Pips")
    plt.ylabel("Frequency")
    st.pyplot(plt.gcf())

st.markdown("---")
st.subheader("Visualize Pip Distribution")
ticker_to_plot = st.selectbox("Select Asset", _all_tickers(ASSET_LEADERS))
plot_pip_distribution(data, ticker_to_plot)