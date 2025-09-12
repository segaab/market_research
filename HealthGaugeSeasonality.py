# === app.py ===
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import concurrent.futures
from typing import Dict, Any, List
from yahooquery import Ticker
import matplotlib.pyplot as plt
import plotly.express as px
import json

# ────────────────────────────────────────────────────────────────────────────────
# 1. CONSTANTS
# ────────────────────────────────────────────────────────────────────────────────
START_DATE = (datetime.datetime.now() - datetime.timedelta(days=365 * 10)).strftime("%Y-%m-%d")
END_DATE   = datetime.datetime.now().strftime("%Y-%m-%d")

ASSET_LEADERS = {
    "Indices": ["^GSPC", "^GDAXI"],              # S&P 500, DAX
    "Forex":   ["EURUSD=X", "USDJPY=X"],          # EUR/USD, USD/JPY
    "Commodities": {
        "Agricultural": ["ZS=F"],                 # Soybeans
        "Energy":       ["CL=F"],                 # Crude
        "Metals":       ["GC=F"]                  # Gold
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

# ============================== ProfitableSeasonalMap ==========================
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
# 2. FUNCTIONS
# ────────────────────────────────────────────────────────────────────────────────

def _category_for_ticker(tkr: str):
    """Return category and subcategory for a given ticker."""
    for cat, subs in ASSET_LEADERS.items():
        if isinstance(subs, dict):
            for sub, tickers in subs.items():
                if tkr in tickers:
                    return cat, sub
        else:
            if tkr in subs:
                return cat, tkr
    return None, None


def fetch_all_asset_data(assets: Dict[str, Any], start: str, end: str, rvol_window: int):
    """Fetch historical data for all tickers using yahooquery."""
    data = {}

    def fetch_single(ticker):
        t = Ticker(ticker)
        df = t.history(start=start, end=end)
        if df.empty:
            return ticker, pd.DataFrame()
        df.reset_index(inplace=True)
        # Ensure tz-naive datetime
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        # Compute rolling volatility
        df['rvol'] = df['close'].pct_change().rolling(rvol_window).std() * np.sqrt(rvol_window)
        return ticker, df

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(fetch_single, t) for cat in assets.values() for t in (cat if isinstance(cat, list) else [tk for sub in cat.values() for tk in sub])]
        for f in concurrent.futures.as_completed(futures):
            ticker, df = f.result()
            data[ticker] = df

    return data


def pip_distribution_tree(data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Compute pip distribution tree per asset based on seasonal map."""
    tree = {}
    for tkr, df in data.items():
        cat, sub = _category_for_ticker(tkr)
        if cat is None:
            continue
        asset_name = TICKER_TO_NAME[tkr]

        if df.empty:
            continue

        # --- tz-naive datetime fix ---
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        df["month_num"] = df["date"].dt.month
        df["phase"] = df["month_num"].map(lambda m: ProfitableSeasonalMap[cat][sub][MONTH_MAP[m]])

        # Group by phase
        pip_dist = df.groupby("phase")["close"].agg(["min", "max", "mean"]).to_dict('index')
        tree[asset_name] = pip_dist

    return tree


def plot_distribution(df: pd.DataFrame, ticker: str):
    """Plot pip-range distribution using Plotly."""
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    fig = px.line(df, x='date', y='close', title=f"{ticker} Price Distribution")
    st.plotly_chart(fig, use_container_width=True)


def download_json(tree: Dict[str, Any], category: str):
    """Provide download link for JSON tree."""
    filename = f"{category}_pip_distribution.json"
    json_str = json.dumps(tree, indent=2)
    st.download_button("Download JSON Tree", data=json_str, file_name=filename, mime="application/json")


# ────────────────────────────────────────────────────────────────────────────────
# 3. STREAMLIT UI
# ────────────────────────────────────────────────────────────────────────────────

st.title("📊 Market Research Dashboard")

category_selected = st.sidebar.selectbox("Choose Asset Category", list(ASSET_LEADERS.keys()))
rvol_window = st.sidebar.number_input("RVol Rolling Window (days)", min_value=1, max_value=100, value=20)

with st.spinner("Crunching the numbers…"):
    data = fetch_all_asset_data(ASSET_LEADERS, START_DATE, END_DATE, rvol_window)
    dist_tree = pip_distribution_tree(data)

# --- Display each asset distribution ---
if category_selected != "All":
    for tkr in (ASSET_LEADERS[category_selected] if isinstance(ASSET_LEADERS[category_selected], list)
                else [tk for sub in ASSET_LEADERS[category_selected].values() for tk in sub]):
        if tkr in data:
            st.subheader(TICKER_TO_NAME[tkr])
            plot_distribution(data[tkr], TICKER_TO_NAME[tkr])

# --- Display JSON tree ---
st.subheader("Seasonal Pip Distribution Tree")
st.json(dist_tree)

# --- Download JSON ---
download_json(dist_tree, category_selected)