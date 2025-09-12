import pandas as pd
import numpy as np
import streamlit as st
import json
from yahooquery import Ticker
import concurrent.futures
from datetime import datetime
import plotly.express as px
import json



# --- Asset definitions ---
ASSET_LEADERS = {
    "Indices": ["^GSPC"],
    "Forex": ["EURUSD=X", "JPY=X"],
    "Agricultural": ["ZS=F"],
    "Energy": ["CL=F"],
    "Metals": ["GC=F"]
}

TICKER_TO_NAME = {
    "^GSPC": "S&P 500",
    "EURUSD=X": "EUR/USD",
    "JPY=X": "USD/JPY",
    "ZS=F": "Soybeans",
    "CL=F": "WTI Crude",
    "GC=F": "Gold"
}

ProfitableSeasonalMap = {
    "Indices": {"S&P 500": {m: "Green" for m in range(1, 13)}},
    "Forex": {"EUR/USD": {m: "Yellow" for m in range(1, 13)}, "USD/JPY": {m: "Red" for m in range(1, 13)}},
    "Agricultural": {"Soybeans": {m: "Green" for m in range(1, 13)}},
    "Energy": {"WTI Crude": {m: "Yellow" for m in range(1, 13)}},
    "Metals": {"Gold": {m: "Red" for m in range(1, 13)}}
}

# --- Sidebar ---
category_selected = st.sidebar.selectbox("Choose Asset Category", list(ASSET_LEADERS.keys()))
rvol_window = st.sidebar.number_input("RVol Rolling Window (days)", min_value=5, max_value=60, value=20)

START_DATE = "2013-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

# --- Helper functions ---
def fetch_single(ticker):
    t = Ticker(ticker)
    df = t.history(start=START_DATE, end=END_DATE)
    if df.empty:
        return ticker, pd.DataFrame()
    
    df.reset_index(inplace=True)

    # Ensure consistent timezone handling - convert to tz-naive
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        if hasattr(df['date'].dt, 'tz') and df['date'].dt.tz is not None:
            df['date'] = df['date'].dt.tz_localize(None)
    
    # Compute rolling volatility
    df['rvol'] = df['close'].pct_change().rolling(rvol_window).std() * np.sqrt(rvol_window)
    return ticker, df

def fetch_all_asset_data(assets, start, end, rvol_window):
    data = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(fetch_single, t) for cat in assets.values() for t in cat]
        for f in concurrent.futures.as_completed(futures):
            ticker, df = f.result()
            data[ticker] = df
    return data

def _category_for_ticker(tkr: str):
    for cat, tickers in ASSET_LEADERS.items():
        if tkr in tickers:
            sub = TICKER_TO_NAME.get(tkr, None)
            return cat, sub
    return None, None

def pip_distribution_tree(data: dict):
    tree = {}
    for tkr, df in data.items():
        cat, sub = _category_for_ticker(tkr)
        if cat is None or df.empty:
            continue
        
        asset_name = TICKER_TO_NAME[tkr]
        
        # Create a copy to avoid modifying the original dataframe
        temp_df = df.copy()
        
        # Extract month number from the date
        if 'date' in temp_df.columns:
            # Ensure dates are tz-naive
            temp_df["month_num"] = pd.to_datetime(temp_df["date"]).dt.month
            temp_df["phase"] = temp_df["month_num"].map(lambda m: ProfitableSeasonalMap[cat][sub][m])
            
            pip_dist = temp_df.groupby("phase")["close"].agg(["min", "max", "mean"]).to_dict()
            tree[asset_name] = pip_dist
    
    return tree


# --- Fetch & process data ---
with st.spinner("Crunching the numbers…"):
    data = fetch_all_asset_data(ASSET_LEADERS, START_DATE, END_DATE, rvol_window)
    dist_tree = pip_distribution_tree(data)

# --- Display each asset distribution ---
if category_selected != "All":
    tickers_to_show = ASSET_LEADERS[category_selected]
else:
    tickers_to_show = [t for sublist in ASSET_LEADERS.values() for t in sublist]

for tkr in tickers_to_show:
    df = data[tkr]
    if df.empty:
        st.warning(f"No data for {TICKER_TO_NAME[tkr]}")
        continue

    # Plot rolling volatility distribution
    fig = px.line(df, x="date", y="rvol", title=f"{TICKER_TO_NAME[tkr]} Rolling Volatility")
    fig.update_layout(
        template="plotly_white",
        title_font_size=18,
        xaxis_title="Date",
        yaxis_title="Rolling Volatility",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Display JSON tree ---
st.subheader("Pip Range Distribution Tree")
st.json(dist_tree)

# --- Download JSON ---
json_filename = f"pip_distribution_{category_selected.replace(' ', '_')}.json"
json_str = json.dumps(dist_tree, indent=4)
st.download_button(
    label="Download JSON Tree",
    data=json_str,
    file_name=json_filename,
    mime="application/json"
)
