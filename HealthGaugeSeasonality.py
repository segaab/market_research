import pandas as pd
import numpy as np
import streamlit as st
import json
from yahooquery import Ticker
import concurrent.futures
from datetime import datetime
import plotly.express as px

# ── Asset definitions ──────────────────────────────────────────────────────────
ASSET_LEADERS = {
    "Indices": ["^GSPC"],
    "Forex": ["EURUSD=X", "JPY=X"],
    "Agricultural": ["ZS=F"],
    "Energy": ["CL=F"],
    "Metals": ["GC=F"],
}

TICKER_TO_NAME = {
    "^GSPC": "S&P 500",
    "EURUSD=X": "EUR/USD",
    "JPY=X": "USD/JPY",
    "ZS=F": "Soybeans",
    "CL=F": "WTI Crude",
    "GC=F": "Gold",
}

ProfitableSeasonalMap = {
    "Indices": {"S&P 500": {m: "Green" for m in range(1, 13)}},
    "Forex": {
        "EUR/USD": {m: "Yellow" for m in range(1, 13)},
        "USD/JPY": {m: "Red" for m in range(1, 13)},
    },
    "Agricultural": {"Soybeans": {m: "Green" for m in range(1, 13)}},
    "Energy": {"WTI Crude": {m: "Yellow" for m in range(1, 13)}},
    "Metals": {"Gold": {m: "Red" for m in range(1, 13)}},
}

# ── Sidebar ────────────────────────────────────────────────────────────────────
category_selected = st.sidebar.selectbox(
    "Choose Asset Category", list(ASSET_LEADERS.keys())
)
rvol_window = st.sidebar.number_input(
    "RVol Rolling Window (days)", min_value=5, max_value=60, value=20
)

START_DATE = "2013-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

# ── Helper functions ───────────────────────────────────────────────────────────
def fetch_single(ticker: str) -> tuple[str, pd.DataFrame]:
    t = Ticker(ticker)
    df = t.history(start=START_DATE, end=END_DATE)
    if df.empty:
        return ticker, pd.DataFrame()

    df.reset_index(inplace=True)
    # Make dates tz-naive
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    # Rolling volatility
    df["rvol"] = df["close"].pct_change().rolling(rvol_window).std() * np.sqrt(rvol_window)

    # NEW: daily percentage returns
    df["return_pct"] = df["close"].pct_change() * 100
    return ticker, df


def fetch_all_asset_data(tickers: list[str]) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(fetch_single, t) for t in tickers]
        for f in concurrent.futures.as_completed(futures):
            ticker, df = f.result()
            data[ticker] = df
    return data


def _category_for_ticker(tkr: str) -> tuple[str | None, str | None]:
    for cat, lst in ASSET_LEADERS.items():
        if tkr in lst:
            return cat, TICKER_TO_NAME[tkr]
    return None, None


def pip_distribution_tree(
    data: dict[str, pd.DataFrame], category: str
) -> dict[str, dict]:
    """
    Build distribution tree ONLY for the currently selected category.
    Adds normalized counts for every colour phase.
    """
    tree: dict[str, dict] = {}

    for tkr, df in data.items():
        cat, asset_name = _category_for_ticker(tkr)
        if cat != category or df.empty:
            continue

        temp = df.copy()
        temp["month_num"] = pd.to_datetime(temp["date"]).dt.month
        temp["phase"] = temp["month_num"].map(
            lambda m: ProfitableSeasonalMap[cat][asset_name][m]
        )

        # Aggregate stats
        agg = temp.groupby("phase")["close"].agg(["min", "max", "mean", "count"])
        agg["normalized"] = agg["count"] / agg["count"].sum()
        tree[asset_name] = agg.round(4).to_dict(orient="index")

    return tree


# ── Orchestrate ────────────────────────────────────────────────────────────────
tickers_to_process = ASSET_LEADERS[category_selected]

with st.spinner("Crunching the numbers…"):
    data = fetch_all_asset_data(tickers_to_process)
    dist_tree = pip_distribution_tree(data, category_selected)

# NEW ➜ build a stacked dataframe of daily returns
df_results = (
    pd.concat(
        [
            df.loc[df["return_pct"].notna(), ["date", "return_pct"]].assign(
                asset=TICKER_TO_NAME[tkr]
            )
            for tkr, df in data.items()
            if not df.empty
        ],
        ignore_index=True,
    )
)




# ── Visualisation: rolling volatility ──────────────────────────────────────────
for tkr in tickers_to_process:
    df = data.get(tkr, pd.DataFrame())
    if df.empty:
        st.warning(f"No data for {TICKER_TO_NAME[tkr]}")
        continue

    fig = px.line(
        df,
        x="date",
        y="rvol",
        title=f"{TICKER_TO_NAME[tkr]} Rolling Volatility",
        template="plotly_white",
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Rolling Volatility",
        title_font_size=18,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Return-distribution visual ─────────────────────────────────────────────────
if not df_results.empty:
    st.subheader("Return Distribution")
    fig = px.histogram(
        df_results,
        x="return_pct",
        nbins=20,
        title="Return Distribution (%)",
        color_discrete_sequence=["#3366CC"],
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No valid return data available for the selected asset group.")

# ── JSON tree + download ───────────────────────────────────────────────────────
st.subheader("Pip Range Distribution Tree")
st.json(dist_tree)

json_filename = f"pip_distribution_{category_selected.replace(' ', '_')}.json"
st.download_button(
    label="Download JSON Tree",
    data=json.dumps(dist_tree, indent=4),
    file_name=json_filename,
    mime="application/json",
)