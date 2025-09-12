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

# COT client
client = Socrata("publicreporting.cftc.gov", None)

def fetch_cot_data(cot_name: str, start_date: str = START_DATE, end_date: str = END_DATE) -> pd.DataFrame:
    query = (
        f"market_and_exchange_names='{cot_name}' "
        f"AND report_date_as_yyyy_mm_dd >= '{start_date}' "
        f"AND report_date_as_yyyy_mm_dd <= '{end_date}'"
    )
    results = client.get("6dca-aqww", where=query, limit=5000)
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame.from_records(results)
    keep_cols = [
        "report_date_as_yyyy_mm_dd",
        "market_and_exchange_names",
        "open_interest_all",
        "commercial_long_all",
        "commercial_short_all",
        "non_commercial_long_all",
        "non_commercial_short_all",
    ]
    df = df[keep_cols]
    for col in keep_cols[2:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["commercial_net"] = df["commercial_long_all"] - df["commercial_short_all"]
    df["non_commercial_net"] = df["non_commercial_long_all"] - df["non_commercial_short_all"]
    df["report_date_as_yyyy_mm_dd"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
    return df.sort_values("report_date_as_yyyy_mm_dd").reset_index(drop=True)

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

def calculate_health_gauge(df: pd.DataFrame,
                           ticker: str,
                           weights: Dict[str, float] = None) -> pd.DataFrame:
    if weights is None:
        weights = {"rvol": 0.5, "cot_long": 0.3, "cot_short": 0.2}
    
    market_name = COT_MARKET_NAMES.get(ticker)
    cot_df = fetch_cot_data(market_name) if market_name else pd.DataFrame()
    
    if cot_df.empty:
        df["long_positions"] = 3000
        df["short_positions"] = 3000
    else:
        length = len(df)
        df["long_positions"] = cot_df["commercial_long_all"].values[:length]
        df["short_positions"] = cot_df["commercial_short_all"].values[:length]
    
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


# ────────────────────────────────────────────────────────────────────────────────
# 3. DAILY-PIP-RANGE DISTRIBUTION by SEASONAL PHASE
# ────────────────────────────────────────────────────────────────────────────────
def _category_for_ticker(ticker: str):
    for cat, items in ASSET_LEADERS.items():
        if isinstance(items, list):
            if ticker in items:
                return cat, None
        else:
            for sub, lst in items.items():
                if ticker in lst:
                    return cat, sub
    return None, None

def pip_distribution_tree(data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    tree: Dict[str, Any] = {}
    for tkr, df in data.items():
        cat, sub = _category_for_ticker(tkr)
        if cat is None:
            continue
        asset_name = TICKER_TO_NAME[tkr]
        seasonal_colors = (ProfitableSeasonalMap[cat][sub]
                           if sub else
                           ProfitableSeasonalMap[cat][asset_name])

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "close"])
        df["month_num"] = df["date"].dt.month
        df["pip"] = df["close"].pct_change().abs() * 10_000

        buckets: Dict[str, List[float]] = {"Green": [], "Yellow": [], "Red": []}
        for _, row in df.dropna(subset=["pip", "month_num"]).iterrows():
            month_num = int(row["month_num"])
            color = seasonal_colors[MONTH_MAP[month_num]]
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

@st.cache_data(show_spinner=False, ttl=3600)
def build_tree(rvol_window: int = 20) -> Dict[str, Any]:
    data = fetch_all_asset_data(ASSET_LEADERS, START_DATE, END_DATE, rvol_window)
    return pip_distribution_tree(data)

# ────────────────────────────────────────────────────────────────────────────────
# 4. STREAMLIT FRONT-END
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="10-Year Market Research", layout="wide")
st.title("📊 Market Research Dashboard")
st.markdown(
    """
    This dashboard downloads **10 years** of history for a handful of Indices, Forex pairs and Commodities,
    computes a *Health Gauge* and the **daily pip-range distribution** for each
    seasonal "phase" (Green / Yellow / Red).  
    Select the category on the left to inspect the JSON tree.
    """
)

category_selected = st.sidebar.selectbox(
    "Choose Asset Category",
    ["All"] + list(ASSET_LEADERS.keys())
)

rvol_window = st.sidebar.slider(
    "RVol Rolling Window (days)",
    min_value=5,
    max_value=60,
    value=20,
    step=1,
    help="Number of days to use for the relative volume calculation"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### COT Data Sources")
for ticker, market_name in COT_MARKET_NAMES.items():
    if market_name:
        st.sidebar.markdown(f"**{TICKER_TO_NAME[ticker]}**: {market_name}")
    else:
        st.sidebar.markdown(f"**{TICKER_TO_NAME[ticker]}**: No COT data available")

with st.spinner("Crunching the numbers…"):
    dist_tree = build_tree(rvol_window)

if category_selected == "All":
    st.subheader("Complete distribution tree")
    st.json(dist_tree)
else:
    st.subheader(f"{category_selected} distribution tree")
    st.json({category_selected: dist_tree.get(category_selected, {})})

if st.checkbox("Show quick visual for selected category"):
    cat_tree = dist_tree if category_selected == "All" else {category_selected: dist_tree.get(category_selected, {})}
    col1, col2 = st.columns(2)
    for cat, assets in cat_tree.items():
        col1.markdown(f"### {cat}")
        for asset, phases in assets.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            phase_colors = []
            mean_vals = []
            median_vals = []
            for phase in ["Green", "Yellow", "Red"]:
                if phase in phases:
                    phase_colors.append(phase)
                    mean_vals.append(phases[phase]["mean"])
                    median_vals.append(phases[phase]["median"])
            x = np.arange(len(phase_colors))
            width = 0.35
            ax.bar(x - width/2, mean_vals, width, color=[c.lower() for c in phase_colors], alpha=0.7, label='Mean')
            ax.bar(x + width/2, median_vals, width, color=[c.lower() for c in phase_colors], alpha=0.4, hatch='///', label='Median')
            for i, v in enumerate(mean_vals):
                ax.text(i - width/2, v + 0.1, f'{v:.1f}', ha='center', fontsize=9)
            for i, v in enumerate(median_vals):
                ax.text(i + width/2, v + 0.1, f'{v:.1f}', ha='center', fontsize=9)
            ax.set_title(f"{asset} -- Daily Pip Range by Phase")
            ax.set_ylabel("Pips (10,000× abs pct change)")
            ax.set_xticks(x)
            ax.set_xticklabels(phase_colors)
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            col1.pyplot(fig)
            stats_df = pd.DataFrame({
                "Phase": phase_colors,
                "Count": [phases[p]["count"] for p in phase_colors],
                "Mean": [f"{phases[p]['mean']:.2f}" for p in phase_colors],
                "Median": [f"{phases[p]['median']:.2f}" for p in phase_colors],
                "Min": [f"{phases[p]['min']:.2f}" for p in phase_colors],
                "Max": [f"{phases[p]['max']:.2f}" for p in phase_colors],
                "StdDev": [f"{phases[p]['std']:.2f}" for p in phase_colors],
            })
            col2.markdown(f"### {asset} Statistics")
            col2.dataframe(stats_df, use_container_width=True)