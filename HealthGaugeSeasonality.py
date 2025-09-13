# app.py (Chunk 1)
import time
import json
import calendar
import concurrent.futures
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sodapy import Socrata
from yahooquery import Ticker

# ── APP CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Multi-Asset Health Gauge", layout="wide")
START_DATE = "2013-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
COT_PAGE_SIZE = 15000
COT_SLEEP    = 0.35
YH_SLEEP     = 0.2

# ── MAPPINGS ──────────────────────────────────────────────────────────────────
ASSET_LEADERS: Dict[str, List[str]] = {
    "Indices":       ["^GSPC"],
    "Forex":         ["EURUSD=X", "USDJPY=X"],
    "Agricultural":  ["ZS=F"],
    "Energy":        ["CL=F"],
    "Metals":        ["GC=F"],
}

TICKER_TO_NAME = {
    "^GSPC":   "S&P 500",
    "EURUSD=X":"EUR/USD",
    "USDJPY=X":"USD/JPY",
    "ZS=F":    "Soybeans",
    "CL=F":    "WTI Crude",
    "GC=F":    "Gold",
}

COT_ASSET_NAMES = {
    "^GSPC": "S&P 500 – Chicago Mercantile Exchange",
    "^GDAXI": None,
    "EURUSD=X": "EURO FX – Chicago Mercantile Exchange",
    "USDJPY=X": "JAPANESE YEN – Chicago Mercantile Exchange",
    "ZS=F": "SOYBEANS – Chicago Board of Trade",
    "CL=F": "WTI-PHYSICAL – New York Mercantile Exchange",
    "GC=F": "GOLD – Commodity Exchange Inc."
}

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
category_selected = st.sidebar.selectbox(
    "Choose Asset Category", list(ASSET_LEADERS.keys())
)
rvol_window = st.sidebar.number_input(
    "RVol Rolling Window (days)", min_value=5, max_value=60, value=20
)
normalize_monthly_checkbox = st.sidebar.checkbox(
    "Normalize monthly distribution counts", value=True
)

# ── CACHING HELPERS ───────────────────────────────────────────────────────────
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
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    df["return_pct"] = df["close"].pct_change() * 100
    time.sleep(YH_SLEEP)
    return df

def _category_for_ticker(ticker: str) -> tuple[str, str]:
    """Return (category, asset_name) for a given ticker"""
    for category, tickers in ASSET_LEADERS.items():
        if ticker in tickers:
            return category, TICKER_TO_NAME.get(ticker, ticker)
    return "", ticker

@st.cache_data(show_spinner=False, ttl=24*3600)
def fetch_cot_data(cot_name: str, start_date: str, end_date: str) -> pd.DataFrame:
    client = Socrata("publicreporting.cftc.gov", None, timeout=30)
    where = (
        f"market_and_exchange_names='{cot_name}' AND "
        f"report_date_as_yyyy_mm_dd >= '{start_date}' AND "
        f"report_date_as_yyyy_mm_dd <= '{end_date}'"
    )
    rows: List[dict] = []
    offset = 0
    while True:
        batch = client.get("6dca-aqww", where=where,
                           order="report_date_as_yyyy_mm_dd",
                           limit=COT_PAGE_SIZE, offset=offset)
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

    # Check if required column exists before accessing it
    if 'report_date_as_yyyy_mm_dd' not in df.columns:
        return pd.DataFrame()  # Return empty DataFrame if key column is missing

    keep = [
        "report_date_as_yyyy_mm_dd",
        "open_interest_all",
        "commercial_long_all",
        "commercial_short_all",
    ]
    keep = [col for col in keep if col in df.columns]

    if "report_date_as_yyyy_mm_dd" not in keep:
        return pd.DataFrame()

    df = df[keep]
    df = df.apply(pd.to_numeric, errors="ignore")
    df["report_date_as_yyyy_mm_dd"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])

    if "commercial_long_all" in df.columns and "commercial_short_all" in df.columns:
        df["commercial_net"] = df["commercial_long_all"] - df["commercial_short_all"]

    return df.sort_values("report_date_as_yyyy_mm_dd").reset_index(drop=True)




# app.py (Chunk 2)

# ── METRICS ───────────────────────────────────────────────────────────────────
def add_rvol(df: pd.DataFrame, window: int) -> pd.DataFrame:
    out = df.copy()
    rolling_mean = out["volume"].rolling(window).mean().replace(0, np.nan)
    out["rvol"] = out["volume"] / rolling_mean
    return out

def merge_cot_price(cot: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    if price.empty or cot.empty:
        return pd.DataFrame()
    cot = cot.copy()
    cot["cot_date"] = cot["report_date_as_yyyy_mm_dd"] - pd.to_timedelta(
        cot["report_date_as_yyyy_mm_dd"].dt.weekday - 4, unit="D"
    )
    cot = cot[["cot_date", "commercial_long_all", "commercial_short_all"]]
    merged = pd.merge_asof(
        price.sort_values("date"),
        cot.sort_values("cot_date"),
        left_on="date",
        right_on="cot_date",
        direction="backward",
    )
    merged = merged.drop(columns="cot_date")
    total = (merged["commercial_long_all"] + merged["commercial_short_all"]).replace(0, np.nan)
    merged["cot_long_norm"]  = merged["commercial_long_all"]  / total
    merged["cot_short_norm"] = merged["commercial_short_all"] / total
    return merged

def add_health_gauge(df: pd.DataFrame,
                     weights: Dict[str, float] = {"rvol": .5, "cot_long": .3, "cot_short": .2}
                    ) -> pd.DataFrame:
    out = df.copy()
    out["health_gauge"] = (
        weights["rvol"]      * out["rvol"].fillna(1) +
        weights["cot_long"]  * out["cot_long_norm"].fillna(0) -
        weights["cot_short"] * out["cot_short_norm"].fillna(0)
    )
    return out

# ── FETCH ALL DATA ────────────────────────────────────────────────────────────
tickers = ASSET_LEADERS[category_selected]

with st.spinner("Downloading & crunching …"):
    # Yahoo prices in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        price_futs = {ex.submit(fetch_price_history, t): t for t in tickers}
        price_data = {price_futs[f]: f.result() for f in concurrent.futures.as_completed(price_futs)}

    # COT sequential
    cot_data = {t: fetch_cot_data(COT_ASSET_NAMES.get(t, ""), START_DATE, END_DATE)
                for t in tickers}

    # merge, rvol, health
    merged_data = {}
    for t in tickers:
        price = add_rvol(price_data[t], rvol_window)
        merged = merge_cot_price(cot_data[t], price)
        merged = add_health_gauge(merged)
        merged_data[t] = merged

# ── VISUALISATIONS ────────────────────────────────────────────────────────────
st.markdown("## Rolling Volatility")
for t in tickers:
    df = merged_data[t]
    if df.empty:
        st.warning(f"No data for {TICKER_TO_NAME[t]}")
        continue
    fig = px.line(df, x="date", y="rvol",
                  title=f"{TICKER_TO_NAME[t]} — RVol ({rvol_window}-day)",
                  template="plotly_white")
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

# ── RETURN DISTRIBUTION ───────────────────────────────────────────────────────
st.markdown("## Return Distribution")
df_results = pd.concat(
    [d.loc[d["return_pct"].notna(), ["return_pct"]].assign(asset=TICKER_TO_NAME[t])
     for t, d in merged_data.items() if not d.empty],
    ignore_index=True
)
if df_results.empty:
    st.info("No return data available.")
else:
    fig = px.histogram(df_results, x="return_pct", nbins=20,
                       color_discrete_sequence=["#3366CC"],
                       title="Daily % Return Distribution")
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

# ── HEALTH GAUGE OVER TIME ────────────────────────────────────────────────────
st.markdown("## Health Gauge")
for t in tickers:
    df = merged_data[t]
    if df.empty:
        continue
    fig = px.line(df, x="date", y="health_gauge",
                  title=f"{TICKER_TO_NAME[t]} — Health Gauge",
                  template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# ── PIP & RETURN DISTRIBUTION TREE ─────────────────────────────────────────────
def pip_and_return_tree_daily_pips(
    data: dict[str, pd.DataFrame], category: str, normalize: bool = True
) -> dict[str, dict]:
    tree: dict[str, dict] = {}
    for tkr, df in data.items():
        cat, asset_name = _category_for_ticker(tkr)
        if cat != category or df.empty:
            continue
        tmp = df.copy()
        tmp["date"] = pd.to_datetime(tmp["date"])
        tmp = tmp.sort_values("date")
        tmp["month_num"] = tmp["date"].dt.month
        tmp["daily_pip"] = tmp["high"] - tmp["low"]
        tmp["return"] = tmp["close"].pct_change() * 100
        tmp_returns = tmp.dropna(subset=["return"])
        tmp_pips = tmp.dropna(subset=["daily_pip"])
        month_dict: dict[str, dict] = {}

        for m in range(1, 13):
            month_data_pip = tmp_pips[tmp_pips["month_num"] == m]
            month_data_ret = tmp_returns[tmp_returns["month_num"] == m]
            if month_data_pip.empty and month_data_ret.empty:
                continue

            pip_stats = {
                "min": round(month_data_pip["daily_pip"].min(), 4) if not month_data_pip.empty else None,
                "max": round(month_data_pip["daily_pip"].max(), 4) if not month_data_pip.empty else None,
                "mean": round(month_data_pip["daily_pip"].mean(), 4) if not month_data_pip.empty else None,
                "count": int(month_data_pip["daily_pip"].count()) if not month_data_pip.empty else 0,
                "normalized": (month_data_pip["daily_pip"].count() / len(tmp_pips)) if normalize and not month_data_pip.empty else 0
            }

            return_stats = {
                "min": round(month_data_ret["return"].min(), 4) if not month_data_ret.empty else None,
                "max": round(month_data_ret["return"].max(), 4) if not month_data_ret.empty else None,
                "mean": round(month_data_ret["return"].mean(), 4) if not month_data_ret.empty else None,
                "count": int(month_data_ret["return"].count()) if not month_data_ret.empty else 0,
                "normalized": (month_data_ret["return"].count() / len(tmp_returns)) if normalize and not month_data_ret.empty else 0
            }

            month_name = calendar.month_abbr[m]
            month_dict[month_name] = {"daily_pip": pip_stats, "return": return_stats}

        # Aggregate node
        agg_pip = {
            "min": round(tmp_pips["daily_pip"].min(), 4),
            "max": round(tmp_pips["daily_pip"].max(), 4),
            "mean": round(tmp_pips["daily_pip"].mean(), 4),
            "count": int(tmp_pips["daily_pip"].count()),
            "normalized": 1.0 if normalize else None
        }
        agg_return = {
            "min": round(tmp_returns["return"].min(), 4),
            "max": round(tmp_returns["return"].max(), 4),
            "mean": round(tmp_returns["return"].mean(), 4),
            "count": int(tmp_returns["return"].count()),
            "normalized": 1.0 if normalize else None
        }
        month_dict["Aggregate"] = {"daily_pip": agg_pip, "return": agg_return}

        tree[asset_name] = month_dict
    return tree

# ── JSON DOWNLOAD ─────────────────────────────────────────────────────────────
st.markdown("## Download Merged Data & Distribution")
export_cols = [
    "date", "open", "high", "low", "close", "volume",
    "return_pct", "rvol",
    "commercial_long_all", "commercial_short_all",
    "cot_long_norm", "cot_short_norm", "health_gauge"
]

payload = {
    t: d[export_cols].round(6).fillna(None).to_dict(orient="records")
    for t, d in merged_data.items() if not d.empty
}

st.download_button(
    "Download JSON (Merged Data)",
    data=json.dumps(payload, indent=2, default=str),
    file_name=f"health_gauge_daily_pip_stats_{category_selected}.json",
    mime="application/json",
)

# ── JSON DOWNLOAD: Daily Pip & Return Tree ────────────────────────────────────
pip_return_tree = pip_and_return_tree_daily_pips(
    merged_data, category_selected, normalize=normalize_monthly_checkbox
)

st.download_button(
    "Download JSON (Daily Pip & Return Distribution)",
    data=json.dumps(pip_return_tree, indent=2, default=str),
    file_name=f"daily_pip_return_distribution_{category_selected}.json",
    mime="application/json",
)