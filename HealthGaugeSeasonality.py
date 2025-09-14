#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Multi-Asset Health Gauge  ·  threaded, batched CFTC downloads (10-year proof)
# -----------------------------------------------------------------------------
import json, time, calendar, math, threading, logging, random
from datetime import datetime, timedelta
from typing import Dict, List, DefaultDict, Tuple, Callable, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sodapy import Socrata
from requests.exceptions import HTTPError, ConnectionError, Timeout, RequestException
from yahooquery import Ticker

# ── APP CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Multi-Asset Health Gauge", layout="wide")

LATEST_DATE = datetime.today()
START_DATE  = (LATEST_DATE - timedelta(days=365 * 10)).strftime("%Y-%m-%d")
END_DATE    = LATEST_DATE.strftime("%Y-%m-%d")

MAX_WORKERS        = 4
MAX_RETRIES        = 4
RETRY_BACKOFF_SECS = 1.5
COT_PAGE_SIZE      = 15_000
YH_SLEEP           = 0.15

# ── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

# ── GLOBAL LOG STORAGE ─────────────────────────────────────────────────────
log_messages: List[str] = []

def log_and_store(level: str, msg: str):
    log_messages.append(f"[{level}] {msg}")
    if level == "INFO":
        logging.info(msg)
    elif level == "WARNING":
        logging.warning(msg)
    else:
        logging.error(msg)

# ── MAPPINGS ─────────────────────────────────────────────────────────────────
ASSET_LEADERS: Dict[str, List[str]] = {
    "Indices": ["^GSPC"],
    "Forex": ["EURUSD=X", "USDJPY=X"],
    "Agricultural": ["ZS=F"],
    "Energy": ["CL=F"],
    "Metals": ["GC=F"],
}
TICKER_TO_NAME = {
    "^GSPC": "S&P 500", "EURUSD=X": "EUR/USD", "USDJPY=X": "USD/JPY",
    "ZS=F": "Soybeans", "CL=F": "WTI Crude", "GC=F": "Gold",
}
COT_ASSET_NAMES = {
    "^GSPC": "S&P 500 -- Chicago Mercantile Exchange",
    "EURUSD=X": "EURO FX -- Chicago Mercantile Exchange",
    "USDJPY=X": "JAPANESE YEN -- Chicago Mercantile Exchange",
    "ZS=F": "SOYBEANS -- Chicago Board of Trade",
    "CL=F": "WTI-PHYSICAL -- New York Mercantile Exchange",
    "GC=F": "GOLD -- Commodity Exchange Inc.",
}

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
category_selected = st.sidebar.selectbox("Choose Asset Category", ASSET_LEADERS.keys())
rvol_window       = st.sidebar.number_input("RVol Rolling Window (days)", 5, 60, 20)

# ── EXPONENTIAL BACKOFF HELPER ──────────────────────────────────────────────
def fetch_with_backoff(func: Callable, *args: Any, max_retries: int = 5, 
                      base_delay: float = 1.0, **kwargs: Any) -> Any:
    retries = 0
    while True:
        try:
            return func(*args, **kwargs)
        except (ConnectionError, Timeout, RequestException) as e:
            retries += 1
            if retries > max_retries:
                log_and_store("ERROR", f"Max retries ({max_retries}) exceeded: {e}")
                raise
            delay = base_delay * (2 ** (retries - 1)) * (0.5 + random.random())
            log_and_store("WARNING", f"Request failed (attempt {retries}/{max_retries}): {e}. Retrying in {delay:.2f}s")
            time.sleep(delay)

# ── FETCHING HELPERS ─────────────────────────────────────────────────────────
SODAPY_APP_TOKEN = "WSCaavlIcDgtLVZbJA1FKkq40"

@st.cache_resource(show_spinner=False)
def _yq_session(symbols: List[str]) -> Ticker:
    log_and_store("INFO", f"Initializing YahooQuery session for: {symbols}")
    return Ticker(symbols, asynchronous=False)

@st.cache_resource(show_spinner=False)
def _cot_client() -> Socrata:
    log_and_store("INFO", "Initializing Socrata CFTC client")
    return Socrata("publicreporting.cftc.gov", SODAPY_APP_TOKEN, timeout=120)

@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_price_history(ticker: str) -> pd.DataFrame:
    log_and_store("INFO", f"Fetching price history for {ticker}")
    def _fetch():
        df = _yq_session([ticker]).history(start=START_DATE, end=END_DATE)
        if df.empty:
            log_and_store("WARNING", f"No price data found for {ticker}")
            return pd.DataFrame()
        df = df.reset_index()
        df["timestamp"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        df["return_pct"] = df["close"].pct_change() * 100
        df["pip_range"] = df["high"] - df["low"]
        time.sleep(YH_SLEEP)
        return df[["timestamp","open","high","low","close","volume","return_pct","pip_range"]]
    return fetch_with_backoff(_fetch, max_retries=MAX_RETRIES, base_delay=RETRY_BACKOFF_SECS)

# ── THREADED CFTC FETCH ─────────────────────────────────────────────────────
def _month_chunks(start: str, end: str) -> List[Tuple[str, str]]:
    s = pd.to_datetime(start).to_period("M")
    e = pd.to_datetime(end).to_period("M")
    return [(str(p.start_time.date()), str(p.end_time.date())) for p in pd.period_range(s, e, freq="M")]

def _fetch_cot_chunk(cot_name: str, date_pair: Tuple[str, str], client: Socrata) -> pd.DataFrame:
    sd, ed = date_pair
    log_and_store("INFO", f"Fetching COT chunk for {cot_name} from {sd} to {ed}")
    where_clause = (
        f"market_and_exchange_names='{cot_name}' AND "
        f"report_date_as_yyyy_mm_dd >= '{sd}' AND report_date_as_yyyy_mm_dd <= '{ed}'"
    )
    def _do_fetch():
        rows, offset = [], 0
        while True:
            batch = client.get(
                "6dca-aqww",
                where=where_clause,
                order="report_date_as_yyyy_mm_dd",
                limit=COT_PAGE_SIZE,
                offset=offset,
            )
            if not batch:
                break
            rows.extend(batch)
            offset += COT_PAGE_SIZE
            if len(batch) < COT_PAGE_SIZE:
                break
        return pd.DataFrame.from_records(rows)
    return fetch_with_backoff(_do_fetch, max_retries=MAX_RETRIES, base_delay=RETRY_BACKOFF_SECS)

@st.cache_data(show_spinner=False, ttl=24 * 3600)
def fetch_cot_data(cot_name: str, start: str, end: str) -> pd.DataFrame:
    if not cot_name:
        return pd.DataFrame()
    client = _cot_client()
    results = []

    def worker(pair):
        try:
            return _fetch_cot_chunk(cot_name, pair, client)
        except Exception as e:
            log_and_store("ERROR", f"CFTC API failed for chunk {pair}: {e}")
            st.error(f"CFTC API failed for chunk {pair}: {e}")
            return pd.DataFrame()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = {exe.submit(worker, p): p for p in _month_chunks(start, end)}
        for fut in as_completed(futures):
            results.append(fut.result())
            log_and_store("INFO", f"Completed chunk {futures[fut]} for {cot_name}")

    df = pd.concat(results, ignore_index=True)
    if df.empty:
        log_and_store("WARNING", f"No COT data returned for {cot_name}")
        return df

    df["timestamp"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
    if {"commercial_long_all", "commercial_short_all"} <= set(df.columns):
        df["commercial_net"] = df["commercial_long_all"] - df["commercial_short_all"]

    daily = (
        df.set_index("timestamp")
          .sort_index()
          .reindex(pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="D"))
          .ffill()
          .reset_index()
          .rename(columns={"index": "timestamp"})
    )
    return daily.sort_values("timestamp").reset_index(drop=True)



# ── METRICS HELPERS ─────────────────────────────────────────────────────────
def add_rvol(df: pd.DataFrame, win: int) -> pd.DataFrame:
    out = df.copy()
    if "volume" in out.columns:
        out["rvol"] = out["volume"] / out["volume"].rolling(win).mean().replace(0, np.nan)
    else:
        log_and_store("WARNING", "'volume' column missing, cannot compute rvol")
        out["rvol"] = np.nan
    return out

def merge_cot_price(cot: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    if price.empty or cot.empty:
        log_and_store("WARNING", "Price or COT data is empty, merge skipped")
        return pd.DataFrame()
    cot2 = cot.copy()
    cot2["cot_date"] = cot2["timestamp"] - pd.to_timedelta(cot2["timestamp"].dt.weekday - 4, unit="D")
    cot2 = cot2[["cot_date", "commercial_long_all", "commercial_short_all"]]
    merged = pd.merge_asof(
        price.sort_values("timestamp"),
        cot2.sort_values("cot_date"),
        left_on="timestamp",
        right_on="cot_date",
        direction="backward",
    ).drop(columns="cot_date")
    tot = (merged["commercial_long_all"] + merged["commercial_short_all"]).replace(0, np.nan)
    merged["cot_long_norm"]  = merged["commercial_long_all"]  / tot
    merged["cot_short_norm"] = merged["commercial_short_all"] / tot
    return merged

def add_health_gauge(df: pd.DataFrame, w: Dict[str, float] = {"rvol": .5, "cot_long": .3, "cot_short": .2}) -> pd.DataFrame:
    out = df.copy()
    for col in ["rvol", "cot_long_norm", "cot_short_norm"]:
        if col not in out.columns:
            log_and_store("WARNING", f"Missing column '{col}' for health gauge calculation")
            out[col] = 0
    out["health_gauge"] = (
        w["rvol"]      * out["rvol"].fillna(1)
        + w["cot_long"]  * out["cot_long_norm"].fillna(0)
        - w["cot_short"] * out["cot_short_norm"].fillna(0)
    )
    return out

def compute_monthly_returns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return pd.DataFrame()
    df["month"] = df["timestamp"].dt.to_period("M")
    out = df.groupby("month")["return_pct"].sum().reset_index()
    out["month_name"] = out["month"].dt.strftime("%B")
    return out

def assign_profit_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    q1, q3 = df["return_pct"].quantile([.33, .77])
    df["profit_label"] = np.where(df["return_pct"] >= q3, "Profitable",
                          np.where(df["return_pct"] <= q1, "Unprofitable", "Average"))
    return df

def calculate_accuracy(df: pd.DataFrame, asset: str, cat: str) -> float:
    if df.empty: return 0.0
    top = "Commodities" if cat in ("Agricultural", "Energy", "Metals") else cat
    seasonal = ProfitableSeasonalMap.get(top, {}).get(asset, {})
    df["expected"] = df["month"].dt.month.map(lambda m: seasonal.get(calendar.month_abbr[m], "⚪"))
    df["hit"] = df.apply(
        lambda x: 1 if ((x["expected"] == "✅" and x["profit_label"] == "Profitable") or
                        (x["expected"] == "❌" and x["profit_label"] == "Unprofitable")) else 0, axis=1)
    return round(df["hit"].mean() * 100, 2)

# ── PROCESS CATEGORY ─────────────────────────────────────────────────────────
def process_category(category: str):
    tickers = ASSET_LEADERS[category]

    merged_data, monthly_distribution, accuracy_stats = {}, {}, {}
    daily_pip_reg : DefaultDict[pd.Timestamp, List[float]] = defaultdict(list)
    daily_hg_reg  : DefaultDict[pd.Timestamp, List[float]] = defaultdict(list)
    monthly_ret_reg  : DefaultDict[int, List[float]]       = defaultdict(list)
    monthly_acc_reg  : DefaultDict[int, List[float]]       = defaultdict(list)

    for tk in tickers:
        with st.spinner(f"Fetching data for {TICKER_TO_NAME[tk]}"):
            log_and_store("INFO", f"Processing {tk} ({TICKER_TO_NAME[tk]})")
            price = add_rvol(fetch_price_history(tk), rvol_window)
            cot   = fetch_cot_data(COT_ASSET_NAMES[tk], START_DATE, END_DATE)
            df    = add_health_gauge(merge_cot_price(cot, price))
            merged_data[tk] = df

        # ---- daily registries ----
        if not df.empty and "health_gauge" in df.columns and "pip_range" in df.columns:
            for _, r in df.iterrows():
                d = r["timestamp"].normalize()
                daily_hg_reg[d].append(round(r["health_gauge"], 4))
                daily_pip_reg[d].append(round(r["pip_range"], 4))

        # ---- monthly registries ----
        mdf = assign_profit_labels(compute_monthly_returns(df))
        monthly_distribution[tk] = mdf
        acc = calculate_accuracy(mdf, TICKER_TO_NAME[tk], category)
        accuracy_stats[tk] = acc
        for _, r in mdf.iterrows():
            m = r["month"].month
            monthly_ret_reg[m].append(round(r["return_pct"], 4))
            monthly_acc_reg[m].append(acc)

    # ── DAILY JSON ───────────────────────────────────────────────────────
    daily_json = [
        {
            "date": dt.strftime("%Y-%m-%d"),
            "avg_pip": float(np.mean(daily_pip_reg[dt])) if daily_pip_reg[dt] else None,
            "avg_hg": float(np.mean(daily_hg_reg[dt])) if daily_hg_reg[dt] else None,
        }
        for dt in sorted(daily_pip_reg.keys())
    ]
    log_and_store("INFO", f"Daily JSON entries -> {len(daily_json)}")

    # ── MONTHLY JSON ─────────────────────────────────────────────────────
    monthly_json = [
        {
            "month": month,
            "avg_return": float(np.mean(monthly_ret_reg[month])) if monthly_ret_reg[month] else None,
            "avg_acc": float(np.mean(monthly_acc_reg[month])) if monthly_acc_reg[month] else None,
        }
        for month in sorted(monthly_ret_reg.keys())
    ]
    log_and_store("INFO", f"Monthly JSON entries -> {len(monthly_json)}")

    # ── EXPORT JSON ──────────────────────────────────────────────────────
    with open(f"{category}_daily.json", "w") as f:
        json.dump(daily_json, f, indent=2)
    with open(f"{category}_monthly.json", "w") as f:
        json.dump(monthly_json, f, indent=2)
    log_and_store("INFO", f"Exported JSON for {category}")

    return merged_data, monthly_json, daily_json

# ── STREAMLIT UI ───────────────────────────────────────────────────────────
st.title("📊 Multi-Asset Health Gauge Dashboard (with Diagnostics)")

category_selected = st.selectbox("Choose Category", list(ASSET_LEADERS.keys()))
rvol_window = st.slider("Rolling Volatility Window", 5, 60, 20)

if st.button("Run Analysis"):
    merged_data, monthly_json, daily_json = process_category(category_selected)

    if merged_data:
        for tk, df in merged_data.items():
            if not df.empty and "health_gauge" in df.columns:
                st.line_chart(df["health_gauge"])
                break

    st.subheader("Daily JSON Preview")
    st.json(daily_json[:5])
    st.subheader("Monthly JSON Preview")
    st.json(monthly_json[:5])

    st.subheader("Diagnostic Log")
    for msg in log_messages:
        st.text(msg)

st.write("### Script Version: threaded CFTC fetch + JSON export + diagnostics")