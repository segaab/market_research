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

MAX_WORKERS        = 4            # parallel CFTC calls
MAX_RETRIES        = 4            # per-chunk retry attempts
RETRY_BACKOFF_SECS = 1.5

COT_PAGE_SIZE = 15_000           # Socrata max is 50 000
YH_SLEEP      = 0.15

# ── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

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
                logging.error(f"Max retries ({max_retries}) exceeded: {e}")
                raise
            delay = base_delay * (2 ** (retries - 1)) * (0.5 + random.random())
            logging.warning(f"Request failed (attempt {retries}/{max_retries}): {e}. Retrying in {delay:.2f} seconds")
            time.sleep(delay)

# ── FETCHING HELPERS ─────────────────────────────────────────────────────────
SODAPY_APP_TOKEN = "WSCaavlIcDgtLVZbJA1FKkq40"

@st.cache_resource(show_spinner=False)
def _yq_session(symbols: List[str]) -> Ticker:
    logging.info(f"Initializing YahooQuery session for: {symbols}")
    return Ticker(symbols, asynchronous=False)

@st.cache_resource(show_spinner=False)
def _cot_client() -> Socrata:
    logging.info("Initializing Socrata CFTC client")
    return Socrata("publicreporting.cftc.gov", SODAPY_APP_TOKEN, timeout=120)

@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_price_history(ticker: str) -> pd.DataFrame:
    logging.info(f"Fetching price history for {ticker}")
    
    def _fetch():
        df = _yq_session([ticker]).history(start=START_DATE, end=END_DATE)
        if df.empty:
            logging.warning(f"No price data found for {ticker}")
            return pd.DataFrame()
        df = df.reset_index()
        df["timestamp"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df["volume"]    = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        df["return_pct"] = df["close"].pct_change() * 100
        df["pip_range"]  = df["high"] - df["low"]
        time.sleep(YH_SLEEP)
        return df[["timestamp","open","high","low","close","volume","return_pct","pip_range"]]
    
    return fetch_with_backoff(_fetch, max_retries=MAX_RETRIES, base_delay=RETRY_BACKOFF_SECS)

# ───────────────────────  THREADED CFTC  ─────────────────────────────────────
def _month_chunks(start: str, end: str) -> List[Tuple[str, str]]:
    s = pd.to_datetime(start).to_period("M")
    e = pd.to_datetime(end).to_period("M")
    return [(str(p.start_time.date()), str(p.end_time.date())) for p in pd.period_range(s, e, freq="M")]

def _fetch_cot_chunk(cot_name: str, date_pair: Tuple[str, str], client: Socrata) -> pd.DataFrame:
    sd, ed = date_pair
    logging.info(f"Fetching COT chunk for {cot_name} from {sd} to {ed}")
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
            logging.error(f"CFTC API failed for chunk {pair}: {e}")
            st.error(f"CFTC API failed for chunk {pair}: {e}")
            return pd.DataFrame()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = {exe.submit(worker, p): p for p in _month_chunks(start, end)}
        for fut in as_completed(futures):
            results.append(fut.result())
            logging.info(f"Completed chunk {futures[fut]} for {cot_name}")

    df = pd.concat(results, ignore_index=True)
    if df.empty:
        logging.warning(f"No COT data returned for {cot_name}")
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
    if df.empty or "volume" not in df.columns:
        df["rvol"] = np.nan
        return df
    out = df.copy()
    out["rvol"] = out["volume"] / out["volume"].rolling(win).mean().replace(0, np.nan)
    return out

def merge_cot_price(cot: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    if price.empty or cot.empty:
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
    # Ensure columns exist to avoid KeyError
    for col in ["rvol", "cot_long_norm", "cot_short_norm"]:
        if col not in out.columns:
            out[col] = 0.0 if "cot" in col else 1.0
    out["health_gauge"] = (
        w["rvol"]      * out["rvol"].fillna(1)
        + w["cot_long"]  * out["cot_long_norm"].fillna(0)
        - w["cot_short"] * out["cot_short_norm"].fillna(0)
    )
    return out


# ── MONTHLY RETURNS & PROFIT LABELS ───────────────────────────────────────────
def compute_monthly_returns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "return_pct" not in df.columns:
        return pd.DataFrame()
    df["month"] = df["timestamp"].dt.to_period("M")
    out = df.groupby("month")["return_pct"].sum().reset_index()
    out["month_name"] = out["month"].dt.strftime("%B")
    return out

def assign_profit_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "return_pct" not in df.columns:
        return df
    q1, q3 = df["return_pct"].quantile([.33, .77])
    df["profit_label"] = np.where(df["return_pct"] >= q3, "Profitable",
                          np.where(df["return_pct"] <= q1, "Unprofitable", "Average"))
    return df

# ── ACCURACY CALCULATION ─────────────────────────────────────────────────────
def calculate_accuracy(df: pd.DataFrame, asset: str, cat: str) -> float:
    if df.empty or "month" not in df.columns or "return_pct" not in df.columns:
        return 0.0
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
            logging.info(f"Processing {tk} ({TICKER_TO_NAME[tk]})")
            price = add_rvol(fetch_price_history(tk), rvol_window)
            cot   = fetch_cot_data(COT_ASSET_NAMES[tk], START_DATE, END_DATE)
            df    = add_health_gauge(merge_cot_price(cot, price))
            merged_data[tk] = df

        # ---- daily registries ---------------------------------------------
        for _, r in df.iterrows():
            d = r["timestamp"].normalize()
            if not math.isnan(r.get("pip_range", np.nan)):
                daily_pip_reg[d].append(round(r["pip_range"], 4))
            daily_hg_reg[d].append(round(r.get("health_gauge", 0), 4))

        # ---- monthly registries -------------------------------------------
        mdf = assign_profit_labels(compute_monthly_returns(df))
        monthly_distribution[tk] = mdf
        acc = calculate_accuracy(mdf, TICKER_TO_NAME[tk], category)
        accuracy_stats[tk] = acc
        for _, r in mdf.iterrows():
            m = r["month"].month
            monthly_ret_reg[m].append(round(r.get("return_pct", 0), 4))
            monthly_acc_reg[m].append(acc)

    # ── DAILY JSON ----------------------------------------------------------
    daily_json = []
    for dt in sorted(daily_pip_reg):
        pips = daily_pip_reg[dt]
        hgs  = daily_hg_reg[dt]
        daily_json.append({
            "date": dt.strftime("%Y-%m-%d"),
            "month": dt.month,
            "pip_range_24h": pips,
            "daily_mean_pip": round(float(np.mean(pips)), 4),
            "daily_median_pip": round(float(np.median(pips)), 4),
            "sd_pip_24h": round(float(np.std(pips, ddof=0)), 4),
            "daily_health_gauge": round(float(np.mean(hgs)), 4),
        })

    # ── MONTHLY JSON --------------------------------------------------------
    monthly_json = []
    for m in range(1, 13):
        if m not in monthly_ret_reg: continue
        rets = monthly_ret_reg[m]
        monthly_json.append({
            "month": calendar.month_name[m],
            "monthly_return": rets,
            "accuracy_pct": round(float(np.mean(monthly_acc_reg[m])), 2),
            "mean_return": round(float(np.mean(rets)), 4),
            "median_return": round(float(np.median(rets)), 4),
            "sd_return": round(float(np.std(rets, ddof=0)), 4),
        })

    # ── DOWNLOAD BUTTONS ----------------------------------------------------
    st.download_button(
        "Download Daily Health Gauge + Pip Stats (JSON)",
        data=json.dumps({category: daily_json}, indent=2),
        file_name=f"daily_health_pip_{category}_{datetime.now():%Y%m%d_%H%M%S}.json",
        mime="application/json",
    )
    st.download_button(
        "Download Monthly Stats (JSON)",
        data=json.dumps({category: monthly_json}, indent=2),
        file_name=f"monthly_stats_{category}_{datetime.now():%Y%m%d_%H%M%S}.json",
        mime="application/json",
    )

    # ── VISUALS -------------------------------------------------------------
    for tk in tickers:
        df = merged_data[tk]
        if df.empty: continue
        st.plotly_chart(px.line(df, x="timestamp", y="health_gauge",
                                title=f"Health Gauge -- {TICKER_TO_NAME[tk]}"),
                        use_container_width=True)
        mdf = monthly_distribution[tk]
        if not mdf.empty:
            st.plotly_chart(px.bar(mdf, x=mdf['month'].dt.strftime('%Y-%m'),
                                   y="return_pct", color="profit_label",
                                   title=f"Monthly Return -- {TICKER_TO_NAME[tk]}"),
                            use_container_width=True)
        st.markdown(f"**Seasonality accuracy:** {accuracy_stats[tk]} %")
        logging.info(f"Finished visualizing {tk}")

# ── RUN APP ──────────────────────────────────────────────────────────────────
process_category(category_selected)
st.write("### Script Version : threaded CFTC fetch + daily & monthly JSON export")
logging.info("Completed processing category: " + category_selected)