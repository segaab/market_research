# HealthGaugeSeasonality.py
# ─────────────────────────────────────────────────────────────────────────────
import os
import calendar
import numpy as np
import pandas as pd
import streamlit as st
from sodapy import Socrata
from yahooquery import Ticker
from concurrent.futures import ThreadPoolExecutor

# ── Globals ──────────────────────────────────────────────────────────────────
START_DATE = "2000-01-01"
END_DATE = pd.Timestamp.today().strftime("%Y-%m-%d")

# Socrata client for CFTC API
client = Socrata("publicreporting.cftc.gov", None)

COT_ASSET_NAMES = {
    # Indices
    "^GSPC": "S&P 500 – Chicago Mercantile Exchange",   # S&P 500
    "^GDAXI": None,  # DAX not directly reported in CFTC COT

    # Forex
    "EURUSD=X": "EURO FX – Chicago Mercantile Exchange",
    "USDJPY=X": "JAPANESE YEN – Chicago Mercantile Exchange",

    # Commodities
    "ZS=F": "SOYBEANS – Chicago Board of Trade",       # Soybeans
    "CL=F": "WTI-PHYSICAL – New York Mercantile Exchange",  # Crude Oil
    "GC=F": "GOLD – Commodity Exchange Inc."          # Gold
}

# ── Data fetchers ────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_history(ticker: str) -> pd.DataFrame:
    """Daily OHLCV from Yahoo; cached for 1 h."""
    yq = Ticker(ticker)
    df = yq.history(start=START_DATE, end=END_DATE)
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    df = df.rename(columns={"symbol": "ticker"})
    return df[["date", "ticker", "open", "high", "low", "close", "volume"]]

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_cot_data(market_name: str) -> pd.DataFrame:
    """COT report data from CFTC Socrata API; cached for 1 h."""
    if not market_name:
        return pd.DataFrame()
    results = client.get("6dca-aqww", where=f"market_and_exchange_names='{market_name}'")
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame.from_records(results)
    if "report_date_as_yyyy_mm_dd" in df.columns:
        df["report_date_as_yyyy_mm_dd"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
    return df

def merge_cot_price(cot: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    if price.empty or cot.empty:
        return pd.DataFrame()

    cot = cot.copy()

    # Ensure timestamp column exists
    if "timestamp" not in cot.columns:
        if "report_date_as_yyyy_mm_dd" in cot.columns:
            cot["timestamp"] = pd.to_datetime(cot["report_date_as_yyyy_mm_dd"])
        else:
            return pd.DataFrame()

    # Align Friday release to trading day
    cot["cot_date"] = cot["timestamp"] - pd.to_timedelta(
        cot["timestamp"].dt.weekday - 4, unit="D"
    )

    # Keep only available columns
    keep_cols = ["cot_date"]
    if "commercial_long_all" in cot.columns:
        keep_cols.append("commercial_long_all")
    if "commercial_short_all" in cot.columns:
        keep_cols.append("commercial_short_all")

    cot = cot[keep_cols]

    merged = pd.merge_asof(
        price.sort_values("date"),
        cot.sort_values("cot_date"),
        left_on="date", right_on="cot_date", direction="backward"
    ).drop(columns="cot_date")

    if {"commercial_long_all", "commercial_short_all"} <= set(merged.columns):
        tot = (merged["commercial_long_all"] + merged["commercial_short_all"]).replace(0, np.nan)
        merged["cot_long_norm"]  = merged["commercial_long_all"]  / tot
        merged["cot_short_norm"] = merged["commercial_short_all"] / tot
    else:
        merged["cot_long_norm"] = np.nan
        merged["cot_short_norm"] = np.nan

    return merged

# ── METRICS & HELPERS ────────────────────────────────────────────────────────
def add_rvol(df: pd.DataFrame, window: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["volume"] = pd.to_numeric(out.get("volume", pd.Series(dtype=float)), errors="coerce").fillna(0)
    rolling_mean = out["volume"].rolling(window, min_periods=1).mean().replace(0, np.nan)
    out["rvol"] = out["volume"] / rolling_mean
    return out

def add_health_gauge(df: pd.DataFrame,
                     weights: Dict[str, float] = {"rvol": .5, "cot_long": .3, "cot_short": .2}
                    ) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    # Ensure the normalized columns exist
    out["cot_long_norm"]  = out.get("cot_long_norm", pd.Series(np.nan, index=out.index))
    out["cot_short_norm"] = out.get("cot_short_norm", pd.Series(np.nan, index=out.index))
    out["rvol"] = out.get("rvol", pd.Series(1.0, index=out.index)).fillna(1.0)
    out["health_gauge"] = (
        weights.get("rvol", 0.5) * out["rvol"].fillna(1.0) +
        weights.get("cot_long", 0.3) * out["cot_long_norm"].fillna(0.0) -
        weights.get("cot_short", 0.2) * out["cot_short_norm"].fillna(0.0)
    )
    return out

# ── UI: ensure sidebar options exist (if not defined earlier) ────────────────
# (These are no-ops if the variables already exist in your top-of-file code.)
try:
    category_selected
except NameError:
    category_selected = list(ASSET_LEADERS.keys())[0]

try:
    rvol_window
except NameError:
    rvol_window = 20

try:
    normalize_monthly_checkbox
except NameError:
    normalize_monthly_checkbox = True

# ── FETCH ALL DATA (parallel prices, sequential COT) ─────────────────────────
tickers = ASSET_LEADERS.get(category_selected, [])
if not tickers:
    st.warning("No tickers found for selected category.")
else:
    with st.spinner("Downloading & crunching …"):
        # Fetch price histories in parallel
        price_data: Dict[str, pd.DataFrame] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            fut_map = {ex.submit(fetch_price_history, t): t for t in tickers}
            for fut in concurrent.futures.as_completed(fut_map):
                t = fut_map[fut]
                try:
                    price_data[t] = fut.result()
                except Exception as e:
                    st.warning(f"Price fetch failed for {t}: {e}")
                    price_data[t] = pd.DataFrame()

        # Fetch COT (cached) sequentially
        cot_data: Dict[str, pd.DataFrame] = {}
        for t in tickers:
            cot_name = COT_ASSET_NAMES.get(t, None)
            try:
                cot_data[t] = fetch_cot_data(cot_name) if cot_name else pd.DataFrame()
            except Exception as e:
                st.warning(f"COT fetch failed for {t}: {e}")
                cot_data[t] = pd.DataFrame()

        # Merge, compute rvol and health gauge
        merged_data: Dict[str, pd.DataFrame] = {}
        for t in tickers:
            price_df = price_data.get(t, pd.DataFrame())
            if price_df.empty:
                merged_data[t] = pd.DataFrame()
                continue

            price_with_rvol = add_rvol(price_df, rvol_window)
            merged = merge_cot_price(cot_data.get(t, pd.DataFrame()), price_with_rvol)

            # If merge returned empty (no cot), use price_with_rvol as basis
            if merged.empty:
                merged = price_with_rvol.copy()

            merged = add_health_gauge(merged)
            merged_data[t] = merged

# ── DAILY PIP & RETURN DISTRIBUTION TREE ─────────────────────────────────────
def pip_and_return_tree_daily_pips(
    data: Dict[str, pd.DataFrame], category: str, normalize: bool = True
) -> Dict[str, Dict]:
    tree: Dict[str, Dict] = {}
    for tkr, df in data.items():
        cat, asset_name = _category_for_ticker(tkr)
        if cat != category or df.empty:
            continue

        tmp = df.copy()
        # Accept either 'date' or 'timestamp' as main time column
        if "date" in tmp.columns:
            tmp["date"] = pd.to_datetime(tmp["date"])
        elif "timestamp" in tmp.columns:
            tmp["date"] = pd.to_datetime(tmp["timestamp"])
        else:
            # no date-like column, skip
            continue

        tmp = tmp.sort_values("date")
        tmp["month_num"] = tmp["date"].dt.month

        # daily pip range (high-low)
        if "high" not in tmp.columns or "low" not in tmp.columns:
            # If high/low missing, attempt to derive from close (zero pip)
            tmp["daily_pip"] = 0.0
        else:
            tmp["daily_pip"] = tmp["high"] - tmp["low"]

        # daily returns %
        tmp["return"] = tmp["close"].pct_change() * 100

        tmp_pips = tmp.dropna(subset=["daily_pip"])
        tmp_returns = tmp.dropna(subset=["return"])

        month_dict: Dict[str, Dict] = {}

        # iterate months Jan..Dec
        for m in range(1, 13):
            pip_m = tmp_pips.loc[tmp_pips["month_num"] == m]
            ret_m = tmp_returns.loc[tmp_returns["month_num"] == m]

            if pip_m.empty and ret_m.empty:
                # still include empty month with None values for clarity
                month_dict[calendar.month_abbr[m]] = {
                    "daily_pip": {"min": None, "max": None, "mean": None, "count": 0, "normalized": 0 if normalize else 0},
                    "return":    {"min": None, "max": None, "mean": None, "count": 0, "normalized": 0 if normalize else 0},
                }
                continue

            pip_stats = {
                "min": round(pip_m["daily_pip"].min(), 4) if not pip_m.empty else None,
                "max": round(pip_m["daily_pip"].max(), 4) if not pip_m.empty else None,
                "mean": round(pip_m["daily_pip"].mean(), 4) if not pip_m.empty else None,
                "count": int(pip_m["daily_pip"].count()) if not pip_m.empty else 0,
                "normalized": (pip_m["daily_pip"].count() / len(tmp_pips)) if (normalize and not pip_m.empty and len(tmp_pips)>0) else (pip_m["daily_pip"].count() if not normalize else 0)
            }

            ret_stats = {
                "min": round(ret_m["return"].min(), 4) if not ret_m.empty else None,
                "max": round(ret_m["return"].max(), 4) if not ret_m.empty else None,
                "mean": round(ret_m["return"].mean(), 4) if not ret_m.empty else None,
                "count": int(ret_m["return"].count()) if not ret_m.empty else 0,
                "normalized": (ret_m["return"].count() / len(tmp_returns)) if (normalize and not ret_m.empty and len(tmp_returns)>0) else (ret_m["return"].count() if not normalize else 0)
            }

            month_dict[calendar.month_abbr[m]] = {"daily_pip": pip_stats, "return": ret_stats}

        # Aggregate (raw values; normalized flag kept but not used for raw counts)
        agg_pips = tmp_pips["daily_pip"] if not tmp_pips.empty else pd.Series(dtype=float)
        agg_returns = tmp_returns["return"] if not tmp_returns.empty else pd.Series(dtype=float)

        month_dict["Aggregate"] = {
            "daily_pip": {
                "min": round(agg_pips.min(), 4) if not agg_pips.empty else None,
                "max": round(agg_pips.max(), 4) if not agg_pips.empty else None,
                "mean": round(agg_pips.mean(), 4) if not agg_pips.empty else None,
                "count": int(agg_pips.count()),
                "normalized": int(agg_pips.count())  # raw count
            },
            "return": {
                "min": round(agg_returns.min(), 4) if not agg_returns.empty else None,
                "max": round(agg_returns.max(), 4) if not agg_returns.empty else None,
                "mean": round(agg_returns.mean(), 4) if not agg_returns.empty else None,
                "count": int(agg_returns.count()),
                "normalized": int(agg_returns.count())  # raw count
            }
        }

        tree[asset_name] = month_dict

    return tree

# ── RETURN-ACCURACY TREE ─────────────────────────────────────────────────────
def return_accuracy_tree(
    data: Dict[str, pd.DataFrame],
    category: str,
    seasonal_map: Dict[str, Dict]
) -> Dict[str, Dict]:
    sign_to_emoji = {1: "✅", 0: "⚪", -1: "❌"}
    tree: Dict[str, Dict] = {}

    top_key = "Commodities" if category in ("Agricultural", "Energy", "Metals") else category

    for tkr, df in data.items():
        _, asset = _category_for_ticker(tkr)
        if df.empty or asset not in seasonal_map.get(top_key, {}):
            continue

        tmp = df.copy()
        if "date" in tmp.columns:
            tmp["month_num"] = pd.to_datetime(tmp["date"]).dt.month
        elif "timestamp" in tmp.columns:
            tmp["month_num"] = pd.to_datetime(tmp["timestamp"]).dt.month
        else:
            continue

        month_dict: Dict[str, Dict] = {}
        for m in range(1, 13):
            ret_m = tmp.loc[tmp["month_num"] == m, "return_pct"].dropna()
            if ret_m.empty:
                continue
            mean_ret = float(ret_m.mean())
            obs_sign = 1 if mean_ret > 0 else -1 if mean_ret < 0 else 0
            observed = sign_to_emoji[obs_sign]
            expected = seasonal_map[top_key][asset][calendar.month_abbr[m]]
            month_dict[calendar.month_abbr[m]] = {
                "expected": expected,
                "observed": observed,
                "accurate": expected == observed
            }
        if month_dict:
            tree[asset] = month_dict
    return tree

# ── DOWNLOADS & UI OUTPUT ────────────────────────────────────────────────────
st.markdown("## Download JSON files")

export_cols = [
    "date", "open", "high", "low", "close", "volume",
    "return_pct", "rvol",
    "commercial_long_all", "commercial_short_all",
    "cot_long_norm", "cot_short_norm", "health_gauge"
]

payload_merged = {
    t: d[export_cols].round(6).fillna(None).to_dict(orient="records")
    for t, d in merged_data.items() if not d.empty
}

st.download_button(
    "Download JSON (Merged Data)",
    json.dumps(payload_merged, indent=2, default=str),
    file_name=f"health_gauge_merged_{category_selected.lower()}.json",
    mime="application/json"
)

pip_return_tree = pip_and_return_tree_daily_pips(merged_data, category_selected, normalize=normalize_monthly_checkbox)
st.download_button(
    "Download JSON (Daily Pip & Return Distribution)",
    json.dumps(pip_return_tree, indent=2, default=str),
    file_name=f"daily_pip_return_{category_selected.lower()}.json",
    mime="application/json"
)

accuracy_tree = return_accuracy_tree(merged_data, category_selected, ProfitableSeasonalMap)
st.download_button(
    "Download JSON (Return-Accuracy)",
    json.dumps(accuracy_tree, indent=2, default=str),
    file_name=f"return_accuracy_{category_selected.lower()}.json",
    mime="application/json"
)

# ── Preview distribution tree in UI ──────────────────────────────────────────
st.markdown("## Daily Pip & Return Distribution Tree")
pip_tree = pip_return_tree
if pip_tree:
    for asset, months in pip_tree.items():
        st.markdown(f"### {asset}")
        st.json(months)
else:
    st.info("No distribution data available for the selected category.")