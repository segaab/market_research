# streamlit_app.py
import streamlit as st
import pandas as pd
from pathlib import Path
import yaml
from datetime import datetime
from backtest_engine import BacktestEngine
from utils.data_loader import DataLoader
from utils.exports import save_backtest_yaml

# --- Presets
PRESETS = {
    "GFC": {"start": "2008-09-15", "end": "2009-03-31", "label": "GFC (Sep 15, 2008 → Mar 31, 2009)"},
    "COVID": {"start": "2020-02-19", "end": "2020-04-30", "label": "COVID crash (Feb 19, 2020 → Apr 30, 2020)"},
    "EURO": {"start": "2011-05-01", "end": "2012-09-30", "label": "Eurozone Debt Crisis (May 2011 → Sep 2012)"},
    "ASIA97": {"start": "1997-07-01", "end": "1998-12-31", "label": "Asian Financial Crisis (Jul 1997 → Dec 1998)"},
    "DOTCOM": {"start": "2000-03-01", "end": "2002-12-31", "label": "Dot-com Bust (Mar 2000 → Dec 2002)"},
}

st.set_page_config(layout="wide", page_title="Backtest: Short-only Crisis Presets")
st.title("Short-only Crisis Backtest Dashboard")

# Sidebar controls
with st.sidebar:
    st.header("Backtest controls")
    preset = st.selectbox("Preset", options=["Custom"] + list(PRESETS.keys()), index=1)
    if preset == "Custom":
        start = st.date_input("Start date", value=datetime(2020, 2, 19))
        end = st.date_input("End date", value=datetime(2020, 4, 30))
    else:
        start = pd.to_datetime(PRESETS[preset]["start"]).date()
        end = pd.to_datetime(PRESETS[preset]["end"]).date()
        st.write(f"Preset: {PRESETS[preset]['label']}")

    # Sectors (short-only stress proxies)
    st.subheader("Select sectors (short-only)")
    sectors = st.multiselect(
        "Sectors to include",
        options=["Energy", "Financials", "Industrials", "Consumer Discretionary", "Materials", "Real Estate", "Technology"],
        default=["Energy", "Financials", "Real Estate"]
    )

    strategy = st.selectbox("Strategy", options=["momentum_breakdown_short_v1", "crash_moment_short_v1", "liquidation_survivor_test_v1"])
    initial_cap = st.number_input("Initial capital (USD)", value=1_000_000)
    slippage_bp = st.number_input("Slippage (bps)", value=20)
    run_bt = st.button("Run backtest")

# Main area
if run_bt:
    st.info("Loading data and running backtest — this may take a few seconds")

    # Data loader will create synthetic data if none exists
    data_root = Path("data/placeholder_parquet")
    dl = DataLoader(data_root)
    asset_map = {
        "Energy": ["CL"],
        "Financials": ["XLF"],
        "Industrials": ["XLI"],
        "Consumer Discretionary": ["XLY"],
        "Materials": ["XLB"],
        "Real Estate": ["XLRE"],
        "Technology": ["NQ"]
    }

    selected_assets = [a for s in sectors for a in asset_map.get(s, [])]
    if not selected_assets:
        st.error("Select at least one sector")
    else:
        # load each asset's OHLCV
        price_data = {asset: dl.load_or_generate(asset, pd.to_datetime(start), pd.to_datetime(end)) for asset in selected_assets}

        engine = BacktestEngine(initial_capital=initial_cap, slippage_bps=slippage_bp)
        bt_res = engine.run(price_data, strategy_name=strategy, start=pd.to_datetime(start), end=pd.to_datetime(end))

        # Show summary metrics
        st.subheader("Backtest summary")
        metrics_df = pd.DataFrame([bt_res["metrics"]])
        st.table(metrics_df)

        st.subheader("Equity curve")
        eq = bt_res["equity_series"].reset_index()
        eq.columns = ["date", "equity"]
        st.line_chart(eq.set_index("date"))

        st.subheader("Trades")
        trades = bt_res["trades"]
        st.dataframe(trades)

        # Save YAML doc
        doc = {
            "run_date": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "preset": preset if preset != "Custom" else "Custom",
            "scenario": PRESETS[preset]["label"] if preset != "Custom" else "Custom range",
            "date_range": {"start": str(start), "end": str(end)},
            "strategy": strategy,
            "params": {"initial_capital": initial_cap, "slippage_bps": slippage_bp, "sectors": sectors}
        }
        yml_path = Path("docs") / f"{datetime.utcnow().strftime('%Y%m%d')}_{preset}_{strategy}.yml"
        yml_path.parent.mkdir(parents=True, exist_ok=True)
        save_backtest_yaml(doc, yml_path)
        st.success(f"Saved backtest doc to {yml_path}")

        # Export one-page PDF placeholder
        if st.button("Export one-page PDF (placeholder)"):
            pdf_path = Path("docs") / f"{datetime.utcnow().strftime('%Y%m%d')}_{preset}_{strategy}.pdf"
            from utils.exports import export_one_page_pdf
            export_one_page_pdf(metrics=bt_res['metrics'], equity_series=bt_res['equity_series'], pdf_path=pdf_path)
            st.success(f"PDF exported to {pdf_path}")