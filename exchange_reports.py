# streamlit_exchange_dashboard.py
import streamlit as st
from datetime import datetime, timedelta

# --- Exchange base URLs ---
EXCHANGE_BASE_URLS = {
    'NYSE': 'https://www.nyse.com/api/quotes/filter',
    'NASDAQ': 'https://www.nasdaqtrader.com/Trader.aspx?id=MarketActivity',
    'LSE': 'https://www.londonstockexchange.com/statistics',
    'CME': 'https://www.cmegroup.com/market-data/daily-volume.html',
    'HKEX': 'https://www.hkex.com.hk/Market-Data/Statistics'
}

# --- Generate date ranges (monthly chunks for 10 years) ---
def generate_date_ranges():
    end_date = datetime.today()
    start_date = end_date - timedelta(days=3650)  # ~10 years
    date_ranges = []
    while start_date < end_date:
        range_end = min(start_date + timedelta(days=30), end_date)
        date_ranges.append((start_date.strftime('%Y-%m-%d'), range_end.strftime('%Y-%m-%d')))
        start_date = range_end + timedelta(days=1)
    return date_ranges

# --- Generate URLs for historical reports ---
def generate_report_links(selected_exchanges):
    date_ranges = generate_date_ranges()
    exchange_links = {}

    for exchange, base_url in EXCHANGE_BASE_URLS.items():
        if exchange not in selected_exchanges:
            continue
        links = []
        for start_date, end_date in date_ranges:
            url = f"{base_url}?startDate={start_date}&endDate={end_date}"
            links.append(url)
        exchange_links[exchange] = links

    return exchange_links

# --- Streamlit UI ---
st.title("Historical Exchange Report Links (Last 10 Years)")

# Select exchanges
selected_exchanges = st.multiselect(
    "Select Exchanges",
    options=list(EXCHANGE_BASE_URLS.keys()),
    default=list(EXCHANGE_BASE_URLS.keys())
)

if st.button("Generate Links"):
    links_dict = generate_report_links(selected_exchanges)
    
    for exchange, links in links_dict.items():
        st.subheader(exchange)
        st.write(f"Total links: {len(links)}")
        # Show only the first 10 links for readability
        for link in links[:10]:
            st.markdown(f"[{link}]({link})")
        if len(links) > 10:
            st.write("... more links not displayed")

st.info("⚠️ Note: These URLs are placeholders. You may need to adjust the query parameters based on each exchange's archive URL structure.")