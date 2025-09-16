# streamlit_exchange_links.py
import streamlit as st
from datetime import datetime

# --- Exchange URL templates ---
EXCHANGE_URL_TEMPLATES = {
    'NYSE': 'https://www.nyse.com/api/quotes/filter?startDate={start_date}&endDate={end_date}',
    'NASDAQ': 'https://www.nasdaqtrader.com/Trader.aspx?id=MarketActivity&startDate={start_date}&endDate={end_date}',
    'LSE': 'https://www.londonstockexchange.com/statistics?startDate={start_date}&endDate={end_date}',
    'CME': 'https://www.cmegroup.com/market-data/daily-volume.html?startDate={start_date}&endDate={end_date}',
    'HKEX': 'https://www.hkex.com.hk/Market-Data/Statistics?startDate={start_date}&endDate={end_date}'
}

# --- Helper function to generate URLs ---
def generate_exchange_urls(start_date: str, end_date: str):
    urls = {}
    for exchange, template in EXCHANGE_URL_TEMPLATES.items():
        urls[exchange] = template.format(start_date=start_date, end_date=end_date)
    return urls

# --- Streamlit UI ---
st.title("Exchange Report URL Generator (Last 10 Years)")

st.markdown("""
Select the start and end dates to generate the report URLs for major exchanges.
""")

# Date input widgets
start_date = st.date_input("Start Date", value=datetime(2015, 9, 16))
end_date = st.date_input("End Date", value=datetime.today())

# Button to generate URLs
if st.button("Generate URLs"):
    # Convert to string in YYYY-MM-DD format
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    urls = generate_exchange_urls(start_str, end_str)
    
    for exchange, url in urls.items():
        st.subheader(exchange)
        st.markdown(f"[{url}]({url})")