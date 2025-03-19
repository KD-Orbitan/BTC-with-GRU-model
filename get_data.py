import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_klines(symbol, interval, start_time, end_time, max_retries=5, delay=5):
    """Fetch kline data from Binance API with retry mechanism"""
    base_url = "https://api.binance.com/api/v3/klines"
    
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': int(start_time.timestamp() * 1000),
        'endTime': int(end_time.timestamp() * 1000),
        'limit': 1000
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error fetching data: {e}. Retrying {attempt+1}/{max_retries}...")
            time.sleep(delay)
    
    logger.error(f"Failed to fetch data after {max_retries} attempts")
    return None

def process_klines(klines):
    """Process kline data into a pandas DataFrame"""
    if not klines:
        return None
    
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Convert numeric columns
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].astype(float)
    
    # Keep only required columns
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

def main():
    # Parameters
    symbol = 'BTCUSDT'
    interval = '1h'
    start_date = datetime(2023, 3, 18)
    end_date = datetime(2025, 3, 18)

    # Initialize empty DataFrame
    all_data = pd.DataFrame()

    # Fetch data in 41-day chunks to optimize API requests
    chunk_size = 41  # ~1000 candles for 1-hour interval
    current_start = start_date
    current_end = min(current_start + timedelta(days=chunk_size), end_date)

    while current_start < end_date:
        logger.info(f"Fetching data from {current_start} to {current_end}")

        # Fetch klines data
        klines = fetch_klines(symbol, interval, current_start, current_end)
        if klines:
            df = process_klines(klines)
            if df is not None and not df.empty:
                all_data = pd.concat([all_data, df], ignore_index=True)
                logger.info(f"Fetched {len(df)} records")
            else:
                logger.warning("No data received for the current time window")

        # Move to the next time window
        current_start = current_end
        current_end = min(current_start + timedelta(days=chunk_size), end_date)

        # Small delay to avoid hitting rate limits
        time.sleep(0.5)

    # Save to CSV
    if not all_data.empty:
        output_file = f"btc_historical_data_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        all_data.to_csv(output_file, index=False)
        logger.info(f"Data saved to {output_file}")
    else:
        logger.error("No data was collected")

if __name__ == "__main__":
    main()
