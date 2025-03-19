import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fetch_klines(symbol, interval, start_time, end_time):
    """
    Fetch kline data from Binance API
    """
    base_url = "https://api.binance.com/api/v3/klines"
    
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': int(start_time.timestamp() * 1000),
        'endTime': int(end_time.timestamp() * 1000),
        'limit': 1000
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data: {e}")
        return None

def process_klines(klines):
    """
    Process kline data into a pandas DataFrame
    """
    if not klines:
        return None
    
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Convert string values to float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    # Select only required columns
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    return df

def main():
    # Parameters
    symbol = 'BTCUSDT'
    interval = '1h'
    start_date = datetime(2023, 3, 18)
    end_date = datetime(2025, 3, 18)
    
    # Initialize empty DataFrame to store all data
    all_data = pd.DataFrame()
    
    # Current time window for fetching
    current_start = start_date
    current_end = min(current_start + timedelta(days=30), end_date)
    
    while current_start < end_date:
        logger.info(f"Fetching data from {current_start} to {current_end}")
        
        # Fetch klines data
        klines = fetch_klines(symbol, interval, current_start, current_end)
        
        if klines:
            # Process the data
            df = process_klines(klines)
            if df is not None and not df.empty:
                all_data = pd.concat([all_data, df], ignore_index=True)
                logger.info(f"Successfully fetched {len(df)} records")
            else:
                logger.warning("No data received for the current time window")
        
        # Move to next time window
        current_start = current_end
        current_end = min(current_start + timedelta(days=30), end_date)
        
        # Add delay to avoid rate limiting
        time.sleep(1)
    
    # Save to CSV
    if not all_data.empty:
        output_file = f"btc_historical_data_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        all_data.to_csv(output_file, index=False)
        logger.info(f"Data saved to {output_file}")
    else:
        logger.error("No data was collected")

if __name__ == "__main__":
    main() 