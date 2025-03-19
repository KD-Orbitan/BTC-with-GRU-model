import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_technical_indicators(df):
    """
    Calculate technical indicators using pandas-ta
    """
    logger.info("Calculating technical indicators...")
    
    # Calculate SMAs
    df['sma_7'] = df.ta.sma(length=7)
    df['sma_21'] = df.ta.sma(length=21)
    
    # Calculate EMA
    df['ema_26'] = df.ta.ema(length=26)
    
    # Calculate RSI
    df['rsi_14'] = df.ta.rsi(length=14)
    
    # Calculate MACD
    macd = df.ta.macd(fast=12, slow=26, signal=9)
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    df['macd_hist'] = macd['MACDh_12_26_9']
    
    # Calculate Bollinger Bands
    bollinger = df.ta.bbands(length=20, std=2)
    df['bb_width'] = (bollinger['BBU_20_2.0'] - bollinger['BBL_20_2.0']) / bollinger['BBM_20_2.0']
    
    # Calculate ATR
    df['atr_14'] = df.ta.atr(length=14)
    
    # Calculate Historical Volatility
    df['returns'] = df['close'].pct_change()
    df['hist_vol_20'] = df['returns'].rolling(window=20).std() * np.sqrt(24)  # Annualized
    
    return df

def normalize_data(df):
    """
    Normalize all numerical columns using Min-Max scaling
    """
    logger.info("Normalizing data...")
    
    # Select columns to normalize (excluding timestamp)
    columns_to_normalize = df.select_dtypes(include=[np.number]).columns
    
    # Initialize scaler
    scaler = MinMaxScaler()
    
    # Normalize the data
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    
    return df

def clean_data(df):
    """
    Remove NaN and infinite values
    """
    logger.info("Cleaning data...")
    
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Remove rows with NaN values
    df = df.dropna()
    
    return df

def main():
    # Read the input CSV file
    input_file = "btc_historical_data_20230318_20250318.csv"
    logger.info(f"Reading data from {input_file}")
    
    try:
        df = pd.read_csv(input_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate technical indicators
        df = calculate_technical_indicators(df)
        
        # Clean the data
        df = clean_data(df)
        
        # Normalize the data
        df = normalize_data(df)
        
        # Save the processed data
        output_file = "btc_processed_data.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Processed data saved to {output_file}")
        
        # Print summary statistics
        logger.info("\nDataset Summary:")
        logger.info(f"Total rows: {len(df)}")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.info("\nColumns in the dataset:")
        for col in df.columns:
            logger.info(f"- {col}")
            
    except Exception as e:
        logger.error(f"Error processing data: {e}")

if __name__ == "__main__":
    main() 