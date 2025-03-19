import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging
import joblib  # Để lưu scaler

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

    df['sma_7'] = df.ta.sma(length=7)
    df['sma_21'] = df.ta.sma(length=21)
    df['ema_26'] = df.ta.ema(length=26)
    df['rsi_14'] = df.ta.rsi(length=14)

    macd = df.ta.macd(fast=12, slow=26, signal=9)
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    df['macd_hist'] = macd['MACDh_12_26_9']

    bollinger = df.ta.bbands(length=20, std=2)
    df['bb_width'] = (bollinger['BBU_20_2.0'] - bollinger['BBL_20_2.0']) / bollinger['BBM_20_2.0']

    df['atr_14'] = df.ta.atr(length=14)

    df['returns'] = df['close'].pct_change()
    df['hist_vol_20'] = df['returns'].rolling(window=20).std() * np.sqrt(24)

    return df

def normalize_data(df, scaler_file="scaler.pkl"):
    """
    Normalize all numerical columns using Min-Max scaling
    """
    logger.info("Normalizing data...")

    # Chỉ chọn các cột số để chuẩn hóa
    columns_to_normalize = df.select_dtypes(include=[np.number]).columns.tolist()

    # Loại bỏ cột 'timestamp' nếu có
    if 'timestamp' in columns_to_normalize:
        columns_to_normalize.remove('timestamp')

    # Khởi tạo scaler
    scaler = MinMaxScaler()

    # Chuẩn hóa dữ liệu
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

    # Lưu scaler để sử dụng sau này
    joblib.dump(scaler, scaler_file)
    logger.info(f"Scaler saved to {scaler_file}")

    return df

def clean_data(df):
    """
    Remove NaN and infinite values
    """
    logger.info("Cleaning data...")

    # Thay thế giá trị vô cực bằng NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Loại bỏ các dòng có giá trị NaN
    df.dropna(inplace=True)

    return df

def main():
    input_file = "btc_historical_data_20230318_20250318.csv"
    output_file = "btc_processed_data.csv"
    scaler_file = "scaler.pkl"

    logger.info(f"Reading data from {input_file}")

    try:
        df = pd.read_csv(input_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Tính toán chỉ báo kỹ thuật
        df = calculate_technical_indicators(df)

        # Dọn dẹp dữ liệu
        df = clean_data(df)

        # Chuẩn hóa dữ liệu
        df = normalize_data(df, scaler_file)

        # Lưu dữ liệu đã xử lý
        df.to_csv(output_file, index=False)
        logger.info(f"Processed data saved to {output_file}")

        # In thống kê dữ liệu
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
