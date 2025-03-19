# I don't have a GPU, so I'm using CPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
print("GPU available:", tf.config.list_physical_devices('GPU'))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sequences(data, y_data, timesteps):
    """
    Create sequences for time series data
    """
    X, y = [], []
    for i in range(len(data) - timesteps):
        X.append(data[i:(i + timesteps)])
        y.append(y_data[i + timesteps])  # Dự đoán giá đóng cửa tiếp theo
    return np.array(X), np.array(y)

def build_gru_model(input_shape):
    """
    Build GRU model architecture
    """
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

def prepare_data(df, timesteps=48):
    """
    Prepare data for training
    """
    logger.info("Preparing data for training...")

    feature_columns = ['open', 'high', 'low', 'close', 'volume', 
                      'sma_7', 'sma_21', 'ema_26', 'rsi_14',
                      'macd', 'macd_signal', 'macd_hist', 'bb_width',
                      'atr_14', 'hist_vol_20']
    
    # Kiểm tra dữ liệu đã chuẩn hóa chưa
    X = df[feature_columns].values
    y = df['close'].values  # Dự đoán giá đóng cửa tiếp theo
    
    logger.info(f"Data min: {X.min()}, max: {X.max()}")

    # Tạo chuỗi dữ liệu
    X_seq, y_seq = create_sequences(X, y, timesteps)

    # Chia tập dữ liệu
    train_size = int(0.7 * len(X_seq))
    val_size = int(0.2 * len(X_seq))

    X_train, y_train = X_seq[:train_size], y_seq[:train_size]
    X_val, y_val = X_seq[train_size:train_size + val_size], y_seq[train_size:train_size + val_size]
    X_test, y_test = X_seq[train_size + val_size:], y_seq[train_size + val_size:]

    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Validation set shape: {X_val.shape}")
    logger.info(f"Test set shape: {X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test

def train_model(X_train, y_train, X_val, y_val):
    """
    Train the GRU model
    """
    logger.info("Building and training model...")

    # Build model
    model = build_gru_model((X_train.shape[1], X_train.shape[2]))

    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    model_checkpoint = ModelCheckpoint(
        'models/best_model.h5',
        monitor='val_loss',
        save_best_only=True,
        mode='min'
    )

    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )

    return model, history

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    """
    logger.info("Evaluating model...")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f"Test MSE: {mse:.4f}")
    logger.info(f"Test MAE: {mae:.4f}")
    logger.info(f"Test R²: {r2:.4f}")

    return mse, mae, r2

def main():
    # Tạo thư mục lưu model
    os.makedirs('models', exist_ok=True)

    try:
        # Load dữ liệu đã chuẩn hóa
        df = pd.read_csv('btc_processed_data.csv')

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Chuẩn bị dữ liệu
        X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(df)

        # Train model
        model, history = train_model(X_train, y_train, X_val, y_val)

        # Evaluate model
        mse, mae, r2 = evaluate_model(model, X_test, y_test)

        # Save model
        model.save('models/btc_gru_model.h5')
        logger.info("Model saved to models/btc_gru_model.h5")

    except Exception as e:
        logger.error(f"Error during model training: {e}")

if __name__ == "__main__":
    main()
