import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import requests
import gzip
import struct

WINDOW_SIZE = 10
PREDICTION_STEPS = 1


def generate_synthetic_traffic(num_samples=10000, seed=42):
    np.random.seed(seed)
    time = np.arange(num_samples)
    base_traffic = 500
    daily_pattern = 200 * np.sin(2 * np.pi * time / 1440)
    hourly_pattern = 100 * np.sin(2 * np.pi * time / 60)
    noise = np.random.normal(0, 30, num_samples)
    spikes = np.zeros(num_samples)
    for _ in range(20):
        idx = np.random.randint(100, num_samples - 100)
        spikes[idx:idx + np.random.randint(5, 20)] = np.random.uniform(200, 500)
    traffic = base_traffic + daily_pattern + hourly_pattern + noise + spikes
    traffic = np.maximum(traffic, 50)
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=num_samples, freq='1min'),
        'traffic': traffic
    })
    return df


def load_mawi_sample():
    url = "http://mawi.wide.ad.jp/mawi/ditl/2024/202401011400.csv.gz"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with gzip.open(os.path.join('data', 'mawi_sample.csv.gz'), 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e:
        print(f"Could not download MAWI data: {e}")
    return False


def create_sequences(data, window_size=WINDOW_SIZE, prediction_steps=PREDICTION_STEPS):
    X, y = [], []
    for i in range(len(data) - window_size - prediction_steps + 1):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size:i + window_size + prediction_steps])
    return np.array(X), np.array(y)


def preprocess_data(df=None, scaler=None, train=True):
    if df is None:
        df = generate_synthetic_traffic()
    
    traffic_data = df['traffic'].values.reshape(-1, 1)
    
    if train:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(traffic_data)
    else:
        scaled_data = scaler.transform(traffic_data)
    
    X, y = create_sequences(scaled_data.flatten())
    
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    
    return X_train, y_train, X_test, y_test, scaler


def inverse_transform(scaler, data):
    return scaler.inverse_transform(data)


if __name__ == "__main__":
    df = generate_synthetic_traffic()
    df.to_csv('data/traffic_data.csv', index=False)
    print(f"Generated {len(df)} samples")
    print(df.head())
    print(f"\nTraffic statistics:")
    print(df['traffic'].describe())
