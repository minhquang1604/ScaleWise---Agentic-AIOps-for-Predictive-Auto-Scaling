#!/usr/bin/env python3
"""
Train model with traffic pattern matching demo scenarios
Traffic phases: NORMAL(200) -> SPIKE-1(600) -> EXTREME(900) -> SPIKE-2(700) -> COOL-DOWN(200)
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

WINDOW_SIZE = 10
PREDICTION_STEPS = 1


def generate_demo_traffic(num_cycles=50, samples_per_cycle=360):
    """
    Generate traffic data matching the demo pattern
    Each cycle: 200 -> 600 -> 900 -> 700 -> 200 RPS over 6 minutes (50 samples)
    """
    np.random.seed(42)
    
    all_traffic = []
    
    for cycle in range(num_cycles):
        phase_traffic = []
        
        # Phase 1: NORMAL - 200 RPS base (10 samples)
        normal = np.random.normal(200, 20, 10)
        normal = np.clip(normal, 150, 250)
        phase_traffic.extend(normal)
        
        # Phase 2: SPIKE-1 - 600 RPS (10 samples)
        spike1 = np.linspace(200, 600, 10) + np.random.normal(0, 30, 10)
        spike1 = np.clip(spike1, 500, 700)
        phase_traffic.extend(spike1)
        
        # Phase 3: EXTREME - 900 RPS (10 samples)
        extreme = np.linspace(600, 900, 10) + np.random.normal(0, 50, 10)
        extreme = np.clip(extreme, 750, 1000)
        phase_traffic.extend(extreme)
        
        # Phase 4: SPIKE-2 - 700 RPS (10 samples)
        spike2 = np.linspace(900, 700, 10) + np.random.normal(0, 40, 10)
        spike2 = np.clip(spike2, 600, 800)
        phase_traffic.extend(spike2)
        
        # Phase 5: COOL-DOWN - 200 RPS (10 samples)
        cooldown = np.linspace(700, 200, 10) + np.random.normal(0, 20, 10)
        cooldown = np.clip(cooldown, 150, 250)
        phase_traffic.extend(cooldown)
        
        # Add some noise and variation between cycles
        cycle_variation = np.random.uniform(0.9, 1.1)
        phase_traffic = [t * cycle_variation for t in phase_traffic]
        
        all_traffic.extend(phase_traffic)
    
    return np.array(all_traffic)


def create_sequences(data, window_size=WINDOW_SIZE, prediction_steps=PREDICTION_STEPS):
    X, y = [], []
    for i in range(len(data) - window_size - prediction_steps + 1):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size:i + window_size + prediction_steps])
    return np.array(X), np.array(y)


def train_model():
    print("=" * 60)
    print("Training Model for AIOps Demo")
    print("=" * 60)
    
    # Generate training data matching demo traffic pattern
    print("\n[1/4] Generating training data...")
    traffic_data = generate_demo_traffic(num_cycles=100, samples_per_cycle=50)
    
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=len(traffic_data), freq='1min'),
        'traffic': traffic_data
    })
    
    print(f"Generated {len(traffic_data)} samples")
    print(f"Traffic range: {traffic_data.min():.0f} - {traffic_data.max():.0f} RPS")
    print(f"Traffic mean: {traffic_data.mean():.0f} RPS")
    
    # Save raw data
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/training_traffic.csv', index=False)
    print("Saved training data to data/training_traffic.csv")
    
    # Scale data
    print("\n[2/4] Scaling data...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(traffic_data.reshape(-1, 1)).flatten()
    
    # Create sequences
    print("\n[3/4] Creating sequences...")
    X, y = create_sequences(scaled_data, WINDOW_SIZE, PREDICTION_STEPS)
    
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    
    print(f"Training: {X_train.shape[0]} samples")
    print(f"Testing: {X_test.shape[0]} samples")
    
    # Save scaler
    os.makedirs('models', exist_ok=True)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Saved scaler to models/scaler.pkl")
    
    # Train using sklearn (simpler, no PyTorch dependency)
    print("\n[4/4] Training model...")
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        
        # Reshape for sklearn
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        X_test_2d = X_test.reshape(X_test.shape[0], -1)
        
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train_2d, y_train.flatten())
        
        # Evaluate
        train_score = model.score(X_train_2d, y_train)
        test_score = model.score(X_test_2d, y_test)
        
        print(f"Train R² Score: {train_score:.4f}")
        print(f"Test R² Score: {test_score:.4f}")
        
        # Test prediction on sample data
        sample = scaled_data[:WINDOW_SIZE].reshape(1, -1)
        pred = model.predict(sample)
        pred_original = scaler.inverse_transform(pred.reshape(-1, 1))[0][0]
        print(f"Sample prediction: {pred_original:.0f} RPS")
        
        # Save model using joblib
        import joblib
        joblib.dump(model, 'models/lstm_model.pkl')
        print("Saved model to models/lstm_model.pkl")
        
    except ImportError:
        print("sklearn not available, trying PyTorch...")
        from src.model.lstm_model import train_model as torch_train_model
        
        model = torch_train_model(
            X_train, y_train,
            X_val=X_test, y_val=y_test,
            epochs=50,
            batch_size=32,
            model_path='models/lstm_model.pth'
        )
        print("Saved PyTorch model to models/lstm_model.pth")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    return scaler, model


if __name__ == "__main__":
    train_model()