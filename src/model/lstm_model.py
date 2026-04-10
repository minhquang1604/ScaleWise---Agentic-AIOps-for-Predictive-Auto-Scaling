import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.data_preprocessing import preprocess_data, WINDOW_SIZE, PREDICTION_STEPS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=PREDICTION_STEPS, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def train_model(X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32, lr=0.001, model_path='models/lstm_model.pth'):
    model = LSTMPredictor(input_size=1, hidden_size=64, num_layers=2, output_size=PREDICTION_STEPS).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train).unsqueeze(-1),
        torch.FloatTensor(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if X_val is not None:
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val).unsqueeze(-1),
            torch.FloatTensor(y_val)
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        if X_val is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), model_path)
        else:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}")
            if train_loss < best_loss:
                best_loss = train_loss
                torch.save(model.state_dict(), model_path)
    
    return model


def predict(model, X, scaler):
    model.eval()
    X_tensor = torch.FloatTensor(X).unsqueeze(-1).to(DEVICE)
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()
    return scaler.inverse_transform(predictions)


def load_model(model_path='models/lstm_model.pth'):
    model = LSTMPredictor(input_size=1, hidden_size=64, num_layers=2, output_size=PREDICTION_STEPS).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


def save_scaler(scaler, path='models/scaler.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(scaler, f)


def load_scaler(path='models/scaler.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    print("Preprocessing data...")
    X_train, y_train, X_test, y_test, scaler = preprocess_data()
    
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
    
    os.makedirs('models', exist_ok=True)
    save_scaler(scaler)
    
    print("\nTraining LSTM model...")
    model = train_model(X_train, y_train, X_test, y_test, epochs=50, batch_size=32, lr=0.001)
    
    print("\nModel trained and saved!")
