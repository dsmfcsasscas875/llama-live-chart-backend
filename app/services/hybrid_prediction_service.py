import torch
import time
torch.set_num_threads(1) # Prevent CPU starvation by limiting torch to 1 thread

import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime, timedelta
import threading
import copy
import os
import joblib # For saving/loading the scaler
from pathlib import Path
from app.services.twelvedata_service import twelvedata_service

# Configuration based on volatile_stock_predictor_v2.py
SEQ_LEN = 60
BATCH_SIZE = 32
HIDDEN_SIZE = 128
EPOCHS = 200
LEARNING_RATE = 0.0005
GRAD_CLIP = 0.5
PATIENCE = 25
NUM_HEADS = 4
ENSEMBLE_SIZE = 5 # Kept at 5 as requested by user
NOISE_STD = 0.005

class VolatileModel(nn.Module):
    def __init__(self, n_features, hidden_size, num_heads=4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.4,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=0.2,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        out = self.attn_norm(lstm_out + attn_out)
        return self.fc(out[:, -1, :])

class HybridPredictionService:
    def __init__(self):
        # Update to VRAM (GPU) as requested
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"HybridPredictionService initialized on: {self.device}")
        self.scaler = RobustScaler()
        self._models = {} # {symbol: list_of_ensemble_models}
        self._cache = {}
        self._training_lock = threading.Lock()
        self._in_progress = set()
        self._last_trained = {} # {symbol: datetime}
        
        # Persistence setup - Using absolute path for better Volume mounting compatibility
        base_path = Path(__file__).resolve().parent.parent.parent # Project root
        self.model_dir = base_path / "data" / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        print(f"Model persistence directory set to: {self.model_dir.resolve()}")
        
        # Try to load existing models at initialization
        self.load_all_existing()

    def calculate_features(self, df):
        """Feature engineering from volatile_stock_predictor_v2.py"""
        df = df.copy()
        close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]

        df["returns"]     = close.pct_change()
        df["log_return"]  = np.log(close / close.shift(1))
        df["ma7"]         = close.rolling(7).mean()
        df["ma21"]        = close.rolling(21).mean()
        df["ma50"]        = close.rolling(50).mean()
        df["std7"]        = close.rolling(7).std()
        df["std21"]       = close.rolling(21).std()
        df["parkinson"]   = np.log(high / low) ** 2 / (4 * np.log(2))

        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        df["rsi"]         = 100 - (100 / (1 + gain / (loss + 1e-9)))
        df["macd"]        = close.ewm(span=12).mean() - close.ewm(span=26).mean()
        df["macd_signal"] = df["macd"].ewm(span=9).mean()

        bb_mid            = close.rolling(20).mean()
        bb_std            = close.rolling(20).std()
        df["bb_upper"]    = bb_mid + 2 * bb_std
        df["bb_lower"]    = bb_mid - 2 * bb_std
        df["bb_width"]    = (df["bb_upper"] - df["bb_lower"]) / (bb_mid + 1e-9)

        df["high_low"]    = high - low
        df["open_close"]  = close.diff()
        df["volume_ma"]   = volume.rolling(10).mean()
        df["volume_ratio"]= volume / (df["volume_ma"] + 1e-9)
        
        # Day and month from index (Twelve Data returns index as DatetimeIndex)
        df["day"]         = df.index.dayofweek
        df["month"]       = df.index.month

        df.dropna(inplace=True)
        return df[[
            "Close", "Open", "High", "Low", "Volume",
            "returns", "log_return",
            "ma7", "ma21", "ma50",
            "std7", "std21", "parkinson",
            "rsi", "macd", "macd_signal",
            "bb_upper", "bb_lower", "bb_width",
            "high_low", "open_close",
            "volume_ma", "volume_ratio",
            "day", "month"
        ]]

    def create_sequences(self, data, seq_len):
        xs, ys = [], []
        for i in range(len(data) - seq_len):
            xs.append(data[i : i + seq_len])
            ys.append(data[i + seq_len, 0]) # Close is at index 0
        return np.array(xs), np.array(ys)

    def augment_sequences(self, X, y, noise_std=0.005, n_copies=2):
        X_aug, y_aug = [X], [y]
        for _ in range(n_copies):
            noise = np.random.normal(0, noise_std, X.shape)
            X_aug.append(X + noise)
            y_aug.append(y)
        return np.concatenate(X_aug), np.concatenate(y_aug)

    def train_single(self, X_train, y_train, X_val, y_val, n_features, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Move training data to VRAM immediately for faster iteration
        train_x_t = torch.as_tensor(X_train, device=self.device, dtype=torch.float32)
        train_y_t = torch.as_tensor(y_train, device=self.device, dtype=torch.float32)
        
        train_loader = DataLoader(
            TensorDataset(train_x_t, train_y_t),
            batch_size=BATCH_SIZE, 
            shuffle=False
        )

        model = VolatileModel(n_features, HIDDEN_SIZE, NUM_HEADS).to(self.device)
        criterion = nn.HuberLoss(delta=1.0)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=1e-3,
            amsgrad=True
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=30, T_mult=2, eta_min=1e-6
        )

        X_val_t = torch.as_tensor(X_val, device=self.device, dtype=torch.float32)
        y_val_t = torch.as_tensor(y_val, device=self.device, dtype=torch.float32)

        best_val_loss = float("inf")
        best_weights = None
        counter = 0

        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0.0
            for bx, by in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(bx).squeeze(), by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val_t).squeeze(), y_val_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())
                counter = 0
            else:
                counter += 1
                if counter >= PATIENCE:
                    break

        if best_weights:
            model.load_state_dict(best_weights)
        return model

    def train_ensemble(self, symbol: str):
        """Train the ensemble for a given symbol using Twelve Data"""
        print(f"Starting daily training for {symbol}...")
        df_raw = twelvedata_service.get_stock_data(symbol)
        if df_raw.empty:
            print(f"Could not fetch data for {symbol} from Twelve Data")
            return

        df = self.calculate_features(df_raw)
        
        n = len(df)
        train_end = int(n * 0.9)
        
        train_df_vals = df.values[:train_end]
        self.scaler = RobustScaler()
        self.scaler.fit(train_df_vals)
        data_scaled = self.scaler.transform(df.values)

        X, y = self.create_sequences(data_scaled, SEQ_LEN)
        X_train_raw, y_train_raw = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:], y[train_end:]

        X_train, y_train = self.augment_sequences(X_train_raw, y_train_raw, noise_std=NOISE_STD, n_copies=2)
        n_features = X_train.shape[2]

        ensemble = []
        for seed in range(ENSEMBLE_SIZE):
            model = self.train_single(X_train, y_train, X_val, y_val, n_features, seed=seed)
            ensemble.append(model)
        
        self._models[symbol] = ensemble
        self._last_trained[symbol] = datetime.now()
        
        # Persist to disk
        self.save_ensemble(symbol)
        
        # Clear VRAM cache after training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print(f"Daily training for {symbol} completed successfully and models are on {self.device} VRAM.")

    def save_ensemble(self, symbol: str):
        """Save ensemble models and scaler to disk"""
        try:
            symbol_dir = self.model_dir / symbol
            symbol_dir.mkdir(exist_ok=True)
            
            # Save scaler
            joblib.dump(self.scaler, symbol_dir / "scaler.gz")
            
            # Save models
            for i, model in enumerate(self._models[symbol]):
                torch.save(model.state_dict(), symbol_dir / f"model_v2_{i}.pth")
                
            print(f"Persistence: Saved models and scaler for {symbol} to {symbol_dir}")
        except Exception as e:
            print(f"Error saving model for {symbol}: {e}")

    def load_all_existing(self):
        """Scans the model directory and loads everything found"""
        if not self.model_dir.exists():
            return
            
        for symbol_path in self.model_dir.iterdir():
            if symbol_path.is_dir():
                symbol = symbol_path.name
                self.load_ensemble(symbol)

    def load_ensemble(self, symbol: str):
        """Load ensemble models and scaler from disk"""
        try:
            symbol_dir = self.model_dir / symbol
            scaler_path = symbol_dir / "scaler.gz"
            
            if not scaler_path.exists():
                return False
                
            # Load scaler
            self.scaler = joblib.load(scaler_path)
            
            # Find n_features from scaler/data if possible, or use a sample
            # Since we know our architecture, we can initialize it. 
            # Features count is 25 in version 2.
            n_features = 25 
            
            ensemble = []
            for i in range(ENSEMBLE_SIZE):
                model_path = symbol_dir / f"model_v2_{i}.pth"
                if model_path.exists():
                    model = VolatileModel(n_features, HIDDEN_SIZE, NUM_HEADS).to(self.device)
                    # Load onto correct device
                    model.load_state_dict(torch.load(model_path, map_location=self.device))
                    model.eval()
                    ensemble.append(model)
            
            if ensemble:
                self._models[symbol] = ensemble
                # Estimate last trained from file timestamp
                mtime = os.path.getmtime(scaler_path)
                self._last_trained[symbol] = datetime.fromtimestamp(mtime)
                print(f"Persistence: Loaded existing ensemble for {symbol} (Last trained: {self._last_trained[symbol]})")
                return True
        except Exception as e:
            print(f"Error loading model for {symbol}: {e}")
        return False

    def forecast_price(self, stock_data_list: list, days_to_predict: int = 14, symbol: str = "UNKNOWN"):
        """Main entry point for forecasting"""
        if symbol not in self._models:
            # If not trained yet (at startup), start training
            if symbol not in self._in_progress:
                thread = threading.Thread(target=self._run_training_sync, args=(symbol,))
                thread.start()
                return {"status": "started", "message": "First-time training started for this session."}
            else:
                return {"status": "processing", "message": "Model is training."}

        # Perform prediction with ensemble
        # Fetch latest data to get the sequence for prediction
        df_raw = twelvedata_service.get_stock_data(symbol, outputsize=SEQ_LEN + 100)
        if df_raw.empty:
            return {"status": "error", "message": "Failed to fetch data for prediction."}

        df = self.calculate_features(df_raw)
        data_scaled = self.scaler.transform(df.values)
        
        last_sequence = data_scaled[-SEQ_LEN:]
        last_sequence_t = torch.as_tensor(last_sequence, device=self.device, dtype=torch.float32).unsqueeze(0)

        all_preds = []
        for model in self._models[symbol]:
            model.eval()
            with torch.no_grad():
                pred = model(last_sequence_t).item()
            all_preds.append(pred)

        avg_pred = np.mean(all_preds)
        
        # Inverse transform to get actual price
        dummy = np.zeros((1, df.shape[1]))
        dummy[0, 0] = avg_pred
        pred_price = self.scaler.inverse_transform(dummy)[0, 0]

        # Simple 14-day projection based on model's 1-day prediction and historical trend
        last_close = df["Close"].iloc[-1]
        forecast = []
        current_date = df.index[-1]
        for i in range(1, days_to_predict + 1):
            current_date += timedelta(days=1)
            while current_date.weekday() >= 5: current_date += timedelta(days=1)
            
            # Simple simulation for future
            daily_change = (pred_price - last_close) / 1.0 # 1-day pred
            sim_price = last_close + (daily_change * i)
            
            forecast.append({
                "Date": current_date.strftime("%Y-%m-%d"),
                "PredictedClose": float(sim_price),
                "Confidence": float(1.0 - (i * 0.05))
            })

        # --- Backtest / Compare Mode Data ---
        # We'll take the last few historical points and see what the model would have predicted
        # to allow the "Compare Mode: Past vs Prediction" requested by the user.
        backtest = []
        hist_size = 30
        if len(data_scaled) > SEQ_LEN + hist_size:
            for i in range(len(data_scaled) - hist_size, len(data_scaled)):
                seq = data_scaled[i-SEQ_LEN:i]
                seq_t = torch.as_tensor(seq, device=self.device, dtype=torch.float32).unsqueeze(0)
                
                sum_p = 0
                for model in self._models[symbol]:
                    model.eval()
                    with torch.no_grad():
                        sum_p += model(seq_t).item()
                avg_p = sum_p / len(self._models[symbol])
                
                # Inverse transform
                d_p = np.zeros((1, df.shape[1])); d_p[0, 0] = avg_p
                p_val = self.scaler.inverse_transform(d_p)[0, 0]
                
                actual_val = df["Close"].iloc[i]
                date_str = df.index[i].strftime("%Y-%m-%d")
                
                backtest.append({
                    "Date": date_str,
                    "Actual": float(actual_val),
                    "Predicted": float(p_val)
                })

        return {
            "symbol": symbol,
            "prediction_1d": float(pred_price),
            "forecast": forecast,
            "backtest": backtest,
            "last_trained": self._last_trained.get(symbol, "Never"),
            "status": "success"
        }

    def _run_training_sync(self, symbol):
        with self._training_lock:
            if symbol in self._in_progress:
                return
            self._in_progress.add(symbol)
        try:
            self.train_ensemble(symbol)
        finally:
            with self._training_lock:
                self._in_progress.remove(symbol)

    def train_daily_all(self, symbols=["NVDA", "AAPL", "TSLA", "MSFT", "GOOGL"]):
        """Method to be called at startup/daily"""
        for sym in symbols:
            # Only train if model doesn't exist OR was trained more than 24h ago
            should_train = True
            if sym in self._models:
                last_time = self._last_trained.get(sym)
                if last_time and (datetime.now() - last_time).total_seconds() < 86400:
                    should_train = False
                    print(f"Startup: Skipping training for {sym}, fresh model already exists.")
            
            if should_train:
                self.train_ensemble(sym)
                time.sleep(5) # Give the system some breathing room between symbols


hybrid_prediction_service = HybridPredictionService()
