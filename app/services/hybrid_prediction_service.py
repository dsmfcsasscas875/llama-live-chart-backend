import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime, timedelta
import threading
import copy
import os
import pickle
import logging
from app.services.twelvedata_service import twelvedata_service
from app.core.config import settings

logger = logging.getLogger(__name__)

# Configuration based on volatile_stock_predictor_v2.py
SEQ_LEN = 60
BATCH_SIZE = 32
HIDDEN_SIZE = 128
EPOCHS = 200
LEARNING_RATE = 0.0005
GRAD_CLIP = 0.5
PATIENCE = 25
NUM_HEADS = 4
ENSEMBLE_SIZE = 5
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
        logger.info(f"HybridPredictionService initialized on: {self.device}")
        self.scaler = RobustScaler()
        self._models  = {}  # {symbol: list_of_ensemble_models}
        self._scalers = {}  # {symbol: fitted RobustScaler}
        self._cache   = {}
        self._training_lock = threading.Lock()
        self._in_progress   = set()

        # Ensure checkpoint directory exists
        os.makedirs(settings.MODEL_CHECKPOINT_DIR, exist_ok=True)

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

        train_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_train).to(self.device),
                torch.FloatTensor(y_train).to(self.device)
            ),
            batch_size=BATCH_SIZE, shuffle=False
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

        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)

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
        return model, best_val_loss

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _checkpoint_dir(self, symbol: str) -> str:
        path = os.path.join(settings.MODEL_CHECKPOINT_DIR, symbol)
        os.makedirs(path, exist_ok=True)
        return path

    def _save_checkpoint(self, symbol: str, version: str,
                         ensemble: list, scaler: RobustScaler,
                         metrics: dict, db_session) -> str:
        """
        Persist ensemble weights + scaler to disk and record the checkpoint
        in the database.  Returns the checkpoint file path.
        """
        from app.models.training_history import ModelCheckpoint

        ckpt_dir  = self._checkpoint_dir(symbol)
        ckpt_path = os.path.join(ckpt_dir, f"{version}.pt")

        payload = {
            "version":        version,
            "symbol":         symbol,
            "n_features":     ensemble[0].cnn[0].in_channels,
            "ensemble_states": [m.state_dict() for m in ensemble],
            "scaler":         scaler,
            "metrics":        metrics,
            "saved_at":       datetime.utcnow().isoformat(),
        }
        torch.save(payload, ckpt_path)
        logger.info(f"[{symbol}] Checkpoint saved → {ckpt_path}")

        # Persist to DB
        record = ModelCheckpoint(
            symbol          = symbol,
            model_version   = version,
            checkpoint_path = ckpt_path,
        )
        db_session.add(record)
        db_session.commit()
        return ckpt_path

    def _load_checkpoint(self, symbol: str, checkpoint_path: str) -> bool:
        """
        Load ensemble weights + scaler from a .pt file into memory.
        Returns True on success, False on failure.
        """
        if not os.path.exists(checkpoint_path):
            logger.warning(f"[{symbol}] Checkpoint file not found: {checkpoint_path}")
            return False
        try:
            payload  = torch.load(checkpoint_path, map_location=self.device)
            n_feat   = payload["n_features"]
            ensemble = []
            for state in payload["ensemble_states"]:
                m = VolatileModel(n_feat, HIDDEN_SIZE, NUM_HEADS).to(self.device)
                m.load_state_dict(state)
                m.eval()
                ensemble.append(m)
            self._models[symbol]  = ensemble
            self._scalers[symbol] = payload["scaler"]
            logger.info(
                f"[{symbol}] Loaded checkpoint v{payload['version']} "
                f"(saved {payload.get('saved_at', 'unknown')})"
            )
            return True
        except Exception as exc:
            logger.error(f"[{symbol}] Failed to load checkpoint {checkpoint_path}: {exc}")
            return False

    def load_latest_model(self, symbol: str, db_session) -> bool:
        """
        Query the DB for the most recent successful checkpoint for *symbol*
        and load it into memory.  Returns True if a model was loaded.
        """
        from app.models.training_history import ModelCheckpoint

        record = (
            db_session.query(ModelCheckpoint)
            .filter(ModelCheckpoint.symbol == symbol)
            .order_by(ModelCheckpoint.created_at.desc())
            .first()
        )
        if record is None:
            logger.info(f"[{symbol}] No checkpoint found in DB — model will train from scratch.")
            return False
        return self._load_checkpoint(symbol, record.checkpoint_path)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_ensemble(self, symbol: str, db_session=None):
        """
        Train the CNN-LSTM ensemble for *symbol*, save a checkpoint to disk,
        and record training history + checkpoint metadata in the database.

        If *db_session* is None a new session is created internally.
        On failure the previous in-memory model (if any) is preserved as
        a fallback so inference keeps working.
        """
        from app.models.training_history import TrainingHistory, ModelCheckpoint
        from app.db.session import SessionLocal

        own_session = db_session is None
        if own_session:
            db_session = SessionLocal()

        version = f"{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        history_record = TrainingHistory(
            symbol        = symbol,
            model_version = version,
            status        = "pending",
        )
        db_session.add(history_record)
        db_session.commit()
        db_session.refresh(history_record)

        logger.info(f"[{symbol}] Starting training (version={version}) …")
        try:
            df_raw = twelvedata_service.get_stock_data(symbol)
            if df_raw.empty:
                raise ValueError(f"Could not fetch data for {symbol} from Twelve Data")

            df = self.calculate_features(df_raw)

            n         = len(df)
            train_end = int(n * 0.9)

            train_df_vals = df.values[:train_end]
            scaler = RobustScaler()
            scaler.fit(train_df_vals)
            data_scaled = scaler.transform(df.values)

            X, y = self.create_sequences(data_scaled, SEQ_LEN)
            X_train_raw, y_train_raw = X[:train_end], y[:train_end]
            X_val, y_val             = X[train_end:], y[train_end:]

            X_train, y_train = self.augment_sequences(
                X_train_raw, y_train_raw, noise_std=NOISE_STD, n_copies=2
            )
            n_features = X_train.shape[2]

            ensemble       = []
            val_losses     = []
            best_val_losses = []
            for seed in range(ENSEMBLE_SIZE):
                model, best_val_loss = self.train_single(
                    X_train, y_train, X_val, y_val, n_features, seed=seed
                )
                ensemble.append(model)
                best_val_losses.append(best_val_loss)
                logger.info(
                    f"[{symbol}] Model {seed + 1}/{ENSEMBLE_SIZE} trained — "
                    f"best_val_loss={best_val_loss:.6f}"
                )

            metrics = {
                "best_val_losses": best_val_losses,
                "avg_best_val_loss": float(np.mean(best_val_losses)),
                "ensemble_size": ENSEMBLE_SIZE,
                "epochs_config": EPOCHS,
                "patience": PATIENCE,
            }

            # Persist checkpoint
            ckpt_path = self._save_checkpoint(
                symbol, version, ensemble, scaler, metrics, db_session
            )

            # Update in-memory state
            self._models[symbol]  = ensemble
            self._scalers[symbol] = scaler
            # Keep legacy self.scaler pointing to the last trained symbol
            self.scaler = scaler

            # Mark training as successful
            history_record.status  = "success"
            history_record.metrics = metrics
            db_session.commit()

            logger.info(
                f"[{symbol}] Training completed successfully. "
                f"avg_best_val_loss={metrics['avg_best_val_loss']:.6f} | "
                f"checkpoint={ckpt_path}"
            )

        except Exception as exc:
            error_msg = str(exc)
            logger.error(f"[{symbol}] Training FAILED: {error_msg}", exc_info=True)

            history_record.status        = "failed"
            history_record.error_message = error_msg
            db_session.commit()

            # Fallback: keep whatever model was already in memory
            if symbol in self._models:
                logger.warning(
                    f"[{symbol}] Falling back to previously loaded model in memory."
                )
            else:
                logger.warning(
                    f"[{symbol}] No fallback model available — predictions will be unavailable."
                )
        finally:
            if own_session:
                db_session.close()

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
        scaler = self._scalers.get(symbol, self.scaler)
        data_scaled = scaler.transform(df.values)

        last_sequence = data_scaled[-SEQ_LEN:]
        last_sequence_t = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.device)

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
        pred_price = scaler.inverse_transform(dummy)[0, 0]

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
                seq_t = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
                
                sum_p = 0
                for model in self._models[symbol]:
                    model.eval()
                    with torch.no_grad():
                        sum_p += model(seq_t).item()
                avg_p = sum_p / len(self._models[symbol])
                
                # Inverse transform
                d_p = np.zeros((1, df.shape[1])); d_p[0, 0] = avg_p
                p_val = scaler.inverse_transform(d_p)[0, 0]
                
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

    def load_all_models(self, symbols: list, db_session=None):
        """
        Load the latest persisted checkpoint for every symbol in *symbols*.
        Called at service startup so predictions are available immediately
        without waiting for a full re-train.

        Returns a dict {symbol: True/False} indicating which symbols were
        successfully restored from disk.
        """
        from app.db.session import SessionLocal

        own_session = db_session is None
        if own_session:
            db_session = SessionLocal()

        results = {}
        try:
            for sym in symbols:
                loaded = self.load_latest_model(sym, db_session)
                results[sym] = loaded
                if loaded:
                    logger.info(f"[{sym}] Model restored from checkpoint at startup.")
                else:
                    logger.info(
                        f"[{sym}] No checkpoint found — model will be trained on first request."
                    )
        finally:
            if own_session:
                db_session.close()

        return results

    def train_daily_all(self, symbols: list = None):
        """
        Re-train all symbols.  Called by the daily cron job.
        Each symbol is trained independently; a failure in one does not
        prevent the others from running.
        """
        if symbols is None:
            symbols = ["NVDA", "AAPL", "TSLA", "MSFT", "GOOGL"]
        for sym in symbols:
            self.train_ensemble(sym)


hybrid_prediction_service = HybridPredictionService()
