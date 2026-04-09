import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from app.services.hybrid_prediction_service import hybrid_prediction_service

class PredictionService:
    @staticmethod
    def forecast_price(stock_data: list, days_to_predict: int = 14, model_type: str = "hybrid", symbol: str = "UNKNOWN"):
        """Always use the advanced Hybrid CNN-LSTM model"""
        return hybrid_prediction_service.forecast_price(stock_data, days_to_predict, symbol)

prediction_service = PredictionService()
