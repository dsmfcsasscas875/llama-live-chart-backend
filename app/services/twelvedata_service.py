from twelvedata import TDClient
import pandas as pd
from app.core.config import settings
import time

class TwelveDataService:
    def __init__(self):
        self.api_key = getattr(settings, "TWELVE_DATA_API_KEY", "demo") # Default to demo if not set
        self.td = TDClient(apikey=self.api_key)

    def get_stock_data(self, symbol: str, interval: str = "1day", outputsize: int = 5000):
        """Fetch historical stock data using Twelve Data"""
        try:
            ts = self.td.time_series(
                symbol=symbol,
                interval=interval,
                outputsize=outputsize,
                order="ASC"
            )
            df = ts.as_pandas()
            
            # Twelve Data returns OHLCV. Some models expect specific case.
            # VolatileModel expects: "Close", "Open", "High", "Low", "Volume"
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            return df
        except Exception as e:
            print(f"Twelve Data Error: {e}")
            return pd.DataFrame()

twelvedata_service = TwelveDataService()
