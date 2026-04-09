import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import List, Dict

class YFinanceService:
    _cache = {}
    _cache_timeout = 300 # 5 minutes
    @staticmethod
    def get_stock_data(symbol: str, period: str = "5y"):
        """Fetch historical stock data using Twelve Data Service"""
        from app.services.twelvedata_service import twelvedata_service
        # Translate period to outputsize roughly (5y ~ 1250 days)
        size = 1250 if period == "5y" else 500 if period == "2y" else 250
        df = twelvedata_service.get_stock_data(symbol, outputsize=size)
        
        if df.empty:
            # Fallback to yfinance if Twelve Data fails (e.g. key issue)
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            df = df.reset_index()
            # Normalize column names for frontend
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
            return df.to_dict(orient='records')

        df = df.reset_index()
        df['Date'] = df['datetime'].dt.strftime('%Y-%m-%d')
        return df.to_dict(orient='records')

    @staticmethod
    def get_stock_news(symbol: str):
        """Fetch latest stock news from yfinance - limited to 5 for speed"""
        ticker = yf.Ticker(symbol)
        news = ticker.news[:5] if ticker.news else []
        
        extracted_news = []
        for item in news:
            extracted_news.append({
                'title': item.get('title', ''),
                'publisher': item.get('publisher', ''),
                'link': item.get('link', ''),
                'providerPublishTime': item.get('providerPublishTime', 0),
                'type': item.get('type', '')
            })
        return extracted_news

    @staticmethod
    def get_market_lists():
        """Get Featured, Popular, and Latest US stocks with caching"""
        cache_key = "market_lists"
        now = time.time()
        
        if cache_key in YFinanceService._cache:
            data, timestamp = YFinanceService._cache[cache_key]
            if now - timestamp < YFinanceService._cache_timeout:
                return data

        # Reduced to 8 for speed, and using a slightly faster batch pattern
        featured = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AVGO", "COST", "ADBE"]
        popular = ["AMZN", "META", "NFLX", "COIN", "MARA", "RIOT", "BABA", "TCEHY"]
        latest = ["PLTR", "ARM", "RDDT", "SMCI", "SNOW", "CRWD", "MSTR", "HOOD"]
        
        def batch_get_info(symbols):
            try:
                # Use download to get data for all symbols at once - much faster
                data = yf.download(symbols, period="5d", interval="1d", group_by='ticker', progress=False, threads=True)
                results = []
                for sym in symbols:
                    try:
                        # Handle both single and multi-index results from yfinance
                        if len(symbols) == 1:
                            ticker_df = data
                        else:
                            ticker_df = data[sym]
                            
                        if not ticker_df.empty:
                            price = ticker_df['Close'].iloc[-1]
                            # Calculate simple change percent from last two closes
                            if len(ticker_df) > 1:
                                prev_price = ticker_df['Close'].iloc[-2]
                                change = ((price - prev_price) / prev_price * 100) if prev_price != 0 else 0
                            else:
                                change = 0
                        else:
                            price, change = 0, 0
                            
                        results.append({
                            'symbol': sym,
                            'name': sym,
                            'price': float(price) if not pd.isna(price) else 0,
                            'change_percent': float(change) if not pd.isna(change) else 0
                        })
                    except Exception as e:
                        print(f"Error processing {sym}: {e}")
                        results.append({'symbol': sym, 'name': sym, 'price': 0, 'change_percent': 0})
                return results
            except Exception as e:
                print(f"Batch fetch error: {e}")
                return [{'symbol': sym, 'name': sym, 'price': 0, 'change_percent': 0} for sym in symbols]

        data = {
            "featured": batch_get_info(featured),
            "popular": batch_get_info(popular),
            "latest": batch_get_info(latest)
        }
        
        # Save to cache
        YFinanceService._cache[cache_key] = (data, now)
        return data

    @staticmethod
    def get_sectors():
        """Common US stock sectors"""
        return [
            "Technology", "Healthcare", "Financial Services", "Consumer Cyclical",
            "Communication Services", "Industrial", "Energy", "Basic Materials",
            "Real Estate", "Utilities"
        ]

    @staticmethod
    def get_stocks_by_sector(sector: str):
        """Sample stocks by sector (Expanded for demo)"""
        sector_map = {
            "Technology": ["AAPL", "MSFT", "NVDA", "AMD", "ORCL", "CRM", "INTC", "CSCO"],
            "Healthcare": ["JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "LLY", "DHR"],
            "Financial Services": ["JPM", "BAC", "GS", "MS", "WFC", "V", "MA", "PYPL"],
            "Consumer Cyclical": ["AMZN", "TSLA", "HD", "NKE", "MCD", "SBUX", "LOW", "BKNG"],
            "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "TMUS", "VZ", "T", "CMCSA"],
            "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "VLO", "PXD"],
        }
        symbols = sector_map.get(sector, ["SPY", "QQQ", "VTI"])
        
        try:
            # Use batch download for sector stocks too
            data = yf.download(symbols, period="1d", progress=False)
            results = []
            for sym in symbols:
                try:
                    price = data['Close'][sym].iloc[-1] if len(symbols) > 1 else data['Close'].iloc[-1]
                    results.append({
                        'symbol': sym,
                        'price': float(price) if not pd.isna(price) else 0,
                        'change_percent': 0 # Download 1d doesn't give change easily, but we can fix later
                    })
                except:
                    results.append({'symbol': sym, 'price': 0, 'change_percent': 0})
            return results
        except:
            return [{'symbol': sym, 'price': 0, 'change_percent': 0} for sym in symbols]

    @staticmethod
    def recommend_stocks(symbol: str):
        """Recommend similar stocks based on sector of the input symbol"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info # Full info is slower but needed for sector
            sector = info.get('sector')
            
            if not sector:
                return []
                
            recommendations = YFinanceService.get_stocks_by_sector(sector)
            return [s for s in recommendations if s['symbol'] != symbol][:6]
        except:
            return []

    @staticmethod
    def get_extended_metrics(symbol: str):
        """Analyze volatility, volume efficiency and market regime"""
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="3mo") # Need 3 months for SMA calculations
        
        if df.empty:
            return {}

        # Volatility
        df['Returns'] = df['Close'].pct_change()
        volatility = df['Returns'].std() * (252**0.5)
        
        # Volume
        avg_volume = df['Volume'].tail(30).mean()
        last_volume = df['Volume'].iloc[-1]
        vol_efficiency = last_volume / avg_volume if avg_volume > 0 else 0
        
        # Market Regime (Bullish/Bearish) using SMA-20
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        last_close = df['Close'].iloc[-1]
        last_sma20 = df['SMA20'].iloc[-1]
        
        regime = "BULLISH" if last_close > last_sma20 else "BEARISH"
        
        return {
            'volatility': float(volatility),
            'is_volatile': bool(volatility > 0.40),
            'volatility_label': "High" if volatility > 0.40 else "Medium" if volatility > 0.20 else "Low",
            'avg_volume': int(avg_volume),
            'last_volume': int(last_volume),
            'vol_efficiency': float(vol_efficiency),
            'is_active': bool(vol_efficiency > 1.2),
            'regime': regime
        }

    @staticmethod
    def get_quote_summary(symbol: str):
        """Get quick summary with extended metrics for UI"""
        ticker = yf.Ticker(symbol)
        info = ticker.fast_info
        metrics = YFinanceService.get_extended_metrics(symbol)
        
        return {
            'symbol': symbol,
            'price': getattr(info, 'last_price', 0) or 0,
            'change': getattr(info, 'regular_market_change', 0) or 0,
            'change_percent': getattr(info, 'regular_market_change_percent', 0) or 0,
            'dayHigh': getattr(info, 'day_high', 0) or 0,
            'dayLow': getattr(info, 'day_low', 0) or 0,
            'metrics': metrics
        }

    @staticmethod
    def get_batch_quotes(symbols: List[str]):
        """Fetch multiple quotes at once - MUCH faster than individual calls"""
        if not symbols:
            return {}
            
        try:
            # Period 1d is enough for current price and change
            data = yf.download(symbols, period="5d", interval="1d", group_by='ticker', progress=False)
            results = {}
            
            for sym in symbols:
                try:
                    if len(symbols) == 1:
                        df = data
                    else:
                        df = data[sym]
                        
                    if not df.empty:
                        price = df['Close'].iloc[-1]
                        prev_price = df['Close'].iloc[-2] if len(df) > 1 else price
                        change = ((price - prev_price) / prev_price * 100) if prev_price != 0 else 0
                        
                        results[sym] = {
                            'symbol': sym,
                            'price': float(price) if not pd.isna(price) else 0,
                            'change_percent': float(change) if not pd.isna(change) else 0,
                            'dayHigh': float(df['High'].iloc[-1]) if 'High' in df else 0,
                            'dayLow': float(df['Low'].iloc[-1]) if 'Low' in df else 0,
                        }
                    else:
                        results[sym] = {'symbol': sym, 'price': 0, 'change_percent': 0}
                except:
                    results[sym] = {'symbol': sym, 'price': 0, 'change_percent': 0}
            return results
        except Exception as e:
            print(f"Batch quote error: {e}")
            return {sym: {'symbol': sym, 'price': 0, 'change_percent': 0} for sym in symbols}

    @staticmethod
    def get_discover_info():
        cache_key = "discover_info"
        now = time.time()
        
        if cache_key in YFinanceService._cache:
            data, timestamp = YFinanceService._cache[cache_key]
            if now - timestamp < 600:
                return data

        # Popular ETFs measuring sector trends
        sector_etfs = {
            "Technology (XLK)": "XLK",
            "Healthcare (XLV)": "XLV",
            "Financials (XLF)": "XLF",
            "Consumer Cyclical (XLY)": "XLY",
            "Energy (XLE)": "XLE",
            "Industrials (XLI)": "XLI"
        }
        
        # A list of popular stocks to find gainers/losers
        pool = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "NFLX", "COIN", "MARA", "PLTR", "ARM", "SMCI", "SNOW", "CRWD", "MSTR", "HOOD", "AMD", "CRM", "INTC", "PYPL", "SQ", "UBER", "ABNB", "SPOT", "SHOP", "ROKU", "ZM", "DOCU", "PTON"]
        
        try:
            sector_data = yf.download(list(sector_etfs.values()), period="5d", interval="1d", progress=False)
            sectors_list = []
            for name, sym in sector_etfs.items():
                try:
                    df = sector_data['Close'][sym] if len(sector_etfs) > 1 else sector_data['Close']
                    if len(df) > 1:
                        price = float(df.iloc[-1])
                        prev = float(df.iloc[-2])
                        change = ((price - prev) / prev) * 100
                        trend = "UPTREND" if change > 0 else "DOWNTREND"
                        sectors_list.append({"name": name, "symbol": sym, "price": price, "change_percent": change, "trend": trend})
                except:
                    pass

            pool_data = yf.download(pool, period="5d", interval="1d", progress=False)
            stocks = []
            for sym in pool:
                try:
                    df = pool_data['Close'][sym]
                    if len(df) > 1:
                        price = float(df.iloc[-1])
                        prev = float(df.iloc[-2])
                        change = ((price - prev) / prev) * 100
                        stocks.append({"symbol": sym, "price": price, "change_percent": change})
                except:
                    pass
                    
            stocks.sort(key=lambda x: x['change_percent'], reverse=True)
            gainers = stocks[:5]
            losers = stocks[-5:][::-1]
            
            data = {
                "sectors": sectors_list,
                "gainers": gainers,
                "losers": losers
            }
            YFinanceService._cache[cache_key] = (data, now)
            return data
        except Exception as e:
            return {"sectors": [], "gainers": [], "losers": []}

    @staticmethod
    def search_symbols(query: str):
        try:
            import requests
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            res = requests.get(f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=5&newsCount=0", headers=headers, timeout=5)
            data = res.json()
            quotes = data.get("quotes", [])
            results = []
            for q in quotes:
                if q.get("quoteType") in ["EQUITY", "ETF", "MUTUALFUND", "INDEX", "CRYPTOCURRENCY"]:
                    results.append({
                        "symbol": q.get("symbol", ""),
                        "name": q.get("shortname", q.get("longname", "")),
                        "type": q.get("quoteType", ""),
                        "exchange": q.get("exchange", "")
                    })
            return results
        except Exception as e:
            print(f"Search error: {e}")
            return []

yfinance_service = YFinanceService()
