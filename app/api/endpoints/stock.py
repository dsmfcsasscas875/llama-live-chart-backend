from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from typing import List, Dict, Any
from app.services.yfinance_service import yfinance_service
from app.services.sentiment_service import sentiment_service
from app.services.prediction_service import prediction_service
import asyncio
import json

router = APIRouter()

@router.get("/stock/data/{symbol}")
def get_stock_data(symbol: str, period: str = "5y") -> Dict[str, Any]:
    """Fetch 5-year stock data"""
    try:
        data = yfinance_service.get_stock_data(symbol, period)
        return {"symbol": symbol, "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stock/sentiment/{symbol}")
async def get_sentiment(symbol: str) -> Dict[str, Any]:
    """Fetch and analyze news sentiment for a symbol using Google News & Groq"""
    try:
        # sentiment_service artık haberleri kendisi çekiyor ve analiz ediyor
        sentiment_data = await sentiment_service.analyze_news(symbol)
        
        if not sentiment_data:
            return {"symbol": symbol, "overall_sentiment": "Neutral", "sentiment_score": 0.5, "news": []}
            
        # Ortalama skoru hesapla (Sentiment score normalde 0-1 arası gelir)
        # 0.5 üstü pozitif, altı negatif kabul edilir
        scores = [n.get('sentiment_score', 0.5) for n in sentiment_data]
        avg_score = sum(scores) / len(scores) if scores else 0.5
        
        # Puanlama mantığı
        overall = "Neutral"
        if avg_score > 0.6: overall = "Positive"
        elif avg_score < 0.4: overall = "Negative"
        
        return {
            "symbol": symbol, 
            "overall_sentiment": overall, 
            "sentiment_score": round(avg_score, 2),
            "news": sentiment_data # UI'da görünecek haber listesi
        }
    except Exception as e:
        print(f"Sentiment API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stock/forecast/{symbol}")
def get_forecast(symbol: str, model: str = "gb") -> Dict[str, Any]:
    """Predict stock prices with historical backtest overlay"""
    try:
        historical_data = yfinance_service.get_stock_data(symbol, "5y")
        result = prediction_service.forecast_price(historical_data, 14, model_type=model, symbol=symbol)
        return {
            "symbol": symbol, 
            "forecast": result.get("forecast", []), 
            "backtest": result.get("backtest", []),
            "rmse": result.get("rmse", 0),
            "status": result.get("status"),
            "message": result.get("message"),
            "returns_forecast": result.get("returns_forecast", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/ws/stock/{symbol}")
async def websocket_stock(websocket: WebSocket, symbol: str):
    """Simulate real-time updates for a stock"""
    await websocket.accept()
    try:
        # Send initial summary immediately
        summary = yfinance_service.get_quote_summary(symbol)
        await websocket.send_json(summary)
        
        while True:
            # Simulate real-time price change (snapshot)
            summary = yfinance_service.get_quote_summary(symbol)
            await websocket.send_json(summary)
            # Wait 10 seconds before next update
            await asyncio.sleep(10)
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for {symbol}")
    except Exception as e:
        print(f"WebSocket Error: {e}")
        await websocket.close()
