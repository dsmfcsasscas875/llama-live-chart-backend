from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from app.services.yfinance_service import yfinance_service

router = APIRouter()

@router.get("/market/lists")
def get_market_lists() -> Dict[str, Any]:
    """Get Featured, Popular, and Latest stocks"""
    try:
        return yfinance_service.get_market_lists()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/market/sectors")
def get_sectors() -> List[str]:
    """Get all available sectors"""
    return yfinance_service.get_sectors()

@router.get("/market/sector/{sector}")
def get_stocks_by_sector(sector: str) -> List[Dict[str, Any]]:
    """Get stocks belonging to a specific sector"""
    try:
        return yfinance_service.get_stocks_by_sector(sector)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stock/recommend/{symbol}")
def get_recommendations(symbol: str) -> List[Dict[str, Any]]:
    """Get recommended similar stocks"""
    try:
        return yfinance_service.recommend_stocks(symbol)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/market/discover")
def get_discover() -> Dict[str, Any]:
    """Get gainers, losers, and sector trends for discovery"""
    try:
        return yfinance_service.get_discover_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/market/search")
def search_stocks(q: str) -> List[Dict[str, Any]]:
    """Search for stocks via Yahoo Finance API"""
    try:
        if not q or len(q) < 1:
            return []
        return yfinance_service.search_symbols(q)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
