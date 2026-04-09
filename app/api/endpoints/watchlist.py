from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.api import deps
from app.models.user import User
from app.models.watchlist import WatchlistItem
from app.services.yfinance_service import yfinance_service

router = APIRouter()

class NotesUpdate(BaseModel):
    notes: str

@router.get("/watchlist")
def get_watchlist(
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user)
) -> List[Dict[str, Any]]:
    """Get watchlist with current prices and notes"""
    items = db.query(WatchlistItem).filter(WatchlistItem.owner_id == current_user.id).all()
    if not items:
        return []
        
    symbols = [item.symbol for item in items]
    batch_data = yfinance_service.get_batch_quotes(symbols)
    
    results = []
    for item in items:
        data = batch_data.get(item.symbol, {"symbol": item.symbol, "price": 0, "change_percent": 0})
        results.append({**data, "id": item.id, "notes": item.notes or ""})
        
    return results

@router.post("/watchlist/{symbol}")
def add_to_watchlist(
    symbol: str,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user)
):
    """Add a symbol to the local watchlist"""
    symbol = symbol.upper()
    exists = db.query(WatchlistItem).filter(
        WatchlistItem.symbol == symbol,
        WatchlistItem.owner_id == current_user.id
    ).first()
    
    if exists:
        return {"status": "success", "message": f"{symbol} already in watchlist"}
    
    item = WatchlistItem(symbol=symbol, owner_id=current_user.id, notes="")
    db.add(item)
    db.commit()
    return {"status": "success", "message": f"Added {symbol} to watchlist"}

@router.delete("/watchlist/{symbol}")
def remove_from_watchlist(
    symbol: str,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user)
):
    """Remove a symbol from the local watchlist"""
    item = db.query(WatchlistItem).filter(
        WatchlistItem.symbol == symbol.upper(),
        WatchlistItem.owner_id == current_user.id
    ).first()
    
    if not item:
        # Try case-insensitive search
        item = db.query(WatchlistItem).filter(
            WatchlistItem.symbol.ilike(symbol),
            WatchlistItem.owner_id == current_user.id
        ).first()
    
    if not item:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found in watchlist")
    
    db.delete(item)
    db.commit()
    return {"status": "success", "message": f"Removed {symbol} from watchlist"}

@router.patch("/watchlist/{symbol}/notes")
def update_watchlist_notes(
    symbol: str,
    body: NotesUpdate,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user)
):
    """Update notes for a watchlist item"""
    item = db.query(WatchlistItem).filter(
        WatchlistItem.symbol == symbol.upper(),
        WatchlistItem.owner_id == current_user.id
    ).first()
    
    if not item:
        raise HTTPException(status_code=404, detail="Symbol not found in watchlist")
    
    item.notes = body.notes
    db.commit()
    db.refresh(item)
    return {"status": "success", "symbol": item.symbol, "notes": item.notes}
