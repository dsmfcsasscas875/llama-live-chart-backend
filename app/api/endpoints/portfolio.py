from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from app.api import deps
from app.models.user import User
from app.models.portfolio import PortfolioItem, PortfolioHistory
import httpx
import json
from datetime import datetime
from app.services.yfinance_service import yfinance_service
import os
from dotenv import load_dotenv

from app.core.config import settings
load_dotenv()

router = APIRouter()

@router.get("/portfolio")
def get_portfolio(
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user)
) -> List[Dict[str, Any]]:
    """Get portfolio with current valuation and profit/loss"""
    holdings = db.query(PortfolioItem).filter(PortfolioItem.owner_id == current_user.id).all()
    if not holdings:
        return []
        
    symbols = list(set([h.symbol for h in holdings]))
    batch_data = yfinance_service.get_batch_quotes(symbols)
    
    results = []
    for item in holdings:
        data = batch_data.get(item.symbol, {})
        current_price = data.get('price', 0)
        
        total_cost = item.shares * item.purchase_price
        current_value = item.shares * current_price
        profit_loss = current_value - total_cost
        profit_loss_percent = (profit_loss / total_cost * 100) if total_cost > 0 else 0
        
        results.append({
            'id': item.id,
            'symbol': item.symbol,
            'shares': item.shares,
            'purchase_price': item.purchase_price,
            'notes': item.notes,
            'current_price': current_price,
            'current_value': current_value,
            'profit_loss': profit_loss,
            'profit_loss_percent': profit_loss_percent,
            'change_percent': data.get('change_percent', 0)
        })
    return results

@router.post("/portfolio")
def add_holding(
    data: Dict[str, Any],
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user)
):
    symbol = data['symbol'].upper()
    shares = float(data.get('shares', 0))
    purchase_price = float(data.get('purchase_price', 0))
    
    # Check if already exists for this user
    item = db.query(PortfolioItem).filter(
        PortfolioItem.symbol == symbol,
        PortfolioItem.owner_id == current_user.id
    ).first()
    
    if item:
        # Update existing
        total_shares = item.shares + shares
        if total_shares > 0:
            item.purchase_price = (item.purchase_price * item.shares + purchase_price * shares) / total_shares
            item.shares = total_shares
    else:
        # Create new
        item = PortfolioItem(
            symbol=symbol,
            shares=shares,
            purchase_price=purchase_price,
            owner_id=current_user.id
        )
    db.add(item)
    
    # Record history
    history = PortfolioHistory(
        symbol=symbol,
        action="BUY",
        shares=shares,
        price=purchase_price,
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        owner_id=current_user.id
    )
    db.add(history)
    
    db.commit()
    db.refresh(item)
    return {"status": "success", "message": f"Added {symbol} to portfolio"}

@router.delete("/portfolio/{symbol}")
def remove_holding(
    symbol: str,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user)
):
    item = db.query(PortfolioItem).filter(
        PortfolioItem.symbol == symbol.upper(),
        PortfolioItem.owner_id == current_user.id
    ).first()
    if not item:
        raise HTTPException(status_code=404, detail="Holding not found")
    
    # Record history
    history = PortfolioHistory(
        symbol=item.symbol,
        action="SELL",
        shares=item.shares,
        price=item.purchase_price, # Ideally this should be current market price, but we use this for now
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        owner_id=current_user.id
    )
    db.add(history)
    
    db.delete(item)
    db.commit()
    return {"status": "success", "message": f"Removed {symbol} from portfolio"}

@router.patch("/portfolio/{symbol}/notes")
def update_notes(
    symbol: str,
    data: Dict[str, Any],
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user)
):
    item = db.query(PortfolioItem).filter(PortfolioItem.symbol == symbol.upper(), PortfolioItem.owner_id == current_user.id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Holding not found")
    item.notes = data.get("notes", "")
    db.commit()
    return {"status": "success"}

@router.get("/portfolio/history")
def get_portfolio_history(
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user)
):
    history = db.query(PortfolioHistory).filter(PortfolioHistory.owner_id == current_user.id).order_by(PortfolioHistory.id.desc()).all()
    return [{"id": h.id, "symbol": h.symbol, "action": h.action, "shares": h.shares, "price": h.price, "timestamp": h.timestamp} for h in history]

@router.get("/portfolio/analysis")
async def analyze_portfolio(
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user)
):
    holdings = db.query(PortfolioItem).filter(PortfolioItem.owner_id == current_user.id).all()
    if not holdings:
        return {"analysis": "Portföyünüz boş. Başlamak için varlık ekleyin."}
    
    portfolio_str = ", ".join([f"{h.shares} units of {h.symbol} at ${h.purchase_price:.2f}" for h in holdings])
    system_prompt = "Sen uzman bir finansal danışmansın. Kullanıcının portföyündeki varlıkları incele ve Türkçe, kısa, profesyonel ama samimi bir değerlendirme yaz. Varlık çeşitliliği, olası riskler ve genel durum hakkında tek paragraf olsun."
    user_prompt = f"Portföyüm: {portfolio_str}"
    
    api_key = settings.GROQ_API_KEY
    if not api_key:
        return {"analysis": "AI Hizmeti şu an devre dışı (API Anahtarı eksik)."}

    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                    "temperature": 0.5
                },
                timeout=15.0
            )
            data = res.json()
            return {"analysis": data['choices'][0]['message']['content']}
    except Exception as e:
        return {"analysis": f"Portföy analiz edilemedi: {str(e)}"}

