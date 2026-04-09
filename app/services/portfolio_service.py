import json
import os
from typing import List, Dict, Any

class PortfolioService:
    def __init__(self, storage_path: str = "portfolio.json"):
        self.storage_path = storage_path
        self._ensure_storage()

    def _ensure_storage(self):
        if not os.path.exists(self.storage_path):
            with open(self.storage_path, 'w') as f:
                json.dump([], f)

    def get_portfolio(self) -> List[Dict[str, Any]]:
        try:
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def add_to_portfolio(self, symbol: str, shares: float, purchase_price: float):
        portfolio = self.get_portfolio()
        # Check if already exists, then update
        for item in portfolio:
            if item['symbol'].upper() == symbol.upper():
                item['shares'] += shares
                # Weighted average purchase price (simplified)
                item['purchase_price'] = (item['purchase_price'] + purchase_price) / 2
                self._save_portfolio(portfolio)
                return portfolio
        
        portfolio.append({
            "symbol": symbol.upper(),
            "shares": shares,
            "purchase_price": purchase_price
        })
        self._save_portfolio(portfolio)
        return portfolio

    def remove_from_portfolio(self, symbol: str):
        portfolio = self.get_portfolio()
        portfolio = [item for item in portfolio if item['symbol'].upper() != symbol.upper()]
        self._save_portfolio(portfolio)
        return portfolio

    def _save_portfolio(self, portfolio: List[Dict[str, Any]]):
        with open(self.storage_path, 'w') as f:
            json.dump(portfolio, f)

portfolio_service = PortfolioService()
