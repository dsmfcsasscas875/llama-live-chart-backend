import json
import os
from typing import List, Set

class WatchlistService:
    def __init__(self, storage_path: str = "watchlist.json"):
        self.storage_path = storage_path
        self._ensure_storage()

    def _ensure_storage(self):
        if not os.path.exists(self.storage_path):
            with open(self.storage_path, 'w') as f:
                json.dump([], f)

    def get_watchlist(self) -> List[str]:
        try:
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def add_to_watchlist(self, symbol: str) -> List[str]:
        watchlist = set(self.get_watchlist())
        watchlist.add(symbol.upper())
        new_list = list(watchlist)
        self._save_watchlist(new_list)
        return new_list

    def remove_from_watchlist(self, symbol: str) -> List[str]:
        watchlist = set(self.get_watchlist())
        symbol_upper = symbol.upper()
        if symbol_upper in watchlist:
            watchlist.remove(symbol_upper)
        new_list = list(watchlist)
        self._save_watchlist(new_list)
        return new_list

    def _save_watchlist(self, watchlist: List[str]):
        with open(self.storage_path, 'w') as f:
            json.dump(watchlist, f)

watchlist_service = WatchlistService()
