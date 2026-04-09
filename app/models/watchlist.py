from sqlalchemy import Column, Integer, String, ForeignKey, Text
from sqlalchemy.orm import relationship
from app.db.base_class import Base

class WatchlistItem(Base):
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True, nullable=False)
    owner_id = Column(Integer, ForeignKey("user.id"))
    notes = Column(Text, nullable=True, default="")
    
    owner = relationship("User", back_populates="watchlist_items")
