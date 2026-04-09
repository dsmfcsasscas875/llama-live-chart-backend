from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship
from app.db.base_class import Base

class PortfolioItem(Base):
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True, nullable=False)
    shares = Column(Float, nullable=False)
    purchase_price = Column(Float, nullable=False)
    notes = Column(String, nullable=True)
    owner_id = Column(Integer, ForeignKey("user.id"))
    
    owner = relationship("User", back_populates="portfolio_items")

class PortfolioHistory(Base):
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True, nullable=False)
    action = Column(String, nullable=False) # "BUY" or "SELL"
    shares = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    timestamp = Column(String, nullable=False)
    owner_id = Column(Integer, ForeignKey("user.id"))

