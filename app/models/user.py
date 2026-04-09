from sqlalchemy import Column, Integer, String, Boolean
from sqlalchemy.orm import relationship
from app.db.base_class import Base

class User(Base):
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean(), default=True)
    
    # User Preferences
    email_alerts_enabled = Column(Boolean(), default=False)
    alert_time = Column(String, default="18:00")
    
    portfolio_items = relationship("PortfolioItem", back_populates="owner", cascade="all, delete-orphan")
    watchlist_items = relationship("WatchlistItem", back_populates="owner", cascade="all, delete-orphan")
