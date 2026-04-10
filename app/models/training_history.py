from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Text
from sqlalchemy.sql import func
from app.db.base_class import Base


class TrainingHistory(Base):
    """
    Stores per-symbol model training runs with metrics and status.
    Each row represents one completed (or failed) training attempt.
    """
    __tablename__ = "training_history"

    id             = Column(Integer, primary_key=True, index=True)
    symbol         = Column(String(20), index=True, nullable=False)
    training_date  = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    model_version  = Column(String(64), nullable=False)   # e.g. "NVDA_20240101_120000"
    metrics        = Column(JSON, nullable=True)           # {"val_loss": [...], "best_val_loss": [...]}
    status         = Column(String(20), nullable=False, default="pending")  # pending | success | failed
    error_message  = Column(Text, nullable=True)


class ModelCheckpoint(Base):
    """
    Tracks the file-system path of every saved model checkpoint.
    The latest successful checkpoint per symbol is used for inference.
    """
    __tablename__ = "model_checkpoint"

    id               = Column(Integer, primary_key=True, index=True)
    symbol           = Column(String(20), index=True, nullable=False)
    model_version    = Column(String(64), nullable=False)
    checkpoint_path  = Column(String(512), nullable=False)  # absolute path to .pt file
    created_at       = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
