from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy.orm import Session
from app.db.session import SessionLocal
from app.models.user import User
from app.models.watchlist import WatchlistItem
from app.services.yfinance_service import yfinance_service
from app.services.email_service import email_service
from app.services.hybrid_prediction_service import hybrid_prediction_service
import pytz
import threading
import logging

logger = logging.getLogger(__name__)

# Symbols managed by the daily training job
MANAGED_SYMBOLS = ["NVDA", "AAPL", "TSLA", "MSFT", "GOOGL"]

def job_send_watchlist_updates():
    """Background job to send watchlist updates via email"""
    print("Executing scheduled task: Watchlist Updates")
    db: Session = SessionLocal()
    try:
        users = db.query(User).filter(User.is_active == True).all()
        for user in users:
            items = db.query(WatchlistItem).filter(WatchlistItem.owner_id == user.id).all()
            if not items:
                continue
                
            watchlist_data = []
            for item in items:
                try:
                    summary = yfinance_service.get_quote_summary(item.symbol)
                    watchlist_data.append(summary)
                except Exception as e:
                    print(f"Skipping {item.symbol} for email: {e}")
                    
            if watchlist_data:
                email_service.send_watchlist_alert(user.email, watchlist_data)
                
    except Exception as e:
        print(f"Scheduler error: {e}")
    finally:
        db.close()

def job_train_models():
    """Background job to re-train the AI models once a day."""
    logger.info("Executing scheduled task: AI Model Training")
    hybrid_prediction_service.train_daily_all(MANAGED_SYMBOLS)


def job_load_models_at_startup():
    """
    Runs once at startup (in a background thread so FastAPI is not blocked).

    Strategy:
      1. Try to restore the latest checkpoint for every managed symbol from DB.
      2. For any symbol that has no checkpoint yet, kick off a fresh training run.
         If that training fails the symbol simply has no in-memory model until
         the next daily job succeeds (graceful degradation).
    """
    logger.info("Startup: loading persisted model checkpoints …")
    results = hybrid_prediction_service.load_all_models(MANAGED_SYMBOLS)

    symbols_to_train = [sym for sym, loaded in results.items() if not loaded]
    if symbols_to_train:
        logger.info(
            f"Startup: no checkpoint found for {symbols_to_train} — "
            "starting initial training …"
        )
        for sym in symbols_to_train:
            try:
                hybrid_prediction_service.train_ensemble(sym)
            except Exception as exc:
                # Training failure must not crash the startup thread
                logger.error(
                    f"Startup: initial training for {sym} failed: {exc}",
                    exc_info=True,
                )
    else:
        logger.info("Startup: all model checkpoints restored successfully.")

class WatchlistScheduler:
    def __init__(self):
        self.scheduler = BackgroundScheduler(timezone=pytz.UTC)
        
    def start(self):
        # 1. Watchlist Updates
        self.scheduler.add_job(
            job_send_watchlist_updates,
            CronTrigger(hour=18, minute=0, timezone=pytz.UTC),
            id='watchlist_daily_job'
        )

        # 2. Model Training (Once a day at midnight UTC)
        self.scheduler.add_job(
            job_train_models,
            CronTrigger(hour=0, minute=0, timezone=pytz.UTC),
            id='model_training_daily_job'
        )

        # 3. At startup: restore checkpoints from DB, train only what is missing.
        #    Runs in a daemon thread so FastAPI startup is not blocked.
        threading.Thread(target=job_load_models_at_startup, daemon=True).start()

        self.scheduler.start()
        logger.info("Scheduler started. Background tasks are running.")
        
    def shutdown(self):
        self.scheduler.shutdown()

scheduler_service = WatchlistScheduler()
