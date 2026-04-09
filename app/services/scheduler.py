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
    """Background job to train the AI models once a day"""
    print("Executing scheduled task: AI Model Training")
    # Train for popular stocks as a start
    symbols = ["NVDA", "AAPL", "TSLA", "MSFT", "GOOGL"]
    hybrid_prediction_service.train_daily_all(symbols)

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

        # 3. Trigger initial training at startup in a separate thread to not block FastAPI
        threading.Thread(target=job_train_models, daemon=True).start()

        self.scheduler.start()
        print("Scheduler started. Background tasks are running.")
        
    def shutdown(self):
        self.scheduler.shutdown()

scheduler_service = WatchlistScheduler()
