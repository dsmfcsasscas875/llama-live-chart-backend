from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

db_uri = settings.sync_database_uri
connect_args = {}
if db_uri.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(db_uri, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
