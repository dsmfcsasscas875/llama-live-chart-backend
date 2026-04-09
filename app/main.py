from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import stock, market, watchlist, portfolio, auth
from app.core.config import settings
from app.db.base import Base
from app.db.session import engine
from app.services.scheduler import scheduler_service

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    description="Stock Analysis Showcase with FinnBERT & yfinance"
)

@app.on_event("startup")
def on_startup():
    scheduler_service.start()

@app.on_event("shutdown")
def on_shutdown():
    scheduler_service.shutdown()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        settings.FRONTEND_URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include endpoints
app.include_router(auth.router, prefix=settings.API_V1_STR, tags=["Authentication"])
app.include_router(stock.router, prefix=settings.API_V1_STR, tags=["Stock Analysis"])
app.include_router(market.router, prefix=settings.API_V1_STR, tags=["Market Monitoring"])
app.include_router(watchlist.router, prefix=settings.API_V1_STR, tags=["User Watchlist"])
app.include_router(portfolio.router, prefix=settings.API_V1_STR, tags=["Portfolio Management"])

@app.get("/")
def read_root():
    return {"message": "Welcome to Live Chart App API", "status": "online"}
