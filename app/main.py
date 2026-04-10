from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from app.api.endpoints import stock, market, watchlist, portfolio, auth
from app.core.config import settings
from app.db.base import Base
from app.db.session import engine
from app.services.scheduler import scheduler_service

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    description="Stock Analysis Showcase with FinnBERT & yfinance"
)

@app.on_event("startup")
def on_startup():
    print("Initializing Database Tables...")
    try:
        Base.metadata.create_all(bind=engine)
        print("Database Tables Synced Successfully.")
    except Exception as e:
        print(f"Error creating tables: {e}")
    scheduler_service.start()

@app.on_event("shutdown")
def on_shutdown():
    scheduler_service.shutdown()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "https://exchangemonster.vercel.app",
        "https://*.vercel.app",
        "*" # Temporary for ease of deployment, change to specific domains later
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["localhost", "127.0.0.1", "*.railway.app", "exchangemonster.vercel.app", "*.vercel.app"]
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred. Our engineers are notified.", "type": "INTERNAL_ERROR"},
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "type": "VALIDATION_ERROR"},
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
