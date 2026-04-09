from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Live Stock Showcase"
    API_V1_STR: str = "/api/v1"
    
    # Database
    DATABASE_URL: str | None = None
    SQLALCHEMY_DATABASE_URI: str = "sqlite:///./sql_app.db"
    
    @property
    def sync_database_uri(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL.replace("postgres://", "postgresql://")
        return self.SQLALCHEMY_DATABASE_URI
    
    FRONTEND_URL: str = "http://localhost:3000"
    
    # Security
    SECRET_KEY: str = "supersecretkeychangeinproduction"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30 * 24 * 60  # 30 days
    
    # External APIs
    RESEND_API_KEY: str | None = None
    GROQ_API_KEY: str | None = None
    TWELVE_DATA_API_KEY: str | None = None
    
    # Model configuration
    SENTIMENT_MODEL: str = "ProsusAI/finbert"
    STOCK_PERIOD: str = "5y"
    
    class Config:
        case_sensitive = True

settings = Settings()
