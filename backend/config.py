"""
Configuration settings for Arkon Financial Analyzer
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings"""
    
    # App Info
    APP_NAME: str = "Arkon Financial Analyzer"
    APP_VERSION: str = "2.0.0"
    APP_DESCRIPTION: str = "AI-Powered Financial Intelligence Platform"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    RELOAD: bool = False
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", SECRET_KEY)
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24
    
    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "postgresql+asyncpg://user:password@localhost/arkon"
    )
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 40
    
    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    REDIS_CACHE_TTL: int = 3600
    
    # Elasticsearch
    ELASTICSEARCH_URL: str = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
    
    # Google OAuth
    GOOGLE_CLIENT_ID: str = os.getenv("GOOGLE_CLIENT_ID", "")
    GOOGLE_CLIENT_SECRET: str = os.getenv("GOOGLE_CLIENT_SECRET", "")
    GOOGLE_REDIRECT_URI: str = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:3000/auth/google/callback")
    
    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_ORG_ID: Optional[str] = os.getenv("OPENAI_ORG_ID", None)
    OPENAI_MODEL: str = "gpt-4o-realtime-preview"
    
    # AWS
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "arkon-financial")
    
    # Email
    SENDGRID_API_KEY: str = os.getenv("SENDGRID_API_KEY", "")
    EMAIL_FROM: str = os.getenv("EMAIL_FROM", "noreply@arkon.finance")
    
    # Blockchain
    WEB3_PROVIDER_URL: str = os.getenv("WEB3_PROVIDER_URL", "https://mainnet.infura.io/v3/YOUR-PROJECT-ID")
    PRIVATE_KEY: str = os.getenv("PRIVATE_KEY", "")
    CONTRACT_ADDRESS: str = os.getenv("CONTRACT_ADDRESS", "")
    
    # Monitoring
    SENTRY_DSN: Optional[str] = os.getenv("SENTRY_DSN", None)
    PROMETHEUS_ENABLED: bool = True
    
    # Features
    ENABLE_VOICE_ASSISTANT: bool = True
    ENABLE_BLOCKCHAIN: bool = True
    ENABLE_ML_PREDICTIONS: bool = True
    ENABLE_REAL_TIME: bool = True
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000
    
    # File Upload
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_UPLOAD_EXTENSIONS: list = [".pdf", ".png", ".jpg", ".jpeg", ".csv", ".xlsx"]
    
    # ML Models
    ML_MODEL_PATH: str = os.getenv("ML_MODEL_PATH", "./models")
    ML_UPDATE_INTERVAL: int = 3600  # 1 hour
    
    # Voice Assistant
    VOICE_SAMPLE_RATE: int = 24000
    VOICE_CHUNK_SIZE: int = 1024
    VOICE_LANGUAGE: str = "en-US"
    
    # Celery
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", REDIS_URL)
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)
    
    # CORS
    CORS_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:8000",
        "https://arkon.finance",
    ]
    
    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Create settings instance
settings = Settings()


# Validate critical settings
def validate_settings():
    """Validate that critical settings are configured"""
    errors = []
    
    if settings.ENVIRONMENT == "production":
        if settings.SECRET_KEY == "your-secret-key-here":
            errors.append("SECRET_KEY must be set in production")
            
        if not settings.GOOGLE_CLIENT_ID:
            errors.append("GOOGLE_CLIENT_ID must be set for authentication")
            
        if not settings.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY must be set for voice assistant")
            
        if not settings.DATABASE_URL.startswith("postgresql"):
            errors.append("PostgreSQL database required in production")
            
    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")


# Run validation
if settings.ENVIRONMENT == "production":
    validate_settings() 