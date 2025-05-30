from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base

SQLALCHEMY_DATABASE_URL = "sqlite:///./financial_docs.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Import all models to ensure they're registered with Base
from models import (
    FinancialDocument, StockNews, Watchlist, StockAnalysisHistory,
    Budget, CategoryCache, AlertLog
)

# Initialize database
def init_db():
    Base.metadata.create_all(bind=engine)

if __name__ == '__main__':
    init_db() 