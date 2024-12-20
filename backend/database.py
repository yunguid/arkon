from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON, ForeignKey, Float
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from datetime import datetime

SQLALCHEMY_DATABASE_URL = "sqlite:///./financial_docs.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class FinancialDocument(Base):
    __tablename__ = "financial_documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    upload_date = Column(DateTime, default=datetime.utcnow)
    summary_json = Column(Text)
    content = Column(Text, nullable=True)

class StockNews(Base):
    __tablename__ = "stock_news"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    date = Column(DateTime, default=datetime.utcnow)
    perplexity_summary = Column(JSON)  # Stores the analyzed data
    sentiment_score = Column(Float)
    source_urls = Column(JSON)  # Store original article URLs
    
    def __repr__(self):
        return f"<StockNews(symbol='{self.symbol}', date='{self.date}')>"

class Watchlist(Base):
    __tablename__ = "watchlists"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)  # For future auth
    symbol = Column(String, index=True)
    added_date = Column(DateTime, default=datetime.utcnow)
    last_analysis = Column(DateTime, nullable=True)
    
    def __repr__(self):
        return f"<Watchlist(symbol='{self.symbol}')>"

class StockAnalysisHistory(Base):
    __tablename__ = "stock_analysis_history"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    analysis_date = Column(DateTime, default=datetime.utcnow)
    perplexity_summary = Column(JSON)
    sentiment_score = Column(Float)
    source_urls = Column(JSON)
    
    def __repr__(self):
        return f"<StockAnalysisHistory(symbol='{self.symbol}', date='{self.analysis_date}')>"

def init_db():
    """Initialize the database, creating all tables if they don't exist"""
    Base.metadata.create_all(bind=engine)

# Remove or comment out this part since we're using init_db function now
# if __name__ == '__main__':
#     Base.metadata.create_all(bind=engine) 