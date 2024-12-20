from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON, ForeignKey, Float
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from datetime import datetime
from typing import Optional, Dict

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
    key_developments = Column(JSON)  # Add this
    price_impact = Column(Text)      # Add this
    risks = Column(JSON)             # Add this
    
    def to_dict(self):
        """Convert model instance to dictionary"""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "date": self.date,
            "summary": self.perplexity_summary,
            "sentiment": self.sentiment_score,
            "sources": self.source_urls,
            "key_developments": self.key_developments,
            "price_impact": self.price_impact,
            "risks": self.risks
        }
    
    def __repr__(self):
        return f"<StockNews(symbol='{self.symbol}', date='{self.date}')>"

class WatchlistStock(Base):
    __tablename__ = "watchlist_stocks"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, unique=True, index=True)
    company_name = Column(String)
    sector = Column(String)
    added_date = Column(DateTime, default=datetime.utcnow)
    last_analyzed = Column(DateTime, nullable=True)
    last_analysis = Column(DateTime, nullable=True)
    cik = Column(String)
    
    # Relationships
    filings = relationship("StockFiling", back_populates="stock")
    metrics = relationship("FinancialMetrics", back_populates="stock")

class StockFiling(Base):
    __tablename__ = "stock_filings"
    
    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("watchlist_stocks.id"))
    filing_type = Column(String)
    filing_date = Column(DateTime)
    filing_url = Column(String)
    filing_data = Column(JSON)
    
    # Relationship
    stock = relationship("WatchlistStock", back_populates="filings")

class FinancialMetrics(Base):
    __tablename__ = "financial_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("watchlist_stocks.id"))
    date = Column(DateTime, default=datetime.utcnow)
    metrics = Column(JSON)
    
    # Relationship
    stock = relationship("WatchlistStock", back_populates="metrics")

class StockAnalysisHistory(Base):
    __tablename__ = "stock_analysis_history"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    analysis_date = Column(DateTime, default=datetime.utcnow)
    perplexity_summary = Column(JSON)
    sentiment_score = Column(Float)
    source_urls = Column(JSON)
    
    def to_dict(self):
        """Convert model instance to dictionary"""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "date": self.analysis_date,
            "summary": self.perplexity_summary,
            "sentiment": self.sentiment_score,
            "sources": self.source_urls
        }
    
    def __repr__(self):
        return f"<StockAnalysisHistory(symbol='{self.symbol}', date='{self.analysis_date}')>"

class StockNewsRepository:
    def __init__(self, session):
        self.session = session

    async def get_latest_news(self, symbol: str) -> Optional[Dict]:
        """Get latest news analysis for a symbol"""
        result = self.session.query(StockNews)\
            .filter(StockNews.symbol == symbol)\
            .order_by(StockNews.date.desc())\
            .first()
        return result.to_dict() if result else None

    async def update_latest_news(self, symbol: str, analysis: Dict) -> None:
        """Update or create latest news analysis"""
        news = StockNews(
            symbol=symbol,
            date=datetime.utcnow(),
            perplexity_summary=analysis.get('summary'),
            sentiment_score=self._calculate_sentiment(analysis.get('market_sentiment')),
            source_urls=analysis.get('source_urls', []),
            key_developments=analysis.get('key_developments'),
            price_impact=analysis.get('price_impact'),
            risks=analysis.get('risks', [])
        )
        self.session.add(news)
        self.session.commit()

    async def add_to_history(self, symbol: str, analysis: Dict) -> None:
        """Move analysis to history"""
        history = StockAnalysisHistory(
            symbol=symbol,
            analysis_date=datetime.utcnow(),
            perplexity_summary=analysis.get('summary'),
            sentiment_score=analysis.get('sentiment'),
            source_urls=analysis.get('sources', [])
        )
        self.session.add(history)
        self.session.commit()

    def _calculate_sentiment(self, sentiment: str) -> float:
        sentiment_map = {
            "positive": 1.0,
            "neutral": 0.0,
            "negative": -1.0
        }
        return sentiment_map.get(sentiment.lower() if sentiment else "neutral", 0.0)

def init_db():
    """Initialize the database, creating all tables if they don't exist"""
    # Drop existing tables
    Base.metadata.drop_all(bind=engine)
    # Create tables with new schema
    Base.metadata.create_all(bind=engine)

# Remove or comment out this part since we're using init_db function now
# if __name__ == '__main__':
#     Base.metadata.create_all(bind=engine) 