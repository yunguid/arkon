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

# Move this to a separate initialization script
if __name__ == '__main__':
    Base.metadata.create_all(bind=engine) 