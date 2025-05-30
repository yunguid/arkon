from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, Float, Boolean, Index
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class FinancialDocument(Base):
    __tablename__ = "financial_documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    upload_date = Column(DateTime, default=datetime.utcnow, index=True)
    summary_json = Column(Text)
    content = Column(Text, nullable=True)
    processing_status = Column(String, default="completed")
    user_id = Column(String, nullable=True, index=True)  # For future multi-user support
    
    __table_args__ = (
        Index('idx_user_upload_date', 'user_id', 'upload_date'),
    )

class StockNews(Base):
    __tablename__ = "stock_news"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    date = Column(DateTime, default=datetime.utcnow, index=True)
    perplexity_summary = Column(JSON)
    sentiment_score = Column(Float)
    source_urls = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_symbol_date', 'symbol', 'date'),
    )
    
    def __repr__(self):
        return f"<StockNews(symbol='{self.symbol}', date='{self.date}')>"

class Watchlist(Base):
    __tablename__ = "watchlists"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, default="default", index=True)
    symbol = Column(String, index=True)
    added_date = Column(DateTime, default=datetime.utcnow)
    last_analysis = Column(DateTime, nullable=True)
    price_alert_above = Column(Float, nullable=True)
    price_alert_below = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)
    
    __table_args__ = (
        Index('idx_user_symbol', 'user_id', 'symbol', unique=True),
    )
    
    def __repr__(self):
        return f"<Watchlist(symbol='{self.symbol}')>"

class StockAnalysisHistory(Base):
    __tablename__ = "stock_analysis_history"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    analysis_date = Column(DateTime, default=datetime.utcnow, index=True)
    perplexity_summary = Column(JSON)
    sentiment_score = Column(Float)
    source_urls = Column(JSON)
    
    __table_args__ = (
        Index('idx_symbol_analysis_date', 'symbol', 'analysis_date'),
    )
    
    def __repr__(self):
        return f"<StockAnalysisHistory(symbol='{self.symbol}', date='{self.analysis_date}')>"

class Budget(Base):
    __tablename__ = "budgets"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, default="default", index=True)
    category = Column(String, index=True)
    subcategory = Column(String, nullable=True)
    monthly_limit = Column(Float)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_user_category', 'user_id', 'category'),
    )

class CategoryCache(Base):
    __tablename__ = "category_cache"
    
    id = Column(Integer, primary_key=True, index=True)
    description = Column(String, unique=True, index=True)
    main_category = Column(String)
    sub_category = Column(String)
    detail_category = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used = Column(DateTime, default=datetime.utcnow)
    usage_count = Column(Integer, default=1)
    
    def __repr__(self):
        return f"<CategoryCache(description='{self.description}', category='{self.main_category}/{self.sub_category}')>"

class AlertLog(Base):
    __tablename__ = "alert_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, default="default", index=True)
    alert_type = Column(String)  # 'budget_exceeded', 'stock_price', etc.
    message = Column(Text)
    metadata = Column(JSON)
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index('idx_user_created', 'user_id', 'created_at'),
        Index('idx_user_unread', 'user_id', 'is_read'),
    ) 