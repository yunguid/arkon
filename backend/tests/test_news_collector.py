import asyncio
import pytest
from datetime import datetime
from services.news_scraper import PerplexityNewsAnalyzer, NewsCollector
from services.news_scheduler import NewsScheduler
from database import SessionLocal, StockNews
import os

@pytest.mark.asyncio
async def test_news_collection():
    # Initialize components
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        pytest.skip("PERPLEXITY_API_KEY not set")
    
    analyzer = PerplexityNewsAnalyzer(api_key)
    db = SessionLocal()
    collector = NewsCollector(db, analyzer)
    
    # Test single symbol collection
    symbol = "AAPL"
    await collector.collect_daily_news(symbol)
    
    # Verify data was stored
    news = db.query(StockNews).filter(StockNews.symbol == symbol).first()
    assert news is not None
    assert news.symbol == symbol
    assert news.perplexity_summary is not None
    assert isinstance(news.sentiment_score, float)
    
    db.close()

@pytest.mark.asyncio
async def test_scheduler():
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        pytest.skip("PERPLEXITY_API_KEY not set")
    
    analyzer = PerplexityNewsAnalyzer(api_key)
    db = SessionLocal()
    collector = NewsCollector(db, analyzer)
    
    scheduler = NewsScheduler(collector, symbols=["AAPL", "MSFT"])
    
    # Test immediate collection
    await scheduler._collect_all()
    
    # Verify multiple symbols
    news_count = db.query(StockNews).count()
    assert news_count >= 2
    
    db.close() 

@pytest.mark.asyncio
async def test_perplexity_analysis():
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        pytest.skip("PERPLEXITY_API_KEY not set")
    
    analyzer = PerplexityNewsAnalyzer(api_key)
    
    # Test with sample articles
    articles = [{
        "title": "Test Article",
        "summary": "This is a test article about stocks",
        "url": "http://example.com",
        "publisher": "Test Publisher",
        "published": "2024-03-14"
    }]
    
    result = await analyzer.analyze_stock_news("TEST", articles)
    
    assert isinstance(result, dict)
    assert "key_developments" in result
    assert "market_sentiment" in result
    assert "price_impact" in result 