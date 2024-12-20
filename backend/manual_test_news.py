import asyncio
import os
from services.news_scraper import PerplexityNewsAnalyzer, NewsCollector
from database import SessionLocal
from dotenv import load_dotenv

async def main():
    load_dotenv()
    
    # Initialize components
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        print("Error: PERPLEXITY_API_KEY not set")
        return
    
    print("Initializing news collector...")
    analyzer = PerplexityNewsAnalyzer(api_key)
    db = SessionLocal()
    collector = NewsCollector(db, analyzer)
    
    # Test collection for a single symbol
    symbol = "AAPL"
    print(f"Collecting news for {symbol}...")
    
    try:
        await collector.collect_daily_news(symbol)
        print("News collection successful!")
        
        # Print the collected data
        from database import StockNews
        news = db.query(StockNews).filter(StockNews.symbol == symbol).first()
        if news:
            print("\nCollected Data:")
            print(f"Date: {news.date}")
            print(f"Sentiment Score: {news.sentiment_score}")
            print("\nSummary:")
            print(news.perplexity_summary)
            print("\nSources:")
            print(news.source_urls)
        else:
            print("No news data found")
            
    except Exception as e:
        print(f"Error during collection: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(main()) 