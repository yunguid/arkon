import asyncio
import os
from services.news_scraper import PerplexityNewsAnalyzer, NewsCollector
from services.news_scheduler import NewsScheduler
from database import SessionLocal
from dotenv import load_dotenv

async def main():
    load_dotenv()
    
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        print("Error: PERPLEXITY_API_KEY not set")
        return
    
    analyzer = PerplexityNewsAnalyzer(api_key)
    db = SessionLocal()
    collector = NewsCollector(db, analyzer)
    
    # Initialize scheduler with test symbols
    scheduler = NewsScheduler(collector, symbols=["AAPL", "MSFT", "GOOGL"])
    
    print("Starting scheduler...")
    scheduler.start()
    
    # Keep the script running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping scheduler...")
        scheduler.scheduler.shutdown()
        db.close()

if __name__ == "__main__":
    asyncio.run(main()) 