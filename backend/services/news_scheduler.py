from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
import asyncio
from typing import List

class NewsScheduler:
    def __init__(self, collector, symbols: List[str]):
        self.collector = collector
        self.symbols = symbols
        self.scheduler = AsyncIOScheduler()
    
    def start(self):
        # Run at market close (4 PM EST)
        self.scheduler.add_job(
            self._collect_all,
            CronTrigger(hour=16, timezone='America/New_York'),
            id='daily_news_collection'
        )
        self.scheduler.start()
    
    async def _collect_all(self):
        for symbol in self.symbols:
            try:
                await self.collector.collect_daily_news(symbol)
            except Exception as e:
                print(f"Failed to collect news for {symbol}: {e}") 