import os
from datetime import datetime
import httpx
from typing import Dict, List, Optional
import json
import asyncio
from bs4 import BeautifulSoup
import yfinance as yf
import logging
from openai import OpenAI
from database import StockNews, StockAnalysisHistory

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class PerplexityNewsAnalyzer:
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.perplexity.ai"
        )
    
    def _clean_json_string(self, json_str: str) -> str:
        """Clean and validate JSON string"""
        # Remove markdown backticks and 'json' tag
        json_str = json_str.replace('```json', '').replace('```', '').strip()
        
        # Fix common JSON issues
        json_str = json_str.replace('\n', ' ')  # Remove newlines
        
        # Fix nested quotes - replace \" with single quotes
        json_str = json_str.replace('\\"', "'")
        
        # Handle any remaining escaped backslashes
        json_str = json_str.replace('\\', '\\\\')
        
        # Fix trailing commas
        json_str = json_str.replace(',}', '}').replace(',]', ']')
        
        logger.debug(f"Cleaned JSON string: {json_str}")
        return json_str

    async def analyze_stock_news(self, symbol: str, articles: List[Dict]) -> Dict:
        try:
            logger.info(f"Starting Perplexity analysis for {symbol}")
            
            prompt = f"""
            Analyze these news articles about {symbol} and return a JSON object.
            
            Rules:
            1. Response must be ONLY valid JSON
            2. No trailing commas
            3. Use single quotes (') for quoted text within strings
            4. DO NOT include citation numbers like [1] or [2] in the text
            5. No comments or additional text
            
            Example format:
            {{
                "key_developments": "Company was rated as buy by analysts",
                "market_sentiment": "positive",
                "price_impact": "Stock rose 5% after strong buy rating",
                "risks": ["Competition from major players"],
                "expert_quotes": ["Analyst says very bullish"],
                "summary": "Overall positive with strong outlook"
            }}
            
            Articles: {json.dumps(articles)}
            """
            
            response = self.client.chat.completions.create(
                model="llama-3.1-sonar-huge-128k-online",
                messages=[
                    {"role": "system", "content": "You are a financial analyst. Return only valid JSON. Do not include citation numbers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            logger.debug(f"Raw API response: {content}")
            
            # Clean and parse JSON
            try:
                cleaned_content = self._clean_json_string(content)
                logger.info(f"Cleaned JSON string: {cleaned_content}")
                result = json.loads(cleaned_content)
                
                # Validate required fields
                required_fields = {
                    "key_developments": str,
                    "market_sentiment": str,
                    "price_impact": str,
                    "risks": list,
                    "expert_quotes": list,
                    "summary": str
                }
                
                for field, field_type in required_fields.items():
                    if field not in result or not isinstance(result[field], field_type):
                        result[field] = [] if field_type == list else "Not available"
                
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {e}")
                # Return a valid fallback structure
                return {
                    "key_developments": "Error parsing analysis",
                    "market_sentiment": "neutral",
                    "price_impact": "Analysis unavailable",
                    "risks": ["Error processing risks"],
                    "expert_quotes": [],
                    "summary": f"Failed to analyze news content: {str(e)}"
                }
                
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return {
                "key_developments": str(e),
                "market_sentiment": "neutral",
                "price_impact": "Analysis failed",
                "risks": ["Analysis error"],
                "expert_quotes": [],
                "summary": "Analysis failed: " + str(e)
            }

class NewsCollector:
    def __init__(self, db, analyzer):
        self.db = db
        self.analyzer = analyzer
        self._news_cache = {}
        self._cache_ttl = 300  # 5 minutes

    async def get_news(self, symbol: str) -> List[Dict]:
        now = datetime.now().timestamp()
        if symbol in self._news_cache:
            cached_time, cached_news = self._news_cache[symbol]
            if now - cached_time < self._cache_ttl:
                return cached_news

        ticker = yf.Ticker(symbol)
        news = ticker.news[:5] if ticker.news else []
        logger.info(f"News: {news}")
        self._news_cache[symbol] = (now, news)
        return news

    async def collect_daily_news(self, symbol: str) -> Dict:
        try:
            current = await self.db.get_latest_news(symbol)
            if current:
                await self.db.add_to_history(symbol, current)
                
            news = await self.get_news(symbol)
            
            # Extract source URLs before analysis
            source_urls = [article.get('link') for article in news if article.get('link')]
            
            # Get analysis without the URLs
            analysis = await self.analyzer.analyze_stock_news(symbol, news)
            
            # Add source URLs separately
            analysis['source_urls'] = source_urls
            
            await self.db.update_latest_news(symbol, analysis)
            return analysis
            
        except Exception as e:
            logger.error(f"News collection failed: {e}")
            raise
    
    def _calculate_sentiment(self, analysis: Dict) -> float:
        # Simple sentiment calculation
        sentiment_map = {
            "positive": 1.0,
            "neutral": 0.0,
            "negative": -1.0
        }
        try:
            sentiment = analysis.get("market_sentiment", "neutral").lower()
            return sentiment_map.get(sentiment, 0.0)
        except:
            return 0.0