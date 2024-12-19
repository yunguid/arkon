import os
from datetime import datetime
import httpx
from typing import Dict, List
import json
import asyncio
from bs4 import BeautifulSoup
import yfinance as yf
import logging
from openai import OpenAI
from database import StockNews

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class PerplexityNewsAnalyzer:
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.perplexity.ai"
        )
    
    async def analyze_stock_news(self, symbol: str, articles: List[Dict]) -> Dict:
        try:
            logger.info(f"Starting Perplexity analysis for {symbol}")
            prompt = self._create_analysis_prompt(symbol, articles)
            
            response = self.client.chat.completions.create(
                model="llama-3.1-sonar-huge-128k-online",
                messages=[
                    {"role": "system", "content": "You are a financial analyst. Provide concise JSON responses."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            logger.info("Raw API Response Content:")
            logger.info("-" * 80)
            logger.info(content)
            logger.info("-" * 80)
            
            # Extract JSON from response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON object found in response")
                
            json_str = content[json_start:json_end]
            
            # Parse and validate
            try:
                result = json.loads(json_str)
                required_fields = ["key_developments", "market_sentiment", "price_impact", "risks", "expert_quotes", "summary"]
                for field in required_fields:
                    if field not in result:
                        result[field] = "Not available" if field not in ["risks", "expert_quotes"] else []
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {e}")
                raise ValueError(f"Invalid JSON format: {e}")
                
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
    
    def _create_analysis_prompt(self, symbol: str, articles: List[Dict]) -> str:
        return f"""
        Analyze these news articles about {symbol} and return a JSON object with this exact structure:
        {{
            "key_developments": "Brief summary of main points",
            "market_sentiment": "positive/negative/neutral",
            "price_impact": "Brief price impact analysis",
            "risks": ["Risk 1", "Risk 2"],
            "expert_quotes": ["Quote 1", "Quote 2"],
            "summary": "Overall summary"
        }}
        
        Articles: {json.dumps(articles)}
        
        Important: Return ONLY the JSON object, no other text.
        """

class NewsCollector:
    def __init__(self, db, analyzer):
        self.db = db
        self.analyzer = analyzer
    
    async def collect_daily_news(self, symbol: str):
        try:
            logger.info(f"Starting news collection for {symbol}")
            ticker = yf.Ticker(symbol)
            news_data = ticker.news
            
            logger.info(f"Found {len(news_data) if news_data else 0} news items for {symbol}")
            
            if not news_data:
                logger.warning(f"No news found for {symbol}")
                return
            
            # Process top 5 most recent articles
            articles = news_data[:5]
            logger.info(f"Processing top {len(articles)} news items for {symbol}")
            
            # Format articles for analysis
            formatted_articles = [
                {
                    "title": article.get("title", ""),
                    "summary": article.get("summary", ""),
                    "url": article.get("link", ""),
                    "publisher": article.get("publisher", ""),
                    "published": article.get("published", "")
                }
                for article in articles
            ]
            
            logger.info(f"Sending {len(formatted_articles)} articles to Perplexity for analysis")
            analysis = await self.analyzer.analyze_stock_news(symbol, formatted_articles)
            logger.info("Perplexity analysis completed")
            
            # Calculate sentiment score
            sentiment_score = self._calculate_sentiment(analysis)
            logger.info(f"Calculated sentiment score: {sentiment_score}")
            
            # Create news entry
            news_entry = StockNews(
                symbol=symbol,
                date=datetime.now(),
                perplexity_summary=analysis,
                source_urls=[a["url"] for a in formatted_articles],
                sentiment_score=sentiment_score
            )
            
            self.db.add(news_entry)
            self.db.commit()
            logger.info(f"Successfully stored news analysis for {symbol}")
            
            return news_entry
            
        except Exception as e:
            logger.error(f"Error collecting news for {symbol}: {str(e)}", exc_info=True)
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