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
        json_str = json_str.replace('\\', '\\\\')  # Escape backslashes
        
        # Fix trailing commas (common AI error)
        json_str = json_str.replace(',}', '}').replace(',]', ']')
        
        return json_str

    async def analyze_stock_news(self, symbol: str, articles: List[Dict]) -> Dict:
        try:
            logger.info(f"Starting Perplexity analysis for {symbol}")
            
            # Update prompt to be more strict about JSON format
            prompt = f"""
            Analyze these news articles about {symbol} and return a JSON object.
            
            Rules:
            1. Response must be ONLY valid JSON
            2. No trailing commas
            3. All strings must be properly escaped
            4. No comments or additional text
            
            Required format:
            {{
                "key_developments": "Brief summary",
                "market_sentiment": "positive/negative/neutral",
                "price_impact": "Brief analysis",
                "risks": ["Risk 1", "Risk 2"],
                "expert_quotes": ["Quote 1", "Quote 2"],
                "summary": "Overall summary"
            }}
            
            Articles: {json.dumps(articles)}
            """
            
            response = self.client.chat.completions.create(
                model="llama-3.1-sonar-huge-128k-online",
                messages=[
                    {"role": "system", "content": "You are a financial analyst. Return only valid JSON."},
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
        self._news_cache[symbol] = (now, news)
        return news

    async def collect_daily_news(self, symbol: str) -> Optional[StockNews]:
        try:
            logger.info(f"Starting news collection for {symbol}")
            news_data = await self.get_news(symbol)
            
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
            
            # Store in history
            history_entry = StockAnalysisHistory(
                symbol=symbol,
                perplexity_summary=analysis,
                sentiment_score=sentiment_score,
                source_urls=[a["url"] for a in formatted_articles]
            )
            self.db.add(history_entry)
            self.db.commit()
            
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