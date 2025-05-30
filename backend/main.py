import io
import os
import json
import logging
import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import polars as pl
from dotenv import load_dotenv
from database import SessionLocal, FinancialDocument, StockNews, Watchlist, StockAnalysisHistory
from sqlalchemy.orm import Session
import ell
from ell.types import Message
from datetime import date, datetime
import yfinance as yf
from services.news_scraper import PerplexityNewsAnalyzer, NewsCollector
from services.news_scheduler import NewsScheduler
from functools import lru_cache
from contextlib import asynccontextmanager
import asyncio
from typing import Dict
from backend.config import settings
from backend.database import init_db
from backend.auth.google_auth import auth_router
from backend.ml_api import ml_router
from backend.graphql_api import graphql_app
from backend.ai.openai_realtime import RealtimeVoiceAssistant
from backend.websocket_server import connection_manager
from backend.utils.logger import get_logger

class DateJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        return super().default(obj)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

ell.init(store='./logdir', autocommit=True)

@ell.simple(model="gpt-4o-mini")
def analyze_transaction(description: str) -> str:
    """Analyze a single transaction description.
    Return format: MainCategory/SubCategory/Detail
    """
    prompt = f"""Analyze this transaction and categorize it into exactly three levels:
    Transaction: {description}

    Rules:
    1. Use format: MainCategory/SubCategory/Detail
    2. MainCategory should be one of: Shopping, Food, Transport, Housing, Entertainment, Utilities, Healthcare, Income, Other
    3. SubCategory should be specific but standardized
    4. Detail should include merchant type or distinctive detail

    Examples:
    - "WALMART GROCERY" -> Shopping/Groceries/Supermarket
    - "UBER TRIP 123" -> Transport/Rideshare/Uber
    - "NETFLIX MONTHLY" -> Entertainment/Streaming/Netflix
    - "SHELL OIL" -> Transport/Fuel/GasStation

    Return only the category string, no other text.
    """
    return prompt

def enhance_categorization(df: pl.DataFrame) -> pl.DataFrame:
    """Add AI-enhanced categorization to the DataFrame"""
    logger.info("Starting AI categorization")
    try:
        unique_descriptions = df.select("Description").unique().to_series().to_list()
        logger.info(f"Unique descriptions: {unique_descriptions}")
        enhanced_categories = []
        
        for desc in unique_descriptions:
            try:
                category_str = analyze_transaction(desc).strip()
                logger.info(f"AI category for '{desc}': '{category_str}'")
                # Validate format
                parts = category_str.split("/")
                if len(parts) < 3:
                    logger.warning(f"Invalid AI category for '{desc}': '{category_str}'. Using fallback.")
                    category_str = "Other/Unknown/Unknown"
                enhanced_categories.append({
                    "Description": desc,
                    "ai_category": category_str
                })
            except Exception as e:
                logger.warning(f"Failed to categorize '{desc}': {e}. Using fallback.")
                enhanced_categories.append({
                    "Description": desc,
                    "ai_category": "Other/Unknown/Unknown"
                })
        
        categories_df = pl.DataFrame(enhanced_categories)
        df = df.join(categories_df, on="Description", how="left")

        # Split categories into separate columns
        df = df.with_columns([
            pl.col("ai_category").str.split("/").list.get(0).fill_null("Other").alias("main_category"),
            pl.col("ai_category").str.split("/").list.get(1).fill_null("Unknown").alias("sub_category"),
            pl.col("ai_category").str.split("/").list.get(2).fill_null("Unknown").alias("detail_category")
        ])
        
        return df
    except Exception as e:
        logger.error(f"AI categorization failed: {e}")
        # In worst case, just return df without AI columns
        return df.with_columns([
            pl.lit("Other").alias("main_category"),
            pl.lit("Unknown").alias("sub_category"),
            pl.lit("Unknown").alias("detail_category")
        ])

app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router, prefix="/api/auth", tags=["Authentication"])
app.include_router(ml_router, prefix="/api/ml", tags=["Machine Learning"])

# Mount GraphQL app
app.mount("/graphql", graphql_app)

# Serve static files (if needed)
# app.mount("/static", StaticFiles(directory="static"), name="static")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    
    # Initialize database
    await init_db()
    
    # Start background tasks
    # TODO: Start ML model updater, monitoring, etc.
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")
    
    # Cleanup voice sessions
    for session_id, assistant in voice_sessions.items():
        await assistant.stop()
    voice_sessions.clear()

@app.on_event("startup")
async def startup_event():
    try:
        # Initialize news collection system
        perplexity_key = os.getenv("PERPLEXITY_API_KEY")
        if not perplexity_key:
            logger.warning("PERPLEXITY_API_KEY not set, news collection disabled")
            return
            
        analyzer = PerplexityNewsAnalyzer(perplexity_key)
        db = SessionLocal()
        collector = NewsCollector(db, analyzer)
        
        # Start scheduler with default symbols
        scheduler = NewsScheduler(collector, symbols=["AAPL", "MSFT", "GOOGL"])
        scheduler.start()
        
        logger.info("News collection system initialized")
    except Exception as e:
        logger.error(f"Failed to initialize news system: {e}")

def process_financial_data(contents: bytes):
    try:
        df = pl.read_csv(io.BytesIO(contents))
        logger.info(f"Loaded DataFrame with columns: {df.columns}")

        required_cols = {"Date", "Amount", "Description"}
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(f"CSV must contain columns: {', '.join(required_cols)}")

        # Clean and parse data
        df = df.with_columns([
            pl.col("Date").str.strptime(pl.Date, format="%m/%d/%Y"),
            pl.col("Amount").cast(pl.Float64),
            pl.col("Description").cast(pl.Utf8).str.to_uppercase().str.strip_chars()
        ])

        # Remove basic categorization and rely only on AI
        df = enhance_categorization(df)

        date_series = df["Date"]
        max_date = date_series.max()
        min_date = date_series.min()
        num_days = max((max_date - min_date).days + 1, 1)
        total_expenses = float(df["Amount"].sum())

        # Daily expenses
        daily = (
            df.group_by("Date")
            .agg(pl.col("Amount").sum().alias("amount"))
            .sort("Date")
            .to_dicts()
        )
        for d in daily:
            d["date"] = d.pop("Date").strftime("%Y-%m-%d")

        # Top 5 expenses
        top_5 = (
            df.group_by("Description")
            .agg(pl.col("Amount").sum().alias("amount"))
            .sort("amount", descending=True)
            .head(5)
            .to_dicts()
        )
        for t in top_5:
            t["description"] = t.pop("Description")

        # Recurring transactions
        recurring = (
            df.group_by("Description")
            .agg([
                pl.col("Amount").count().alias("count"),
                pl.col("Amount").sum().alias("totalamount"),
                (pl.col("Amount").sum() / pl.col("Amount").count()).alias("averageamount")
            ])
            .filter(pl.col("count") > 1)
            .sort("totalamount", descending=True)
            .to_dicts()
        )
        for r in recurring:
            r["description"] = r.pop("Description")

        # Cumulative spending
        daily_df = pl.DataFrame(daily)
        daily_df = daily_df.with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))
        daily_df = daily_df.sort("date")
        daily_df = daily_df.with_columns(pl.col("amount").cum_sum().alias("cumulative"))
        cumulative_spending = []
        for row in daily_df.to_dicts():
            row["date"] = row["date"].strftime("%Y-%m-%d")
            cumulative_spending.append(row)

        # AI categories
        ai_categories = []
        if "main_category" in df.columns and "sub_category" in df.columns and "detail_category" in df.columns:
            ai_categories = (
                df.group_by(["main_category", "sub_category", "detail_category"])
                .agg([
                    pl.col("Amount").sum().alias("amount"),
                    pl.col("Amount").count().alias("count")
                ])
                .sort("amount", descending=True)
                .to_dicts()
            )

        logger.info(f"AI categories:\n{json.dumps(ai_categories, indent=2)}")

        summary = {
            "total_expenses": total_expenses,
            "average_daily": float(total_expenses / num_days),
            "average_monthly": float(total_expenses / (num_days / 30)),
            "daily_expenses": daily,
            "top_5_expenses": top_5,
            "recurring_transactions": recurring,
            "cumulative_spending": cumulative_spending,
            "ai_categories": ai_categories,
            "category_hierarchy": {
                "main": df["main_category"].unique().to_list() if "main_category" in df.columns else [],
                "sub": df["sub_category"].unique().to_list() if "sub_category" in df.columns else [],
                "detail": df["detail_category"].unique().to_list() if "detail_category" in df.columns else []
            }
        }

        return summary

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    logger.info(f"Received file upload: {file.filename}")

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    base_name, ext = os.path.splitext(file.filename)
    new_filename = file.filename
    doc_count = db.query(FinancialDocument).filter(FinancialDocument.filename == file.filename).count()
    if doc_count > 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"{base_name}_{timestamp}{ext}"

    summary = process_financial_data(contents)

    doc = FinancialDocument(
        filename=new_filename,
        summary_json=json.dumps(summary, cls=DateJSONEncoder)
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    return {
        "status": "success",
        "file_id": doc.id,
        "summary": summary,
        "filename": new_filename
    }

@app.get("/files")
def list_files(db: Session = Depends(get_db)):
    docs = db.query(FinancialDocument).order_by(FinancialDocument.upload_date.desc()).all()
    return [
        {
            "id": doc.id,
            "filename": doc.filename,
            "upload_date": doc.upload_date.isoformat() if doc.upload_date else None
        }
        for doc in docs
    ]

@app.get("/file/{file_id}")
def get_file(file_id: int, db: Session = Depends(get_db)):
    doc = db.query(FinancialDocument).filter(FinancialDocument.id == file_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="File not found")

    summary = json.loads(doc.summary_json)
    return {
        "status": "success",
        "summary": summary
    }

@lru_cache(maxsize=100)
def get_cached_price(symbol: str, timestamp: int) -> float:
    """Cache price data for 1 minute"""
    ticker = yf.Ticker(symbol)
    data = ticker.history(period="1d")
    return float(data['Close'].iloc[-1])

@app.get("/stock_price")
async def get_stock_price(symbol: str):
    """Fetch current stock price with caching"""
    try:
        # Use 1-minute timestamp for cache key
        timestamp = int(datetime.now().timestamp() / 60)
        price = get_cached_price(symbol, timestamp)
        return {"symbol": symbol, "price": price}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_stock_data")
async def upload_stock_file(file: UploadFile = File(...)):
    """Process uploaded stock CSV data."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        # Parse CSV with polars for better performance
        df = pl.read_csv(io.BytesIO(contents))
        
        required_cols = {"Date", "Price"}
        if not required_cols.issubset(set(df.columns)):
            raise HTTPException(
                status_code=400, 
                detail=f"CSV must contain columns: {', '.join(required_cols)}"
            )

        # Process data
        df = df.with_columns([
            pl.col("Date").str.strptime(pl.Date, format="%Y-%m-%d"),
            pl.col("Price").cast(pl.Float64)
        ])

        # Generate summary
        summary = {
            "daily_prices": df.sort("Date").select([
                "Date", "Price"
            ]).rename({
                "Date": "date",
                "Price": "price"
            }).with_columns([
                pl.col("date").cast(str)
            ]).to_dicts()
        }

        # Add volume analysis if available
        if "Volume" in df.columns:
            top_volume = df.sort("Volume", descending=True).head(5)
            summary["top_volume_days"] = top_volume.select([
                "Date", "Volume"
            ]).rename({
                "Date": "date",
                "Volume": "volume"
            }).with_columns([
                pl.col("date").cast(str)
            ]).to_dicts()

        return {"status": "success", "summary": summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stock/{symbol}/news")
async def get_stock_news(
    symbol: str, 
    start_date: datetime = None, 
    end_date: datetime = None,
    db: Session = Depends(get_db)
):
    """Get historical news and analysis for a stock"""
    query = db.query(StockNews).filter(StockNews.symbol == symbol)
    
    if start_date:
        query = query.filter(StockNews.date >= start_date)
    if end_date:
        query = query.filter(StockNews.date <= end_date)
        
    news = query.order_by(StockNews.date.desc()).all()
    
    return {
        "symbol": symbol,
        "news": [
            {
                "date": n.date,
                "summary": n.perplexity_summary,
                "sentiment": n.sentiment_score,
                "sources": n.source_urls
            }
            for n in news
        ]
    }

@app.post("/stock/{symbol}/collect_news")
async def trigger_news_collection(
    symbol: str,
    db: Session = Depends(get_db)
):
    """Manually trigger news collection for a symbol"""
    try:
        logger.info(f"Manual news collection triggered for {symbol}")
        
        analyzer = PerplexityNewsAnalyzer(os.getenv("PERPLEXITY_API_KEY"))
        collector = NewsCollector(db, analyzer)
        
        news_entry = await collector.collect_daily_news(symbol)
        
        if news_entry:
            logger.info(f"Successfully collected news for {symbol}")
            return {"status": "success", "message": "News collected successfully"}
        else:
            logger.warning(f"No news collected for {symbol}")
            return {"status": "warning", "message": "No news found"}
            
    except Exception as e:
        logger.error(f"Error collecting news for {symbol}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/watchlist")
async def get_watchlist(db: Session = Depends(get_db)):
    """Get user's watchlist"""
    try:
        watchlist = db.query(Watchlist).all()
        return {
            "watchlist": [
                {
                    "symbol": item.symbol,
                    "added_date": item.added_date,
                    "last_analysis": item.last_analysis
                }
                for item in watchlist
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching watchlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/watchlist/{symbol}")
async def add_to_watchlist(symbol: str, db: Session = Depends(get_db)):
    existing = db.query(Watchlist).filter(Watchlist.symbol == symbol).first()
    if existing:
        return {"status": "exists", "message": f"{symbol} already in watchlist"}
    
    watchlist_item = Watchlist(symbol=symbol)
    db.add(watchlist_item)
    db.commit()
    return {"status": "success", "message": f"Added {symbol} to watchlist"}

@app.delete("/watchlist/{symbol}")
async def remove_from_watchlist(symbol: str, db: Session = Depends(get_db)):
    db.query(Watchlist).filter(Watchlist.symbol == symbol).delete()
    db.commit()
    return {"status": "success", "message": f"Removed {symbol} from watchlist"}

@app.post("/watchlist/analyze")
async def analyze_watchlist(db: Session = Depends(get_db)):
    """Analyze all stocks in watchlist"""
    try:
        watchlist = db.query(Watchlist).all()
        if not watchlist:
            return {"status": "warning", "message": "Watchlist is empty"}
            
        analyzer = PerplexityNewsAnalyzer(os.getenv("PERPLEXITY_API_KEY"))
        collector = NewsCollector(db, analyzer)
        
        results = []
        for item in watchlist:
            try:
                logger.info(f"Analyzing {item.symbol}")
                news_entry = await collector.collect_daily_news(item.symbol)
                
                # Update last_analysis timestamp
                item.last_analysis = datetime.utcnow()
                db.commit()
                
                if news_entry:
                    results.append({
                        "symbol": item.symbol,
                        "status": "success"
                    })
            except Exception as e:
                logger.error(f"Error analyzing {item.symbol}: {e}")
                results.append({
                    "symbol": item.symbol,
                    "status": "error",
                    "error": str(e)
                })
                
        return {
            "status": "complete",
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stock/{symbol}/analysis_history")
async def get_analysis_history(
    symbol: str,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get historical analyses for a stock"""
    history = db.query(StockAnalysisHistory)\
        .filter(StockAnalysisHistory.symbol == symbol)\
        .order_by(StockAnalysisHistory.analysis_date.desc())\
        .limit(limit)\
        .all()
        
    return {
        "symbol": symbol,
        "history": [
            {
                "date": h.analysis_date,
                "summary": h.perplexity_summary,
                "sentiment": h.sentiment_score,
                "sources": h.source_urls
            }
            for h in history
        ]
    }

# Add Yahoo Finance configuration
YAHOO_CREDENTIALS = {
    "app_id": os.getenv("YAHOO_FINANCE_APP_ID"),
    "client_id": os.getenv("YAHOO_FINANCE_CLIENT_ID"),
    "client_secret": os.getenv("YAHOO_FINANCE_CLIENT_SECRET")
}

# Active voice assistant sessions
voice_sessions: Dict[str, RealtimeVoiceAssistant] = {}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "features": {
            "voice_assistant": settings.ENABLE_VOICE_ASSISTANT,
            "blockchain": settings.ENABLE_BLOCKCHAIN,
            "ml_predictions": settings.ENABLE_ML_PREDICTIONS,
            "real_time": settings.ENABLE_REAL_TIME
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": asyncio.get_event_loop().time()
    }

@app.websocket("/ws/voice-assistant")
async def voice_assistant_websocket(websocket: WebSocket):
    """WebSocket endpoint for voice assistant"""
    await websocket.accept()
    session_id = None
    
    try:
        # Get initial configuration
        config_data = await websocket.receive_json()
        user_id = config_data.get("userId")
        session_id = f"{user_id}_{asyncio.get_event_loop().time()}"
        
        # Create voice assistant instance
        assistant = RealtimeVoiceAssistant(user_id=user_id)
        voice_sessions[session_id] = assistant
        
        # Start the assistant
        await assistant.start()
        
        # Send ready signal
        await websocket.send_json({
            "type": "ready",
            "sessionId": session_id
        })
        
        # Handle messages
        while True:
            # Receive data (can be JSON or binary audio)
            try:
                # Try to receive JSON first
                data = await websocket.receive_json()
                
                if data["type"] == "configure":
                    # Handle configuration updates
                    pass
                elif data["type"] == "start_listening":
                    # Start listening
                    pass
                elif data["type"] == "stop_listening":
                    # Stop listening
                    pass
                    
            except:
                # If not JSON, assume it's audio data
                audio_data = await websocket.receive_bytes()
                # Process audio data
                # This would be forwarded to OpenAI Realtime API
                pass
                
    except WebSocketDisconnect:
        logger.info(f"Voice assistant session {session_id} disconnected")
    except Exception as e:
        logger.error(f"Voice assistant error: {e}")
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
    finally:
        # Cleanup
        if session_id and session_id in voice_sessions:
            await voice_sessions[session_id].stop()
            del voice_sessions[session_id]

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """General WebSocket endpoint for real-time updates"""
    await connection_manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive and broadcast messages
            data = await websocket.receive_json()
            
            # Handle different message types
            if data.get("type") == "subscribe":
                # Subscribe to specific channels
                channels = data.get("channels", [])
                await connection_manager.subscribe(client_id, channels)
                
            elif data.get("type") == "unsubscribe":
                # Unsubscribe from channels
                channels = data.get("channels", [])
                await connection_manager.unsubscribe(client_id, channels)
                
            else:
                # Broadcast message to relevant clients
                await connection_manager.broadcast(data, data.get("channel"))
                
    except WebSocketDisconnect:
        await connection_manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await connection_manager.disconnect(client_id)

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Not found", "status_code": 404}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return {"error": "Internal server error", "status_code": 500}

if __name__ == "__main__":
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)