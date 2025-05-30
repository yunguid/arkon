import io
import os
import json
import logging
import datetime
import csv
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import polars as pl
from dotenv import load_dotenv
from database import SessionLocal, init_db
from models import (
    FinancialDocument, StockNews, Watchlist, StockAnalysisHistory,
    Budget, CategoryCache, AlertLog
)
from sqlalchemy.orm import Session
import ell
from ell.types import Message
from datetime import date, datetime
import yfinance as yf
from services.news_scraper import PerplexityNewsAnalyzer, NewsCollector
from services.news_scheduler import NewsScheduler
from functools import lru_cache
from utils import (
    rate_limiter, category_cache, cache_result, validate_csv_data,
    check_budget_alerts, prepare_export_data, task_queue, paginate_query
)
from typing import Optional

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('financial_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

# Validate environment variables
REQUIRED_ENV_VARS = ["ANTHROPIC_API_KEY"]
for var in REQUIRED_ENV_VARS:
    if not os.getenv(var):
        logger.error(f"Missing required environment variable: {var}")
        raise ValueError(f"{var} environment variable is not set")

# Initialize ell
ell.init(store='./logdir', autocommit=True)

# Custom exception handlers
class AppException(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    task_queue.start_workers(num_workers=3)
    
    try:
        # Initialize news collection system
        perplexity_key = os.getenv("PERPLEXITY_API_KEY")
        if perplexity_key:
            analyzer = PerplexityNewsAnalyzer(perplexity_key)
            db = SessionLocal()
            collector = NewsCollector(db, analyzer)
            scheduler = NewsScheduler(collector, symbols=["AAPL", "MSFT", "GOOGL"])
            scheduler.start()
            logger.info("News collection system initialized")
    except Exception as e:
        logger.error(f"Failed to initialize news system: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")

app = FastAPI(
    title="Arkon Financial Analyzer API",
    description="Advanced financial analysis and stock tracking system",
    version="2.0.0",
    lifespan=lifespan
)

# Enhanced CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    if not rate_limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please try again later."}
        )
    response = await call_next(request)
    return response

# Dependency for database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Enhanced transaction analysis with caching
@ell.simple(model="gpt-4o-mini", max_tokens=100)
def analyze_transaction_enhanced(description: str) -> str:
    """Analyze transaction with improved categorization"""
    prompt = f"""Analyze this transaction and categorize it into exactly three levels:
    Transaction: {description}

    Rules:
    1. Use format: MainCategory/SubCategory/Detail
    2. MainCategory must be one of: Shopping, Food, Transport, Housing, Entertainment, Utilities, Healthcare, Income, Investment, Other
    3. SubCategory should be specific but standardized
    4. Detail should include merchant type or distinctive detail
    
    Consider common variations and abbreviations.
    
    Return only the category string, no other text.
    """
    return prompt

def enhance_categorization_with_cache(df: pl.DataFrame, db: Session) -> pl.DataFrame:
    """Enhanced categorization with database caching"""
    logger.info("Starting enhanced AI categorization")
    
    try:
        unique_descriptions = df.select("Description").unique().to_series().to_list()
        enhanced_categories = []
        
        for desc in unique_descriptions:
            # Check cache first
            cached = db.query(CategoryCache).filter(
                CategoryCache.description == desc
            ).first()
            
            if cached:
                # Update usage stats
                cached.last_used = datetime.utcnow()
                cached.usage_count += 1
                db.commit()
                
                enhanced_categories.append({
                    "Description": desc,
                    "main_category": cached.main_category,
                    "sub_category": cached.sub_category,
                    "detail_category": cached.detail_category
                })
            else:
                # Get AI categorization
                try:
                    category_str = analyze_transaction_enhanced(desc).strip()
                    parts = category_str.split("/")
                    
                    if len(parts) >= 3:
                        main_cat, sub_cat, detail_cat = parts[0], parts[1], parts[2]
                    else:
                        main_cat, sub_cat, detail_cat = "Other", "Unknown", "Unknown"
                    
                    # Save to cache
                    cache_entry = CategoryCache(
                        description=desc,
                        main_category=main_cat,
                        sub_category=sub_cat,
                        detail_category=detail_cat
                    )
                    db.add(cache_entry)
                    db.commit()
                    
                    enhanced_categories.append({
                        "Description": desc,
                        "main_category": main_cat,
                        "sub_category": sub_cat,
                        "detail_category": detail_cat
                    })
                except Exception as e:
                    logger.warning(f"Failed to categorize '{desc}': {e}")
                    enhanced_categories.append({
                        "Description": desc,
                        "main_category": "Other",
                        "sub_category": "Unknown",
                        "detail_category": "Unknown"
                    })
        
        # Join categories back to dataframe
        categories_df = pl.DataFrame(enhanced_categories)
        df = df.join(categories_df, on="Description", how="left")
        
        return df
        
    except Exception as e:
        logger.error(f"Enhanced categorization failed: {e}")
        return df.with_columns([
            pl.lit("Other").alias("main_category"),
            pl.lit("Unknown").alias("sub_category"),
            pl.lit("Unknown").alias("detail_category")
        ])

def process_financial_data_enhanced(contents: bytes, db: Session, user_id: str = "default"):
    """Enhanced financial data processing with validation and budget alerts"""
    try:
        df = pl.read_csv(io.BytesIO(contents))
        
        # Validate data
        is_valid, error_msg = validate_csv_data(df)
        if not is_valid:
            raise ValueError(error_msg)
        
        # Process data
        df = df.with_columns([
            pl.col("Date").str.strptime(pl.Date, format="%m/%d/%Y"),
            pl.col("Amount").cast(pl.Float64),
            pl.col("Description").cast(pl.Utf8).str.to_uppercase().str.strip_chars()
        ])
        
        # Enhanced categorization with caching
        df = enhance_categorization_with_cache(df, db)
        
        # Calculate summary statistics
        date_series = df["Date"]
        max_date = date_series.max()
        min_date = date_series.min()
        num_days = max((max_date - min_date).days + 1, 1)
        total_expenses = float(df["Amount"].sum())
        
        # Category totals for budget checking
        category_totals = {}
        if "main_category" in df.columns:
            cat_summary = df.group_by("main_category").agg(
                pl.col("Amount").sum().alias("total")
            ).to_dicts()
            category_totals = {row["main_category"]: row["total"] for row in cat_summary}
        
        # Check budget alerts
        budget_alerts = check_budget_alerts(db, user_id, category_totals)
        
        # Prepare all summaries
        summary = {
            "total_expenses": total_expenses,
            "average_daily": float(total_expenses / num_days),
            "average_monthly": float(total_expenses / (num_days / 30)),
            "date_range": {
                "start": min_date.strftime("%Y-%m-%d"),
                "end": max_date.strftime("%Y-%m-%d"),
                "days": num_days
            },
            "daily_expenses": df.group_by("Date").agg(
                pl.col("Amount").sum().alias("amount")
            ).sort("Date").with_columns(
                pl.col("Date").cast(str).alias("date")
            ).drop("Date").to_dicts(),
            "top_5_expenses": df.group_by("Description").agg(
                pl.col("Amount").sum().alias("amount")
            ).sort("amount", descending=True).head(5).rename({
                "Description": "description"
            }).to_dicts(),
            "recurring_transactions": df.group_by("Description").agg([
                pl.col("Amount").count().alias("count"),
                pl.col("Amount").sum().alias("totalamount"),
                pl.col("Amount").mean().alias("averageamount")
            ]).filter(pl.col("count") > 1).sort("totalamount", descending=True).rename({
                "Description": "description"
            }).to_dicts(),
            "ai_categories": df.group_by(["main_category", "sub_category", "detail_category"]).agg([
                pl.col("Amount").sum().alias("amount"),
                pl.col("Amount").count().alias("count")
            ]).sort("amount", descending=True).to_dicts() if "main_category" in df.columns else [],
            "budget_alerts": budget_alerts,
            "statistics": {
                "median_transaction": float(df["Amount"].median()),
                "std_deviation": float(df["Amount"].std()),
                "min_transaction": float(df["Amount"].min()),
                "max_transaction": float(df["Amount"].max())
            }
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# API Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.post("/upload")
async def upload_file_enhanced(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user_id: str = Query(default="default")
):
    """Enhanced file upload with better error handling"""
    logger.info(f"Received file upload: {file.filename} from user: {user_id}")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    
    # Check file size limit (10MB)
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds 10MB limit")
    
    # Process with enhanced function
    summary = process_financial_data_enhanced(contents, db, user_id)
    
    # Save to database
    doc = FinancialDocument(
        filename=file.filename,
        summary_json=json.dumps(summary),
        user_id=user_id,
        processing_status="completed"
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    
    return {
        "status": "success",
        "file_id": doc.id,
        "summary": summary,
        "filename": file.filename,
        "has_budget_alerts": len(summary.get("budget_alerts", [])) > 0
    }

@app.get("/files")
def list_files_paginated(
    db: Session = Depends(get_db),
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=20, ge=1, le=100),
    user_id: str = Query(default="default")
):
    """List files with pagination"""
    query = db.query(FinancialDocument).filter(
        FinancialDocument.user_id == user_id
    ).order_by(FinancialDocument.upload_date.desc())
    
    result = paginate_query(query, page, per_page)
    
    return {
        "files": [
            {
                "id": doc.id,
                "filename": doc.filename,
                "upload_date": doc.upload_date.isoformat() if doc.upload_date else None,
                "status": doc.processing_status
            }
            for doc in result["items"]
        ],
        "pagination": {
            "page": result["page"],
            "per_page": result["per_page"],
            "total": result["total"],
            "pages": result["pages"]
        }
    }

@app.get("/export/{file_id}")
async def export_file_data(
    file_id: int,
    format: str = Query(default="csv", regex="^(csv|json)$"),
    db: Session = Depends(get_db)
):
    """Export financial data in various formats"""
    doc = db.query(FinancialDocument).filter(FinancialDocument.id == file_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="File not found")
    
    summary = json.loads(doc.summary_json)
    export_data = prepare_export_data(summary, format)
    
    if format == "csv":
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=["date", "amount", "type"])
        writer.writeheader()
        writer.writerows(export_data["transactions"])
        
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=export_{file_id}.csv"}
        )
    else:
        return StreamingResponse(
            io.BytesIO(json.dumps(export_data, indent=2).encode()),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=export_{file_id}.json"}
        )

# Budget Management Endpoints

@app.post("/budgets")
async def create_budget(
    category: str,
    monthly_limit: float,
    subcategory: Optional[str] = None,
    db: Session = Depends(get_db),
    user_id: str = Query(default="default")
):
    """Create a new budget"""
    # Check if budget already exists
    existing = db.query(Budget).filter(
        Budget.user_id == user_id,
        Budget.category == category,
        Budget.is_active == True
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="Budget already exists for this category")
    
    budget = Budget(
        user_id=user_id,
        category=category,
        subcategory=subcategory,
        monthly_limit=monthly_limit
    )
    db.add(budget)
    db.commit()
    db.refresh(budget)
    
    return {
        "status": "success",
        "budget": {
            "id": budget.id,
            "category": budget.category,
            "monthly_limit": budget.monthly_limit
        }
    }

@app.get("/budgets")
def get_budgets(
    db: Session = Depends(get_db),
    user_id: str = Query(default="default")
):
    """Get all active budgets"""
    budgets = db.query(Budget).filter(
        Budget.user_id == user_id,
        Budget.is_active == True
    ).all()
    
    return {
        "budgets": [
            {
                "id": b.id,
                "category": b.category,
                "subcategory": b.subcategory,
                "monthly_limit": b.monthly_limit,
                "created_at": b.created_at.isoformat()
            }
            for b in budgets
        ]
    }

@app.put("/budgets/{budget_id}")
async def update_budget(
    budget_id: int,
    monthly_limit: float,
    db: Session = Depends(get_db)
):
    """Update budget limit"""
    budget = db.query(Budget).filter(Budget.id == budget_id).first()
    if not budget:
        raise HTTPException(status_code=404, detail="Budget not found")
    
    budget.monthly_limit = monthly_limit
    budget.updated_at = datetime.utcnow()
    db.commit()
    
    return {"status": "success", "message": "Budget updated"}

@app.delete("/budgets/{budget_id}")
async def delete_budget(
    budget_id: int,
    db: Session = Depends(get_db)
):
    """Delete (deactivate) a budget"""
    budget = db.query(Budget).filter(Budget.id == budget_id).first()
    if not budget:
        raise HTTPException(status_code=404, detail="Budget not found")
    
    budget.is_active = False
    db.commit()
    
    return {"status": "success", "message": "Budget deleted"}

# Alert Management

@app.get("/alerts")
def get_alerts(
    db: Session = Depends(get_db),
    user_id: str = Query(default="default"),
    unread_only: bool = Query(default=False)
):
    """Get user alerts"""
    query = db.query(AlertLog).filter(AlertLog.user_id == user_id)
    
    if unread_only:
        query = query.filter(AlertLog.is_read == False)
    
    alerts = query.order_by(AlertLog.created_at.desc()).limit(50).all()
    
    return {
        "alerts": [
            {
                "id": a.id,
                "type": a.alert_type,
                "message": a.message,
                "metadata": a.metadata,
                "is_read": a.is_read,
                "created_at": a.created_at.isoformat()
            }
            for a in alerts
        ]
    }

@app.put("/alerts/{alert_id}/read")
async def mark_alert_read(
    alert_id: int,
    db: Session = Depends(get_db)
):
    """Mark alert as read"""
    alert = db.query(AlertLog).filter(AlertLog.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    alert.is_read = True
    db.commit()
    
    return {"status": "success"}

# Enhanced Stock Features

@app.get("/watchlist")
async def get_enhanced_watchlist(
    db: Session = Depends(get_db),
    user_id: str = Query(default="default")
):
    """Get enhanced watchlist with current prices"""
    watchlist = db.query(Watchlist).filter(
        Watchlist.user_id == user_id
    ).all()
    
    items = []
    for item in watchlist:
        try:
            # Get current price
            ticker = yf.Ticker(item.symbol)
            current_price = float(ticker.info.get('currentPrice', 0))
            
            # Check price alerts
            alert_triggered = False
            if item.price_alert_above and current_price >= item.price_alert_above:
                alert_triggered = True
            elif item.price_alert_below and current_price <= item.price_alert_below:
                alert_triggered = True
            
            items.append({
                "symbol": item.symbol,
                "added_date": item.added_date.isoformat(),
                "last_analysis": item.last_analysis.isoformat() if item.last_analysis else None,
                "current_price": current_price,
                "price_alert_above": item.price_alert_above,
                "price_alert_below": item.price_alert_below,
                "alert_triggered": alert_triggered,
                "notes": item.notes
            })
        except Exception as e:
            logger.error(f"Error fetching price for {item.symbol}: {e}")
            items.append({
                "symbol": item.symbol,
                "added_date": item.added_date.isoformat(),
                "error": "Failed to fetch current price"
            })
    
    return {"watchlist": items}

@app.post("/watchlist/{symbol}")
async def add_to_enhanced_watchlist(
    symbol: str,
    price_alert_above: Optional[float] = None,
    price_alert_below: Optional[float] = None,
    notes: Optional[str] = None,
    db: Session = Depends(get_db),
    user_id: str = Query(default="default")
):
    """Add symbol to watchlist with optional price alerts"""
    existing = db.query(Watchlist).filter(
        Watchlist.user_id == user_id,
        Watchlist.symbol == symbol
    ).first()
    
    if existing:
        # Update existing entry
        existing.price_alert_above = price_alert_above
        existing.price_alert_below = price_alert_below
        existing.notes = notes
        db.commit()
        return {"status": "updated", "message": f"Updated {symbol} in watchlist"}
    
    watchlist_item = Watchlist(
        user_id=user_id,
        symbol=symbol,
        price_alert_above=price_alert_above,
        price_alert_below=price_alert_below,
        notes=notes
    )
    db.add(watchlist_item)
    db.commit()
    
    return {"status": "success", "message": f"Added {symbol} to watchlist"}

# Statistics endpoint
@app.get("/statistics")
async def get_statistics(
    db: Session = Depends(get_db),
    user_id: str = Query(default="default")
):
    """Get overall statistics"""
    total_files = db.query(FinancialDocument).filter(
        FinancialDocument.user_id == user_id
    ).count()
    
    active_budgets = db.query(Budget).filter(
        Budget.user_id == user_id,
        Budget.is_active == True
    ).count()
    
    watchlist_count = db.query(Watchlist).filter(
        Watchlist.user_id == user_id
    ).count()
    
    unread_alerts = db.query(AlertLog).filter(
        AlertLog.user_id == user_id,
        AlertLog.is_read == False
    ).count()
    
    # Get cached categories count
    cached_categories = db.query(CategoryCache).count()
    
    return {
        "total_files": total_files,
        "active_budgets": active_budgets,
        "watchlist_count": watchlist_count,
        "unread_alerts": unread_alerts,
        "cached_categories": cached_categories
    }

if __name__ == "__main__":
    logger.info("Starting enhanced FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000) 