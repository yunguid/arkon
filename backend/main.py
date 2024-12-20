import io
import os
import json
import logging
import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import polars as pl
from dotenv import load_dotenv
from database import SessionLocal, FinancialDocument, StockNews, WatchlistStock, StockAnalysisHistory, init_db, StockNewsRepository, FinancialMetrics, StockFiling
from sqlalchemy.orm import Session
import ell
from ell.types import Message
from datetime import date, datetime
import yfinance as yf
from services.news_scraper import PerplexityNewsAnalyzer, NewsCollector
from services.news_scheduler import NewsScheduler
from functools import lru_cache
from contextlib import asynccontextmanager
from sec_api import QueryApi, PdfGeneratorApi
from sec_cik_mapper import StockMapper  # Add this import at the top

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

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('financial_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize database
init_db()

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

app = FastAPI()

# Initialize components
perplexity_key = os.getenv("PERPLEXITY_API_KEY")
analyzer = PerplexityNewsAnalyzer(perplexity_key) if perplexity_key else None
db = SessionLocal()
collector = NewsCollector(db, analyzer) if analyzer else None
scheduler = NewsScheduler(collector, symbols=["AAPL", "MSFT", "GOOGL"]) if collector else None

# Add SEC API setup
SEC_API_KEY = os.getenv('SEC_API_KEY')
if not SEC_API_KEY:
    logger.warning("SEC API key not set. SEC filing features will be disabled.")
    queryApi = None
    pdfApi = None
else:
    try:
        queryApi = QueryApi(api_key=SEC_API_KEY)
        pdfApi = PdfGeneratorApi(api_key=SEC_API_KEY)
        logger.info("SEC API initialization successful")
    except Exception as e:
        logger.error(f"SEC API initialization failed: {str(e)}")
        queryApi = None
        pdfApi = None

# Initialize the stock mapper once
stock_mapper = StockMapper()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing database...")
    init_db()
    if scheduler:
        logger.info("News collection system initializing...")
        scheduler.start()
    yield
    # Shutdown
    if scheduler:
        scheduler.shutdown()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
async def trigger_news_collection(symbol: str, db: Session = Depends(get_db)):
    try:
        logger.info(f"Manual news collection triggered for {symbol}")
        
        analyzer = PerplexityNewsAnalyzer(os.getenv("PERPLEXITY_API_KEY"))
        news_repo = StockNewsRepository(db)
        collector = NewsCollector(news_repo, analyzer)
        
        analysis = await collector.collect_daily_news(symbol)
        
        if analysis:
            logger.info(f"Successfully collected news for {symbol}")
            return {
                "status": "success", 
                "news": {
                    "summary": analysis,  # This contains all the fields
                    "source_urls": analysis.get('source_urls', [])
                }
            }
    except Exception as e:
        logger.error(f"Error collecting news for {symbol}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/watchlist")
async def get_watchlist(db: Session = Depends(get_db)):
    """Get user's watchlist with enhanced data"""
    try:
        stocks = db.query(WatchlistStock).all()
        return {
            "watchlist": [
                {
                    "symbol": stock.symbol,
                    "company_name": stock.company_name,
                    "sector": stock.sector,
                    "added_date": stock.added_date,
                    "last_analysis": stock.last_analyzed,
                    "has_filings": len(stock.filings) > 0,
                    "metrics_available": bool(db.query(FinancialMetrics)
                        .filter(FinancialMetrics.stock_id == stock.id)
                        .first())
                }
                for stock in stocks
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching watchlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/watchlist/{symbol}")
async def add_to_watchlist(symbol: str, db: Session = Depends(get_db)):
    """Add stock to watchlist with enhanced company info"""
    try:
        # Check if already exists
        existing = db.query(WatchlistStock).filter(WatchlistStock.symbol == symbol).first()
        if existing:
            return {"status": "exists", "message": f"{symbol} already in watchlist"}
        
        # Get CIK from SEC mapper
        cik = stock_mapper.ticker_to_cik.get(symbol.upper())
        if not cik:
            logger.warning(f"No CIK found for {symbol}")
        
        # Fetch basic company info from Yahoo Finance
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Create new watchlist entry with enhanced data
        stock = WatchlistStock(
            symbol=symbol,
            company_name=info.get('longName', '') or stock_mapper.ticker_to_company_name.get(symbol.upper(), ''),
            sector=info.get('sector', ''),
            cik=cik
        )
        
        db.add(stock)
        db.commit()
        db.refresh(stock)
        
        return {
            "status": "success",
            "message": f"Added {symbol} to watchlist",
            "data": {
                "symbol": stock.symbol,
                "company_name": stock.company_name,
                "sector": stock.sector,
                "cik": cik
            }
        }
        
    except Exception as e:
        logger.error(f"Error adding to watchlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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

@app.get("/stock/{symbol}/price")
async def get_stock_price(symbol: str):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d")
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
            
        if 'Close' not in data.columns:
            raise HTTPException(status_code=500, detail="Invalid data format from yfinance")
            
        current_price = float(data['Close'].iloc[-1])
        return {
            "symbol": symbol,
            "price": current_price,
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching price for {symbol}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to fetch price for {symbol}: {str(e)}"
        )

@app.get("/stock/{symbol}/research")
async def get_stock_research(symbol: str, db: Session = Depends(get_db)):
    """Get comprehensive research data for a stock"""
    try:
        # Get stock from watchlist
        stock = db.query(WatchlistStock).filter(WatchlistStock.symbol == symbol).first()
        if not stock:
            raise HTTPException(status_code=404, detail="Stock not found in watchlist")
        
        # Get latest financial metrics
        metrics = db.query(FinancialMetrics)\
            .filter(FinancialMetrics.stock_id == stock.id)\
            .order_by(FinancialMetrics.date.desc())\
            .first()
        
        # Get SEC filings
        filings = db.query(StockFiling)\
            .filter(StockFiling.stock_id == stock.id)\
            .order_by(StockFiling.filing_date.desc())\
            .all()
        
        # Format response
        return {
            "symbol": stock.symbol,
            "company_name": stock.company_name,
            "sector": stock.sector,
            "metrics": metrics.metrics if metrics else None,
            "filings": [
                {
                    "type": f.filing_type,
                    "date": f.filing_date,
                    "url": f.filing_url,
                    "data": f.filing_data
                }
                for f in filings
            ] if filings else []
        }
        
    except Exception as e:
        logger.error(f"Error fetching research data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stock/{symbol}/fetch-latest-10k")
async def fetch_latest_10k(symbol: str, db: Session = Depends(get_db)):
    """Fetch latest 10-K filing for a stock"""
    if not queryApi:
        raise HTTPException(
            status_code=503,
            detail="SEC API not configured"
        )
    
    try:
        logger.info(f"Fetching latest 10-K for {symbol}")
        stock = db.query(WatchlistStock).filter(WatchlistStock.symbol == symbol).first()
        if not stock:
            raise HTTPException(status_code=404, detail="Stock not found in watchlist")

        # Format CIK - remove leading zeros as per API docs
        cik = str(int(stock.cik))  # Remove leading zeros
        logger.info(f"Using CIK: {cik}")
        
        # Query for latest 10-K using proper format
        query = {
            "query": f'cik:{cik} AND formType:"10-K"',
            "from": "0",
            "size": "1",
            "sort": [{"filedAt": {"order": "desc"}}]
        }
        
        logger.info(f"SEC API Query: {json.dumps(query, indent=2)}")
        filings = queryApi.get_filings(query)
        logger.info(f"SEC API Response: {json.dumps(filings, indent=2)}")
        
        if not filings or 'filings' not in filings or not filings['filings']:
            # Try alternative query format
            alternative_query = {
                "query": f'ticker:{symbol} AND formType:"10-K"',
                "from": "0",
                "size": "1",
                "sort": [{"filedAt": {"order": "desc"}}]
            }
            logger.info(f"Trying alternative query by ticker: {json.dumps(alternative_query, indent=2)}")
            filings = queryApi.get_filings(alternative_query)
            logger.info(f"Alternative query response: {json.dumps(filings, indent=2)}")
            
            if not filings or 'filings' not in filings or not filings['filings']:
                raise HTTPException(status_code=404, detail="No 10-K filings found")
        
        latest_filing = filings['filings'][0]
        filing_url = latest_filing['linkToFilingDetails']
        
        # Get the filing date and parse it correctly
        filed_at = latest_filing['filedAt']
        try:
            # Handle both -05:00 and -04:00 timezones
            base_date = filed_at.rsplit('-', 1)[0]  # More robust splitting
            filing_date = datetime.strptime(base_date, '%Y-%m-%dT%H:%M:%S')
        except ValueError as e:
            logger.error(f"Failed to parse date '{filed_at}': {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Invalid date format from SEC API: {filed_at}"
            )
        
        new_filing = StockFiling(
            stock_id=stock.id,
            filing_type="10-K",
            filing_date=filing_date,
            filing_url=filing_url,
            filing_data={
                'accessionNo': latest_filing['accessionNo'],
                'description': latest_filing.get('description', ''),
                'periodOfReport': latest_filing.get('periodOfReport', '')
            }
        )
        db.add(new_filing)
        db.commit()
        
        return {
            "status": "success",
            "filing_type": "10-K",
            "filing_date": new_filing.filing_date.strftime("%Y-%m-%d"),
            "filing_url": filing_url,
            "message": f"Successfully fetched latest 10-K from {new_filing.filing_date.strftime('%Y-%m-%d')}"
        }
            
    except ValueError as e:
        logger.error(f"Date parsing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Invalid date format: {str(e)}")

@app.get("/stock/{symbol}/metrics")
async def get_stock_metrics(symbol: str, db: Session = Depends(get_db)):
    """Get financial metrics for a stock"""
    try:
        stock = db.query(WatchlistStock).filter(WatchlistStock.symbol == symbol).first()
        if not stock:
            raise HTTPException(status_code=404, detail="Stock not found in watchlist")
        
        # Get latest metrics
        metrics = db.query(FinancialMetrics)\
            .filter(FinancialMetrics.stock_id == stock.id)\
            .order_by(FinancialMetrics.date.desc())\
            .first()

        if not metrics:
            logger.info(f"No metrics found for {symbol}, fetching new data...")
            # Add metrics fetching logic here
            return {}

        logger.info(f"Retrieved metrics for {symbol}: {metrics.metrics}")
        return metrics.metrics

    except Exception as e:
        logger.error(f"Error fetching financial metrics: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)