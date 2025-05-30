import hashlib
import json
import logging
from datetime import datetime, timedelta
from functools import wraps
from typing import Optional, Dict, Any, List
import asyncio
from collections import defaultdict
import time

logger = logging.getLogger(__name__)

# Rate limiting
class RateLimiter:
    def __init__(self):
        self.requests = defaultdict(list)
        
    def is_allowed(self, key: str, max_requests: int = 100, window_seconds: int = 3600) -> bool:
        """Check if request is allowed based on rate limit"""
        now = time.time()
        # Clean old requests
        self.requests[key] = [req_time for req_time in self.requests[key] 
                             if now - req_time < window_seconds]
        
        if len(self.requests[key]) < max_requests:
            self.requests[key].append(now)
            return True
        return False

rate_limiter = RateLimiter()

# Simple in-memory cache
class SimpleCache:
    def __init__(self, ttl_seconds: int = 3600):
        self.cache = {}
        self.ttl_seconds = ttl_seconds
        
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            value, timestamp = self.cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.ttl_seconds):
                return value
            else:
                del self.cache[key]
        return None
        
    def set(self, key: str, value: Any):
        self.cache[key] = (value, datetime.now())
        
    def clear(self):
        self.cache.clear()

# Category cache instance
category_cache = SimpleCache(ttl_seconds=86400)  # 24 hours

# Cache decorator
def cache_result(ttl_seconds: int = 3600):
    def decorator(func):
        cache = SimpleCache(ttl_seconds)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache_key = hashlib.md5(
                f"{func.__name__}:{str(args)}:{str(kwargs)}".encode()
            ).hexdigest()
            
            cached = cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached
                
            result = await func(*args, **kwargs)
            cache.set(cache_key, result)
            return result
            
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            cache_key = hashlib.md5(
                f"{func.__name__}:{str(args)}:{str(kwargs)}".encode()
            ).hexdigest()
            
            cached = cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached
                
            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            return result
            
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# Data validation helpers
def validate_csv_data(df) -> tuple[bool, Optional[str]]:
    """Validate uploaded CSV data"""
    required_cols = {"Date", "Amount", "Description"}
    
    if not required_cols.issubset(set(df.columns)):
        missing = required_cols - set(df.columns)
        return False, f"Missing required columns: {', '.join(missing)}"
        
    # Check for empty data
    if len(df) == 0:
        return False, "CSV file is empty"
        
    # Check data types can be converted
    try:
        # Test date parsing
        sample_date = df["Date"].iloc[0]
        if isinstance(sample_date, str):
            # Try common date formats
            for fmt in ["%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y"]:
                try:
                    datetime.strptime(sample_date, fmt)
                    break
                except:
                    continue
            else:
                return False, "Unable to parse date format"
                
        # Test amount conversion
        float(df["Amount"].iloc[0])
        
    except Exception as e:
        return False, f"Data validation error: {str(e)}"
        
    return True, None

# Budget checking helper
def check_budget_alerts(db_session, user_id: str, category_totals: Dict[str, float]) -> List[Dict[str, Any]]:
    """Check if any budgets are exceeded and return alerts"""
    from models import Budget, AlertLog
    
    alerts = []
    budgets = db_session.query(Budget).filter(
        Budget.user_id == user_id,
        Budget.is_active == True
    ).all()
    
    for budget in budgets:
        category_total = category_totals.get(budget.category, 0)
        if category_total > budget.monthly_limit:
            percentage = (category_total / budget.monthly_limit) * 100
            alert = {
                "category": budget.category,
                "limit": budget.monthly_limit,
                "spent": category_total,
                "percentage": percentage,
                "message": f"Budget exceeded for {budget.category}: ${category_total:.2f} of ${budget.monthly_limit:.2f} ({percentage:.0f}%)"
            }
            alerts.append(alert)
            
            # Log alert to database
            alert_log = AlertLog(
                user_id=user_id,
                alert_type="budget_exceeded",
                message=alert["message"],
                metadata=alert
            )
            db_session.add(alert_log)
            
    return alerts

# Export data helpers
def prepare_export_data(summary: Dict[str, Any], format: str = "csv") -> Dict[str, Any]:
    """Prepare financial data for export"""
    export_data = {
        "metadata": {
            "exported_at": datetime.now().isoformat(),
            "total_expenses": summary.get("total_expenses", 0),
            "average_daily": summary.get("average_daily", 0),
            "average_monthly": summary.get("average_monthly", 0)
        }
    }
    
    if format == "csv":
        # Flatten data for CSV export
        transactions = []
        for daily in summary.get("daily_expenses", []):
            transactions.append({
                "date": daily["date"],
                "amount": daily["amount"],
                "type": "daily_total"
            })
            
        export_data["transactions"] = transactions
        
    elif format == "json":
        # Include full summary for JSON export
        export_data.update(summary)
        
    return export_data

# Async task queue for background processing
class AsyncTaskQueue:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.workers = []
        
    async def add_task(self, task_func, *args, **kwargs):
        await self.queue.put((task_func, args, kwargs))
        
    async def worker(self):
        while True:
            try:
                task_func, args, kwargs = await self.queue.get()
                await task_func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Task error: {e}")
            finally:
                self.queue.task_done()
                
    def start_workers(self, num_workers: int = 3):
        for _ in range(num_workers):
            worker = asyncio.create_task(self.worker())
            self.workers.append(worker)

# Initialize task queue
task_queue = AsyncTaskQueue()

# Pagination helper
def paginate_query(query, page: int = 1, per_page: int = 20):
    """Add pagination to SQLAlchemy query"""
    total = query.count()
    items = query.offset((page - 1) * per_page).limit(per_page).all()
    
    return {
        "items": items,
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": (total + per_page - 1) // per_page
    } 