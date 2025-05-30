"""
GraphQL API for Arkon Financial Intelligence
Provides advanced querying, mutations, and subscriptions
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, date
import strawberry
from strawberry import subscription
from strawberry.scalars import JSON
from strawberry.fastapi import GraphQLRouter
from strawberry.types import Info
import asyncio
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
import json

from database import SessionLocal
from models import (
    FinancialDocument, Budget, StockNews, Watchlist,
    StockAnalysisHistory, CategoryCache, AlertLog
)
from ml_engine import ml_engine
from websocket_server import connection_manager

# GraphQL Types
@strawberry.type
class FinancialDocumentType:
    id: int
    filename: str
    upload_date: datetime
    summary_json: JSON
    processing_status: str
    user_id: Optional[str]

@strawberry.type
class BudgetType:
    id: int
    user_id: str
    category: str
    subcategory: Optional[str]
    monthly_limit: float
    is_active: bool
    created_at: datetime
    updated_at: datetime

@strawberry.type
class AlertType:
    id: int
    user_id: str
    alert_type: str
    message: str
    metadata: JSON
    is_read: bool
    created_at: datetime

@strawberry.type
class MLPredictionType:
    prediction_type: str
    value: float
    confidence: float
    insights: List[str]
    metadata: JSON

@strawberry.type
class HealthScoreType:
    overall_score: float
    grade: str
    components: JSON
    recommendations: List[str]

@strawberry.type
class AnomalyType:
    date: str
    description: str
    amount: float
    anomaly_score: float
    reason: str

@strawberry.type
class CategoryInsightType:
    category: str
    amount: float
    percentage: float
    trend: str
    subcategories: JSON

@strawberry.type
class FinancialSummaryType:
    total_expenses: float
    average_daily: float
    average_monthly: float
    date_range: JSON
    top_categories: List[CategoryInsightType]
    budget_status: JSON
    predictions: Optional[MLPredictionType]

@strawberry.type
class WatchlistItemType:
    id: int
    symbol: str
    added_date: datetime
    last_analysis: Optional[datetime]
    price_alert_above: Optional[float]
    price_alert_below: Optional[float]
    notes: Optional[str]
    current_price: Optional[float]
    price_change: Optional[float]

@strawberry.type
class TransactionType:
    date: date
    amount: float
    description: str
    category: Optional[str]
    subcategory: Optional[str]
    is_recurring: bool
    anomaly_score: Optional[float]

# Input Types
@strawberry.input
class BudgetInput:
    category: str
    subcategory: Optional[str] = None
    monthly_limit: float

@strawberry.input
class DateRangeInput:
    start_date: date
    end_date: date

@strawberry.input
class PaginationInput:
    page: int = 1
    per_page: int = 20

@strawberry.input
class FilterInput:
    categories: Optional[List[str]] = None
    min_amount: Optional[float] = None
    max_amount: Optional[float] = None
    date_range: Optional[DateRangeInput] = None
    search_term: Optional[str] = None

# Queries
@strawberry.type
class Query:
    @strawberry.field
    async def financial_documents(
        self,
        info: Info,
        pagination: Optional[PaginationInput] = None,
        user_id: str = "default"
    ) -> List[FinancialDocumentType]:
        """Get paginated list of financial documents"""
        db: Session = info.context["db"]
        
        query = db.query(FinancialDocument).filter(
            FinancialDocument.user_id == user_id
        ).order_by(FinancialDocument.upload_date.desc())
        
        if pagination:
            offset = (pagination.page - 1) * pagination.per_page
            query = query.offset(offset).limit(pagination.per_page)
        
        documents = query.all()
        
        return [
            FinancialDocumentType(
                id=doc.id,
                filename=doc.filename,
                upload_date=doc.upload_date,
                summary_json=json.loads(doc.summary_json),
                processing_status=doc.processing_status,
                user_id=doc.user_id
            )
            for doc in documents
        ]
    
    @strawberry.field
    async def financial_summary(
        self,
        info: Info,
        file_id: int,
        include_predictions: bool = False
    ) -> FinancialSummaryType:
        """Get comprehensive financial summary with optional ML predictions"""
        db: Session = info.context["db"]
        
        doc = db.query(FinancialDocument).filter(
            FinancialDocument.id == file_id
        ).first()
        
        if not doc:
            raise Exception("Document not found")
        
        summary = json.loads(doc.summary_json)
        
        # Process categories
        top_categories = []
        if 'ai_categories' in summary:
            total = sum(cat['amount'] for cat in summary['ai_categories'])
            for cat in summary['ai_categories'][:5]:
                top_categories.append(CategoryInsightType(
                    category=cat['main_category'],
                    amount=cat['amount'],
                    percentage=(cat['amount'] / total * 100) if total > 0 else 0,
                    trend="stable",  # Would calculate from historical data
                    subcategories={cat['sub_category']: cat['amount']}
                ))
        
        # Get budget status
        user_id = doc.user_id or "default"
        budgets = db.query(Budget).filter(
            Budget.user_id == user_id,
            Budget.is_active == True
        ).all()
        
        budget_status = {}
        for budget in budgets:
            category_total = next(
                (cat['amount'] for cat in summary.get('ai_categories', [])
                 if cat['main_category'] == budget.category),
                0
            )
            budget_status[budget.category] = {
                "limit": budget.monthly_limit,
                "spent": category_total,
                "percentage": (category_total / budget.monthly_limit * 100)
                             if budget.monthly_limit > 0 else 0
            }
        
        # Get ML predictions if requested
        predictions = None
        if include_predictions and ml_engine.models_trained:
            try:
                pred_result = await ml_engine.predict_future_spending(30)
                predictions = MLPredictionType(
                    prediction_type=pred_result.prediction_type,
                    value=pred_result.value,
                    confidence=pred_result.confidence,
                    insights=pred_result.insights,
                    metadata=pred_result.metadata
                )
            except:
                pass
        
        return FinancialSummaryType(
            total_expenses=summary.get('total_expenses', 0),
            average_daily=summary.get('average_daily', 0),
            average_monthly=summary.get('average_monthly', 0),
            date_range=summary.get('date_range', {}),
            top_categories=top_categories,
            budget_status=budget_status,
            predictions=predictions
        )
    
    @strawberry.field
    async def budgets(
        self,
        info: Info,
        user_id: str = "default",
        active_only: bool = True
    ) -> List[BudgetType]:
        """Get user budgets"""
        db: Session = info.context["db"]
        
        query = db.query(Budget).filter(Budget.user_id == user_id)
        if active_only:
            query = query.filter(Budget.is_active == True)
        
        budgets = query.all()
        
        return [
            BudgetType(
                id=b.id,
                user_id=b.user_id,
                category=b.category,
                subcategory=b.subcategory,
                monthly_limit=b.monthly_limit,
                is_active=b.is_active,
                created_at=b.created_at,
                updated_at=b.updated_at
            )
            for b in budgets
        ]
    
    @strawberry.field
    async def alerts(
        self,
        info: Info,
        user_id: str = "default",
        unread_only: bool = False,
        limit: int = 50
    ) -> List[AlertType]:
        """Get user alerts"""
        db: Session = info.context["db"]
        
        query = db.query(AlertLog).filter(AlertLog.user_id == user_id)
        if unread_only:
            query = query.filter(AlertLog.is_read == False)
        
        alerts = query.order_by(AlertLog.created_at.desc()).limit(limit).all()
        
        return [
            AlertType(
                id=a.id,
                user_id=a.user_id,
                alert_type=a.alert_type,
                message=a.message,
                metadata=a.metadata or {},
                is_read=a.is_read,
                created_at=a.created_at
            )
            for a in alerts
        ]
    
    @strawberry.field
    async def ml_health_score(
        self,
        info: Info,
        file_id: int
    ) -> HealthScoreType:
        """Get ML-powered financial health score"""
        db: Session = info.context["db"]
        
        # Get transaction data
        doc = db.query(FinancialDocument).filter(
            FinancialDocument.id == file_id
        ).first()
        
        if not doc:
            raise Exception("Document not found")
        
        # Get health score from ML engine
        # This is simplified - would need actual transaction data
        health_data = {
            "overall_score": 75.5,
            "grade": "B+",
            "components": {
                "consistency": 80,
                "budget_adherence": 70,
                "savings_potential": 75,
                "diversity": 77
            },
            "recommendations": [
                "Reduce dining out expenses by 20%",
                "Consider automating savings transfers",
                "Review subscription services for potential cuts"
            ]
        }
        
        return HealthScoreType(
            overall_score=health_data["overall_score"],
            grade=health_data["grade"],
            components=health_data["components"],
            recommendations=health_data["recommendations"]
        )
    
    @strawberry.field
    async def detect_anomalies(
        self,
        info: Info,
        file_id: int,
        threshold: float = 0.7
    ) -> List[AnomalyType]:
        """Detect anomalous transactions"""
        db: Session = info.context["db"]
        
        # Simplified anomaly detection
        anomalies = [
            AnomalyType(
                date="2024-01-15",
                description="UNUSUAL STORE PURCHASE",
                amount=1250.00,
                anomaly_score=0.92,
                reason="Amount significantly higher than average"
            ),
            AnomalyType(
                date="2024-01-20",
                description="MIDNIGHT TRANSACTION",
                amount=89.99,
                anomaly_score=0.78,
                reason="Unusual transaction time (2:30 AM)"
            )
        ]
        
        return [a for a in anomalies if a.anomaly_score >= threshold]
    
    @strawberry.field
    async def watchlist(
        self,
        info: Info,
        user_id: str = "default"
    ) -> List[WatchlistItemType]:
        """Get user's stock watchlist with current prices"""
        db: Session = info.context["db"]
        
        items = db.query(Watchlist).filter(
            Watchlist.user_id == user_id
        ).all()
        
        watchlist = []
        for item in items:
            # Would fetch real-time price here
            current_price = 150.00 + (item.id * 10)  # Mock price
            price_change = 2.5  # Mock change
            
            watchlist.append(WatchlistItemType(
                id=item.id,
                symbol=item.symbol,
                added_date=item.added_date,
                last_analysis=item.last_analysis,
                price_alert_above=item.price_alert_above,
                price_alert_below=item.price_alert_below,
                notes=item.notes,
                current_price=current_price,
                price_change=price_change
            ))
        
        return watchlist
    
    @strawberry.field
    async def search_transactions(
        self,
        info: Info,
        file_id: int,
        filter: Optional[FilterInput] = None
    ) -> List[TransactionType]:
        """Search and filter transactions with advanced criteria"""
        db: Session = info.context["db"]
        
        # Get document
        doc = db.query(FinancialDocument).filter(
            FinancialDocument.id == file_id
        ).first()
        
        if not doc:
            raise Exception("Document not found")
        
        summary = json.loads(doc.summary_json)
        
        # Mock transaction data based on summary
        transactions = []
        for expense in summary.get('daily_expenses', [])[:10]:
            transactions.append(TransactionType(
                date=datetime.strptime(expense['date'], '%Y-%m-%d').date(),
                amount=expense['amount'],
                description="Daily transactions",
                category="Various",
                subcategory=None,
                is_recurring=False,
                anomaly_score=None
            ))
        
        # Apply filters
        if filter:
            if filter.min_amount:
                transactions = [t for t in transactions if t.amount >= filter.min_amount]
            if filter.max_amount:
                transactions = [t for t in transactions if t.amount <= filter.max_amount]
            if filter.date_range:
                transactions = [
                    t for t in transactions
                    if filter.date_range.start_date <= t.date <= filter.date_range.end_date
                ]
        
        return transactions

# Mutations
@strawberry.type
class Mutation:
    @strawberry.mutation
    async def create_budget(
        self,
        info: Info,
        budget_input: BudgetInput,
        user_id: str = "default"
    ) -> BudgetType:
        """Create a new budget"""
        db: Session = info.context["db"]
        
        # Check if budget exists
        existing = db.query(Budget).filter(
            Budget.user_id == user_id,
            Budget.category == budget_input.category,
            Budget.is_active == True
        ).first()
        
        if existing:
            raise Exception("Budget already exists for this category")
        
        budget = Budget(
            user_id=user_id,
            category=budget_input.category,
            subcategory=budget_input.subcategory,
            monthly_limit=budget_input.monthly_limit,
            is_active=True
        )
        
        db.add(budget)
        db.commit()
        db.refresh(budget)
        
        # Send notification
        await connection_manager.broadcast({
            "type": "budget_created",
            "data": {
                "category": budget.category,
                "limit": budget.monthly_limit
            }
        })
        
        return BudgetType(
            id=budget.id,
            user_id=budget.user_id,
            category=budget.category,
            subcategory=budget.subcategory,
            monthly_limit=budget.monthly_limit,
            is_active=budget.is_active,
            created_at=budget.created_at,
            updated_at=budget.updated_at
        )
    
    @strawberry.mutation
    async def update_budget(
        self,
        info: Info,
        budget_id: int,
        monthly_limit: float
    ) -> BudgetType:
        """Update budget limit"""
        db: Session = info.context["db"]
        
        budget = db.query(Budget).filter(Budget.id == budget_id).first()
        if not budget:
            raise Exception("Budget not found")
        
        budget.monthly_limit = monthly_limit
        budget.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(budget)
        
        return BudgetType(
            id=budget.id,
            user_id=budget.user_id,
            category=budget.category,
            subcategory=budget.subcategory,
            monthly_limit=budget.monthly_limit,
            is_active=budget.is_active,
            created_at=budget.created_at,
            updated_at=budget.updated_at
        )
    
    @strawberry.mutation
    async def mark_alert_read(
        self,
        info: Info,
        alert_id: int
    ) -> AlertType:
        """Mark an alert as read"""
        db: Session = info.context["db"]
        
        alert = db.query(AlertLog).filter(AlertLog.id == alert_id).first()
        if not alert:
            raise Exception("Alert not found")
        
        alert.is_read = True
        db.commit()
        db.refresh(alert)
        
        return AlertType(
            id=alert.id,
            user_id=alert.user_id,
            alert_type=alert.alert_type,
            message=alert.message,
            metadata=alert.metadata or {},
            is_read=alert.is_read,
            created_at=alert.created_at
        )
    
    @strawberry.mutation
    async def add_to_watchlist(
        self,
        info: Info,
        symbol: str,
        price_alert_above: Optional[float] = None,
        price_alert_below: Optional[float] = None,
        notes: Optional[str] = None,
        user_id: str = "default"
    ) -> WatchlistItemType:
        """Add stock to watchlist"""
        db: Session = info.context["db"]
        
        # Check if already exists
        existing = db.query(Watchlist).filter(
            Watchlist.user_id == user_id,
            Watchlist.symbol == symbol
        ).first()
        
        if existing:
            # Update existing
            existing.price_alert_above = price_alert_above
            existing.price_alert_below = price_alert_below
            existing.notes = notes
            watchlist_item = existing
        else:
            # Create new
            watchlist_item = Watchlist(
                user_id=user_id,
                symbol=symbol,
                price_alert_above=price_alert_above,
                price_alert_below=price_alert_below,
                notes=notes
            )
            db.add(watchlist_item)
        
        db.commit()
        db.refresh(watchlist_item)
        
        return WatchlistItemType(
            id=watchlist_item.id,
            symbol=watchlist_item.symbol,
            added_date=watchlist_item.added_date,
            last_analysis=watchlist_item.last_analysis,
            price_alert_above=watchlist_item.price_alert_above,
            price_alert_below=watchlist_item.price_alert_below,
            notes=watchlist_item.notes,
            current_price=150.00,  # Mock
            price_change=2.5  # Mock
        )

# Subscriptions
@strawberry.type
class Subscription:
    @strawberry.subscription
    async def budget_alerts(
        self,
        info: Info,
        user_id: str = "default"
    ) -> AlertType:
        """Subscribe to budget alerts"""
        while True:
            # Check for new alerts every 5 seconds
            await asyncio.sleep(5)
            
            db: Session = SessionLocal()
            try:
                # Get latest unread alert
                alert = db.query(AlertLog).filter(
                    AlertLog.user_id == user_id,
                    AlertLog.is_read == False,
                    AlertLog.alert_type == "budget_exceeded"
                ).order_by(AlertLog.created_at.desc()).first()
                
                if alert:
                    yield AlertType(
                        id=alert.id,
                        user_id=alert.user_id,
                        alert_type=alert.alert_type,
                        message=alert.message,
                        metadata=alert.metadata or {},
                        is_read=alert.is_read,
                        created_at=alert.created_at
                    )
            finally:
                db.close()
    
    @strawberry.subscription
    async def market_updates(
        self,
        info: Info,
        symbols: List[str]
    ) -> WatchlistItemType:
        """Subscribe to real-time market updates"""
        while True:
            await asyncio.sleep(2)  # Update every 2 seconds
            
            # Mock real-time data
            import random
            symbol = random.choice(symbols)
            
            yield WatchlistItemType(
                id=0,
                symbol=symbol,
                added_date=datetime.now(),
                last_analysis=None,
                price_alert_above=None,
                price_alert_below=None,
                notes=None,
                current_price=150.00 + random.uniform(-5, 5),
                price_change=random.uniform(-2, 2)
            )
    
    @strawberry.subscription
    async def ml_insights_stream(
        self,
        info: Info,
        file_id: int
    ) -> MLPredictionType:
        """Stream ML insights as they're generated"""
        while True:
            await asyncio.sleep(10)  # Generate insights every 10 seconds
            
            # Mock ML insights
            insights = [
                "Spending trend is increasing by 3% month-over-month",
                "Dining expenses are 25% above average this week",
                "Predicted overspend in Entertainment category by month end"
            ]
            
            yield MLPredictionType(
                prediction_type="real_time_insight",
                value=random.uniform(1000, 5000),
                confidence=random.uniform(0.7, 0.95),
                insights=random.sample(insights, 2),
                metadata={"timestamp": datetime.now().isoformat()}
            )

# Context getter for dependency injection
async def get_context() -> Dict[str, Any]:
    db = SessionLocal()
    try:
        return {"db": db}
    finally:
        db.close()

# Create GraphQL schema
schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    subscription=Subscription
)

# Create GraphQL app
graphql_app = GraphQLRouter(
    schema,
    context_getter=get_context,
    graphql_ide="apollo-sandbox"  # Use Apollo Studio sandbox
) 