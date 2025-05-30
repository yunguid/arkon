"""
Machine Learning API endpoints for financial analysis
"""

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket
from sqlalchemy.orm import Session
import pandas as pd
import polars as pl
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import asyncio
import logging

from database import get_db
from models import FinancialDocument, Budget
from ml_engine import ml_engine, PredictionResult
from websocket_server import websocket_endpoint, connection_manager

logger = logging.getLogger(__name__)

# Create ML router
ml_router = APIRouter(prefix="/ml", tags=["machine_learning"])

async def get_transaction_data(file_id: int, db: Session) -> pd.DataFrame:
    """Retrieve and parse transaction data from database"""
    doc = db.query(FinancialDocument).filter(FinancialDocument.id == file_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Parse summary data
    summary = json.loads(doc.summary_json)
    
    # Reconstruct transaction data from summary
    transactions = []
    
    # Daily expenses
    for daily in summary.get('daily_expenses', []):
        transactions.append({
            'Date': pd.to_datetime(daily['date']),
            'Amount': daily['amount'],
            'Description': 'Daily Total',
            'Type': 'aggregate'
        })
    
    # Add category information if available
    if 'ai_categories' in summary:
        for cat in summary['ai_categories']:
            transactions.append({
                'Date': pd.to_datetime(summary.get('date_range', {}).get('end', datetime.now())),
                'Amount': cat['amount'],
                'Description': f"{cat['main_category']}/{cat['sub_category']}",
                'main_category': cat['main_category'],
                'sub_category': cat['sub_category'],
                'detail_category': cat['detail_category'],
                'Type': 'category'
            })
    
    df = pd.DataFrame(transactions)
    return df

@ml_router.post("/train")
async def train_ml_models(file_id: int, db: Session = Depends(get_db)):
    """Train ML models on financial data"""
    try:
        # Get transaction data
        df = await get_transaction_data(file_id, db)
        
        if len(df) < 10:
            raise HTTPException(status_code=400, detail="Insufficient data for training")
        
        # Train models
        await ml_engine.train_models(df)
        
        # Save models
        ml_engine.save_models(f"models/financial_ml_{file_id}.pkl")
        
        return {
            "status": "success",
            "message": "Models trained successfully",
            "data_points": len(df)
        }
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@ml_router.get("/predict")
async def predict_spending(
    file_id: int,
    days: int = Query(default=30, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """Get spending predictions"""
    try:
        # Check if models are trained
        if not ml_engine.models_trained:
            # Try to load saved models
            try:
                ml_engine.load_models(f"models/financial_ml_{file_id}.pkl")
            except:
                # Train if not available
                df = await get_transaction_data(file_id, db)
                await ml_engine.train_models(df)
        
        # Get predictions
        prediction = await ml_engine.predict_future_spending(days)
        
        # Send real-time update via WebSocket
        await connection_manager.broadcast({
            "type": "prediction_update",
            "data": {
                "file_id": file_id,
                "prediction": prediction.value,
                "confidence": prediction.confidence,
                "days": days
            },
            "timestamp": datetime.now()
        })
        
        return {
            "prediction_type": prediction.prediction_type,
            "value": prediction.value,
            "confidence": prediction.confidence,
            "insights": prediction.insights,
            "metadata": prediction.metadata
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@ml_router.get("/anomalies")
async def detect_anomalies(
    file_id: int,
    threshold: float = Query(default=0.05, ge=0.01, le=0.2),
    db: Session = Depends(get_db)
):
    """Detect anomalous transactions"""
    try:
        # Get transaction data
        df = await get_transaction_data(file_id, db)
        
        # Ensure models are trained
        if not ml_engine.models_trained:
            await ml_engine.train_models(df)
        
        # Detect anomalies
        anomalies = await ml_engine.detect_anomalies(df)
        
        # Filter by threshold
        filtered_anomalies = [a for a in anomalies if a['anomaly_score'] > threshold]
        
        # Send alerts for high-score anomalies
        for anomaly in filtered_anomalies[:3]:  # Top 3
            if anomaly['anomaly_score'] > 0.8:
                await connection_manager.broadcast({
                    "type": "anomaly_detected",
                    "data": anomaly,
                    "timestamp": datetime.now()
                })
        
        return {
            "anomalies": filtered_anomalies,
            "total_detected": len(anomalies),
            "threshold": threshold
        }
        
    except Exception as e:
        logger.error(f"Anomaly detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@ml_router.get("/health-score")
async def get_financial_health_score(
    file_id: int,
    db: Session = Depends(get_db)
):
    """Calculate financial health score"""
    try:
        # Get transaction data
        df = await get_transaction_data(file_id, db)
        
        # Get health score
        health_score = await ml_engine.get_financial_health_score(df)
        
        # Check budgets for accurate budget adherence score
        doc = db.query(FinancialDocument).filter(FinancialDocument.id == file_id).first()
        user_id = doc.user_id if doc else "default"
        
        budgets = db.query(Budget).filter(
            Budget.user_id == user_id,
            Budget.is_active == True
        ).all()
        
        if budgets and 'ai_categories' in json.loads(doc.summary_json):
            # Calculate real budget adherence
            summary = json.loads(doc.summary_json)
            category_totals = {}
            
            for cat in summary['ai_categories']:
                category_totals[cat['main_category']] = cat['amount']
            
            exceeded_count = 0
            for budget in budgets:
                if budget.category in category_totals:
                    if category_totals[budget.category] > budget.monthly_limit:
                        exceeded_count += 1
            
            budget_score = max(0, 100 - (exceeded_count / len(budgets) * 100))
            health_score['components']['budget_adherence'] = budget_score
            
            # Recalculate overall score
            health_score['overall_score'] = sum(health_score['components'].values()) / len(health_score['components'])
            health_score['grade'] = ml_engine._get_grade(health_score['overall_score'])
        
        return health_score
        
    except Exception as e:
        logger.error(f"Health score error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@ml_router.get("/insights")
async def get_smart_insights(
    file_id: int,
    db: Session = Depends(get_db)
):
    """Get AI-powered financial insights"""
    try:
        # Get transaction data
        df = await get_transaction_data(file_id, db)
        
        # Get insights
        insights = await ml_engine.get_smart_insights(df)
        
        # Add budget-based insights
        doc = db.query(FinancialDocument).filter(FinancialDocument.id == file_id).first()
        if doc:
            summary = json.loads(doc.summary_json)
            
            # Check for savings opportunities
            if 'statistics' in summary:
                stats = summary['statistics']
                if stats.get('std_deviation', 0) > stats.get('median_transaction', 1) * 0.5:
                    insights.append({
                        "type": "savings_opportunity",
                        "title": "High Spending Variability",
                        "description": f"Your spending varies significantly (std: ${stats['std_deviation']:.2f}). Consider setting stricter daily limits.",
                        "severity": "warning",
                        "value": stats['std_deviation']
                    })
            
            # Check for category concentration
            if 'ai_categories' in summary and len(summary['ai_categories']) > 0:
                top_category = summary['ai_categories'][0]
                total = sum(cat['amount'] for cat in summary['ai_categories'])
                concentration = (top_category['amount'] / total) * 100
                
                if concentration > 40:
                    insights.append({
                        "type": "category_analysis",
                        "title": "Spending Concentration Alert",
                        "description": f"{concentration:.0f}% of your spending is in {top_category['main_category']}. Consider diversifying.",
                        "severity": "info",
                        "value": concentration
                    })
        
        # Send top insights via WebSocket
        for insight in insights[:3]:
            await connection_manager.broadcast({
                "type": "insight",
                "data": insight,
                "timestamp": datetime.now()
            })
        
        return {
            "insights": insights,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Insights error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@ml_router.get("/recurring-analysis")
async def analyze_recurring_payments(
    file_id: int,
    db: Session = Depends(get_db)
):
    """Detailed analysis of recurring payments"""
    try:
        df = await get_transaction_data(file_id, db)
        
        # This would need actual transaction-level data
        # For now, return structured analysis
        doc = db.query(FinancialDocument).filter(FinancialDocument.id == file_id).first()
        summary = json.loads(doc.summary_json)
        
        recurring = summary.get('recurring_transactions', [])
        
        # Analyze patterns
        analysis = {
            "total_recurring": len(recurring),
            "monthly_commitment": sum(r.get('averageamount', 0) for r in recurring),
            "categories": {},
            "recommendations": []
        }
        
        # Categorize recurring payments
        for r in recurring:
            desc = r.get('description', '').upper()
            category = 'Other'
            
            if any(x in desc for x in ['NETFLIX', 'SPOTIFY', 'HULU', 'DISNEY']):
                category = 'Entertainment'
            elif any(x in desc for x in ['GYM', 'FITNESS', 'YOGA']):
                category = 'Health & Fitness'
            elif any(x in desc for x in ['INSURANCE', 'GEICO', 'ALLSTATE']):
                category = 'Insurance'
            elif any(x in desc for x in ['VERIZON', 'ATT', 'TMOBILE', 'COMCAST']):
                category = 'Utilities'
            
            if category not in analysis['categories']:
                analysis['categories'][category] = {
                    'count': 0,
                    'total': 0,
                    'items': []
                }
            
            analysis['categories'][category]['count'] += 1
            analysis['categories'][category]['total'] += r.get('averageamount', 0)
            analysis['categories'][category]['items'].append({
                'description': r.get('description'),
                'amount': r.get('averageamount', 0),
                'frequency': r.get('count', 0)
            })
        
        # Generate recommendations
        if analysis['monthly_commitment'] > 500:
            analysis['recommendations'].append(
                "Your recurring payments exceed $500/month. Review subscriptions for potential savings."
            )
        
        entertainment_total = analysis['categories'].get('Entertainment', {}).get('total', 0)
        if entertainment_total > 100:
            analysis['recommendations'].append(
                f"You're spending ${entertainment_total:.2f}/month on entertainment subscriptions. Consider bundling or eliminating redundant services."
            )
        
        return analysis
        
    except Exception as e:
        logger.error(f"Recurring analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@ml_router.websocket("/ws/{client_id}")
async def ml_websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time ML updates"""
    await websocket_endpoint(websocket, client_id, connection_manager)

@ml_router.get("/comparison")
async def compare_periods(
    file_id: int,
    period1_start: str,
    period1_end: str,
    period2_start: str,
    period2_end: str,
    db: Session = Depends(get_db)
):
    """Compare spending between two periods"""
    try:
        doc = db.query(FinancialDocument).filter(FinancialDocument.id == file_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="File not found")
        
        summary = json.loads(doc.summary_json)
        
        # Parse dates
        p1_start = pd.to_datetime(period1_start)
        p1_end = pd.to_datetime(period1_end)
        p2_start = pd.to_datetime(period2_start)
        p2_end = pd.to_datetime(period2_end)
        
        # Filter daily expenses
        daily_expenses = summary.get('daily_expenses', [])
        
        period1_total = sum(
            d['amount'] for d in daily_expenses
            if p1_start <= pd.to_datetime(d['date']) <= p1_end
        )
        
        period2_total = sum(
            d['amount'] for d in daily_expenses
            if p2_start <= pd.to_datetime(d['date']) <= p2_end
        )
        
        # Calculate metrics
        change = period2_total - period1_total
        change_pct = (change / period1_total * 100) if period1_total > 0 else 0
        
        # Period lengths
        p1_days = (p1_end - p1_start).days + 1
        p2_days = (p2_end - p2_start).days + 1
        
        comparison = {
            "period1": {
                "start": period1_start,
                "end": period1_end,
                "total": period1_total,
                "daily_avg": period1_total / p1_days if p1_days > 0 else 0,
                "days": p1_days
            },
            "period2": {
                "start": period2_start,
                "end": period2_end,
                "total": period2_total,
                "daily_avg": period2_total / p2_days if p2_days > 0 else 0,
                "days": p2_days
            },
            "change": {
                "amount": change,
                "percentage": change_pct,
                "daily_avg_change": (period2_total / p2_days) - (period1_total / p1_days) if p1_days > 0 and p2_days > 0 else 0
            },
            "insights": []
        }
        
        # Generate insights
        if change_pct > 20:
            comparison['insights'].append(
                f"Spending increased significantly by {change_pct:.1f}% between periods"
            )
        elif change_pct < -20:
            comparison['insights'].append(
                f"Great job! Spending decreased by {abs(change_pct):.1f}% between periods"
            )
        
        if comparison['period2']['daily_avg'] > comparison['period1']['daily_avg'] * 1.5:
            comparison['insights'].append(
                "Daily average spending increased by more than 50%. Review recent transactions."
            )
        
        return comparison
        
    except Exception as e:
        logger.error(f"Comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Include router in main app
def include_ml_routes(app):
    """Include ML routes in the main FastAPI app"""
    app.include_router(ml_router) 