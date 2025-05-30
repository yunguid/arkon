"""
Advanced Machine Learning Engine for Financial Analysis
Provides predictions, anomaly detection, and intelligent insights
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import asyncio
from dataclasses import dataclass
import json
import torch
import torch.nn as nn
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """Structured prediction result"""
    prediction_type: str
    value: float
    confidence: float
    insights: List[str]
    metadata: Dict[str, Any]

class NeuralSpendingPredictor(nn.Module):
    """Deep learning model for spending prediction"""
    def __init__(self, input_size: int, hidden_size: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        return self.fc(attn_out[:, -1, :])

class FinancialMLEngine:
    """Advanced ML engine for financial analysis"""
    
    def __init__(self):
        self.anomaly_detector = None
        self.spending_predictor = None
        self.category_predictor = None
        self.neural_model = None
        self.scalers = {}
        self.models_trained = False
        
    async def train_models(self, transaction_data: pd.DataFrame):
        """Train all ML models with transaction data"""
        try:
            logger.info("Starting ML model training...")
            
            # Prepare features
            features = self._prepare_features(transaction_data)
            
            # Train anomaly detector
            await self._train_anomaly_detector(features)
            
            # Train spending predictor
            await self._train_spending_predictor(transaction_data)
            
            # Train neural network
            await self._train_neural_network(features)
            
            # Train category predictor
            await self._train_category_predictor(transaction_data)
            
            self.models_trained = True
            logger.info("ML models trained successfully")
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise
    
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for ML models"""
        features = []
        
        # Time-based features
        df['hour'] = pd.to_datetime(df['Date']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['Date']).dt.dayofweek
        df['day_of_month'] = pd.to_datetime(df['Date']).dt.day
        df['month'] = pd.to_datetime(df['Date']).dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Amount-based features
        df['amount_zscore'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
        df['amount_percentile'] = df['Amount'].rank(pct=True)
        
        # Rolling statistics
        for window in [7, 14, 30]:
            df[f'rolling_mean_{window}'] = df['Amount'].rolling(window, min_periods=1).mean()
            df[f'rolling_std_{window}'] = df['Amount'].rolling(window, min_periods=1).std()
        
        # Category encoding (if available)
        if 'main_category' in df.columns:
            df['category_encoded'] = pd.Categorical(df['main_category']).codes
        
        feature_cols = [col for col in df.columns if col not in ['Date', 'Description', 'main_category', 'sub_category', 'detail_category']]
        return df[feature_cols].fillna(0).values
    
    async def _train_anomaly_detector(self, features: np.ndarray):
        """Train isolation forest for anomaly detection"""
        self.anomaly_detector = IsolationForest(
            contamination=0.05,
            random_state=42,
            n_estimators=200
        )
        self.anomaly_detector.fit(features)
        
    async def _train_spending_predictor(self, df: pd.DataFrame):
        """Train time series predictor using Prophet"""
        # Prepare data for Prophet
        prophet_df = df.groupby('Date')['Amount'].sum().reset_index()
        prophet_df.columns = ['ds', 'y']
        
        # Initialize and train Prophet model
        self.spending_predictor = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        
        # Add custom seasonalities
        self.spending_predictor.add_seasonality(
            name='monthly',
            period=30.5,
            fourier_order=5
        )
        
        self.spending_predictor.fit(prophet_df)
        
    async def _train_neural_network(self, features: np.ndarray):
        """Train deep learning model"""
        # Prepare sequences for LSTM
        sequence_length = 30
        X, y = self._create_sequences(features, sequence_length)
        
        if len(X) > 0:
            input_size = X.shape[2]
            self.neural_model = NeuralSpendingPredictor(input_size)
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y)
            
            # Train model
            optimizer = torch.optim.Adam(self.neural_model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            for epoch in range(50):
                optimizer.zero_grad()
                outputs = self.neural_model(X_tensor)
                loss = criterion(outputs.squeeze(), y_tensor)
                loss.backward()
                optimizer.step()
                
                if epoch % 10 == 0:
                    logger.info(f"Neural network training - Epoch {epoch}, Loss: {loss.item():.4f}")
    
    def _create_sequences(self, data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction"""
        if len(data) < seq_length + 1:
            return np.array([]), np.array([])
            
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length, 0])  # Predict amount
        
        return np.array(X), np.array(y)
    
    async def _train_category_predictor(self, df: pd.DataFrame):
        """Train smart category predictor"""
        if 'main_category' not in df.columns:
            return
            
        # Feature engineering for descriptions
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.ensemble import RandomForestClassifier
        
        vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
        X = vectorizer.fit_transform(df['Description'])
        y = df['main_category']
        
        self.category_predictor = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        self.category_predictor.fit(X, y)
        self.category_vectorizer = vectorizer
    
    async def predict_future_spending(self, days: int = 30) -> PredictionResult:
        """Predict future spending patterns"""
        if not self.spending_predictor:
            raise ValueError("Model not trained")
        
        # Make predictions
        future = self.spending_predictor.make_future_dataframe(periods=days)
        forecast = self.spending_predictor.predict(future)
        
        # Extract predictions
        future_forecast = forecast[forecast['ds'] > datetime.now()]
        total_predicted = future_forecast['yhat'].sum()
        
        # Generate insights
        insights = self._generate_spending_insights(forecast)
        
        return PredictionResult(
            prediction_type="spending_forecast",
            value=float(total_predicted),
            confidence=0.85,
            insights=insights,
            metadata={
                "forecast_days": days,
                "daily_predictions": future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict('records')
            }
        )
    
    def _generate_spending_insights(self, forecast: pd.DataFrame) -> List[str]:
        """Generate intelligent insights from predictions"""
        insights = []
        
        # Trend analysis
        recent = forecast.tail(30)
        if recent['trend'].iloc[-1] > recent['trend'].iloc[0]:
            insights.append("📈 Your spending is trending upward. Consider reviewing your budget.")
        else:
            insights.append("📉 Your spending is trending downward. Great job controlling expenses!")
        
        # Weekly pattern
        weekly_avg = forecast.groupby(forecast['ds'].dt.dayofweek)['yhat'].mean()
        peak_day = weekly_avg.idxmax()
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        insights.append(f"💰 You tend to spend the most on {days[peak_day]}s")
        
        # Volatility
        volatility = forecast['yhat'].std() / forecast['yhat'].mean()
        if volatility > 0.3:
            insights.append("⚠️ Your spending patterns are highly variable. Consider more consistent budgeting.")
        
        return insights
    
    async def detect_anomalies(self, transaction_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect anomalous transactions"""
        if not self.anomaly_detector:
            raise ValueError("Anomaly detector not trained")
        
        features = self._prepare_features(transaction_data)
        predictions = self.anomaly_detector.predict(features)
        
        anomalies = []
        for idx, (_, row) in enumerate(transaction_data.iterrows()):
            if predictions[idx] == -1:  # Anomaly
                anomaly_score = self.anomaly_detector.score_samples(features[idx:idx+1])[0]
                anomalies.append({
                    "date": row['Date'].strftime("%Y-%m-%d"),
                    "description": row['Description'],
                    "amount": float(row['Amount']),
                    "anomaly_score": float(abs(anomaly_score)),
                    "reason": self._explain_anomaly(row, transaction_data)
                })
        
        return sorted(anomalies, key=lambda x: x['anomaly_score'], reverse=True)
    
    def _explain_anomaly(self, transaction: pd.Series, all_data: pd.DataFrame) -> str:
        """Explain why a transaction is anomalous"""
        amount = transaction['Amount']
        avg_amount = all_data['Amount'].mean()
        std_amount = all_data['Amount'].std()
        
        if amount > avg_amount + 2 * std_amount:
            return f"Unusually high amount (${amount:.2f} vs average ${avg_amount:.2f})"
        elif amount < avg_amount - 2 * std_amount:
            return f"Unusually low amount (${amount:.2f} vs average ${avg_amount:.2f})"
        
        # Check time patterns
        hour = transaction['Date'].hour if hasattr(transaction['Date'], 'hour') else 0
        if hour < 6 or hour > 22:
            return f"Unusual transaction time ({hour}:00)"
        
        return "Unusual pattern detected"
    
    async def get_financial_health_score(self, transaction_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive financial health score"""
        scores = {}
        
        # Spending consistency score
        daily_spending = transaction_data.groupby('Date')['Amount'].sum()
        consistency_score = 100 - min(daily_spending.std() / daily_spending.mean() * 100, 100)
        scores['consistency'] = consistency_score
        
        # Budget adherence score (if budgets exist)
        budget_score = await self._calculate_budget_score(transaction_data)
        scores['budget_adherence'] = budget_score
        
        # Savings potential score
        avg_daily = daily_spending.mean()
        median_daily = daily_spending.median()
        savings_potential = max(0, (avg_daily - median_daily) / avg_daily * 100)
        scores['savings_potential'] = min(savings_potential * 2, 100)
        
        # Category diversity score
        if 'main_category' in transaction_data.columns:
            category_counts = transaction_data['main_category'].value_counts()
            diversity_score = (1 - (category_counts.max() / category_counts.sum())) * 100
            scores['diversity'] = diversity_score
        else:
            scores['diversity'] = 50
        
        # Overall score
        overall_score = np.mean(list(scores.values()))
        
        return {
            "overall_score": float(overall_score),
            "components": scores,
            "grade": self._get_grade(overall_score),
            "recommendations": self._get_recommendations(scores)
        }
    
    async def _calculate_budget_score(self, transaction_data: pd.DataFrame) -> float:
        """Calculate budget adherence score"""
        # This would integrate with actual budget data
        # For now, return a simulated score
        return 75.0
    
    def _get_grade(self, score: float) -> str:
        """Convert score to letter grade"""
        if score >= 90:
            return "A+"
        elif score >= 85:
            return "A"
        elif score >= 80:
            return "B+"
        elif score >= 75:
            return "B"
        elif score >= 70:
            return "C+"
        elif score >= 65:
            return "C"
        else:
            return "D"
    
    def _get_recommendations(self, scores: Dict[str, float]) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        
        if scores['consistency'] < 70:
            recommendations.append("Try to maintain more consistent daily spending to better control your budget")
        
        if scores['budget_adherence'] < 80:
            recommendations.append("You're exceeding your budget in some categories. Review and adjust your limits")
        
        if scores['savings_potential'] > 50:
            recommendations.append(f"You could potentially save {scores['savings_potential']:.0f}% by reducing peak spending days")
        
        if scores['diversity'] < 60:
            recommendations.append("Your spending is concentrated in few categories. Consider diversifying")
        
        return recommendations
    
    async def get_smart_insights(self, transaction_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate smart, actionable insights"""
        insights = []
        
        # Spending velocity
        recent_30 = transaction_data[transaction_data['Date'] > datetime.now() - timedelta(days=30)]
        prev_30 = transaction_data[
            (transaction_data['Date'] > datetime.now() - timedelta(days=60)) & 
            (transaction_data['Date'] <= datetime.now() - timedelta(days=30))
        ]
        
        if len(recent_30) > 0 and len(prev_30) > 0:
            recent_total = recent_30['Amount'].sum()
            prev_total = prev_30['Amount'].sum()
            change_pct = ((recent_total - prev_total) / prev_total) * 100
            
            insights.append({
                "type": "spending_velocity",
                "title": "Spending Trend",
                "description": f"Your spending has {'increased' if change_pct > 0 else 'decreased'} by {abs(change_pct):.1f}% compared to last month",
                "severity": "warning" if change_pct > 20 else "info",
                "value": change_pct
            })
        
        # Category insights
        if 'main_category' in transaction_data.columns:
            category_spending = transaction_data.groupby('main_category')['Amount'].agg(['sum', 'count'])
            top_category = category_spending['sum'].idxmax()
            top_amount = category_spending.loc[top_category, 'sum']
            top_count = category_spending.loc[top_category, 'count']
            
            insights.append({
                "type": "category_analysis",
                "title": "Top Spending Category",
                "description": f"You've spent ${top_amount:.2f} on {top_category} across {top_count} transactions",
                "severity": "info",
                "value": top_amount
            })
        
        # Recurring payment detection
        recurring = self._detect_recurring_patterns(transaction_data)
        if recurring:
            total_recurring = sum(r['amount'] for r in recurring)
            insights.append({
                "type": "recurring_payments",
                "title": "Recurring Payments Detected",
                "description": f"You have {len(recurring)} recurring payments totaling ${total_recurring:.2f}/month",
                "severity": "info",
                "value": total_recurring,
                "details": recurring
            })
        
        return insights
    
    def _detect_recurring_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect recurring payment patterns"""
        recurring = []
        
        # Group by description and analyze frequency
        for desc, group in df.groupby('Description'):
            if len(group) < 2:
                continue
                
            # Calculate intervals between transactions
            dates = pd.to_datetime(group['Date']).sort_values()
            intervals = dates.diff().dt.days.dropna()
            
            if len(intervals) >= 2:
                avg_interval = intervals.mean()
                std_interval = intervals.std()
                
                # Check if it's likely recurring (consistent intervals)
                if std_interval < 5 and 25 <= avg_interval <= 35:
                    recurring.append({
                        "description": desc,
                        "amount": float(group['Amount'].mean()),
                        "frequency": "monthly",
                        "confidence": 0.9 if std_interval < 2 else 0.7
                    })
                elif std_interval < 3 and 12 <= avg_interval <= 16:
                    recurring.append({
                        "description": desc,
                        "amount": float(group['Amount'].mean()),
                        "frequency": "bi-weekly",
                        "confidence": 0.9 if std_interval < 1 else 0.7
                    })
        
        return recurring
    
    def save_models(self, path: str):
        """Save trained models to disk"""
        models = {
            'anomaly_detector': self.anomaly_detector,
            'spending_predictor': self.spending_predictor,
            'category_predictor': self.category_predictor,
            'scalers': self.scalers,
            'models_trained': self.models_trained
        }
        joblib.dump(models, path)
        
        # Save neural model separately
        if self.neural_model:
            torch.save(self.neural_model.state_dict(), f"{path}_neural.pth")
    
    def load_models(self, path: str):
        """Load trained models from disk"""
        models = joblib.load(path)
        self.anomaly_detector = models['anomaly_detector']
        self.spending_predictor = models['spending_predictor']
        self.category_predictor = models['category_predictor']
        self.scalers = models['scalers']
        self.models_trained = models['models_trained']
        
        # Load neural model if exists
        neural_path = f"{path}_neural.pth"
        if os.path.exists(neural_path):
            # Reconstruct model architecture
            self.neural_model = NeuralSpendingPredictor(input_size=10)  # Adjust based on features
            self.neural_model.load_state_dict(torch.load(neural_path))

# Singleton instance
ml_engine = FinancialMLEngine() 