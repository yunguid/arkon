import React, { useState, useEffect, useRef } from 'react';
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar, RadialBarChart, RadialBar,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, PieChart, Pie, Cell
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import './MLInsights.css';

const MLInsights = ({ fileId }) => {
  const [predictions, setPredictions] = useState(null);
  const [anomalies, setAnomalies] = useState([]);
  const [healthScore, setHealthScore] = useState(null);
  const [insights, setInsights] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedTimeframe, setSelectedTimeframe] = useState(30);
  const [wsConnected, setWsConnected] = useState(false);
  const ws = useRef(null);
  
  // Colors for charts
  const COLORS = {
    primary: '#3498db',
    secondary: '#2ecc71',
    warning: '#f39c12',
    danger: '#e74c3c',
    info: '#9b59b6',
    dark: '#2c3e50'
  };

  // WebSocket connection
  useEffect(() => {
    const clientId = `user_${Math.random().toString(36).substring(7)}`;
    ws.current = new WebSocket(`ws://localhost:8000/ws/${clientId}`);
    
    ws.current.onopen = () => {
      console.log('WebSocket connected');
      setWsConnected(true);
      ws.current.send(JSON.stringify({ type: 'ping' }));
    };
    
    ws.current.onmessage = (event) => {
      const message = JSON.parse(event.data);
      handleWebSocketMessage(message);
    };
    
    ws.current.onerror = (error) => {
      console.error('WebSocket error:', error);
      setWsConnected(false);
    };
    
    ws.current.onclose = () => {
      console.log('WebSocket disconnected');
      setWsConnected(false);
    };
    
    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, []);

  const handleWebSocketMessage = (message) => {
    switch (message.type) {
      case 'notification':
        handleNotification(message.data);
        break;
      case 'anomaly_detected':
        setAnomalies(prev => [message.data, ...prev].slice(0, 10));
        break;
      case 'prediction_update':
        setPredictions(message.data);
        break;
      case 'insight':
        setInsights(prev => [message.data, ...prev].slice(0, 5));
        break;
      default:
        break;
    }
  };

  const handleNotification = (notification) => {
    // Show notification using a toast library or custom implementation
    console.log('Notification:', notification);
  };

  // Fetch ML data
  useEffect(() => {
    if (fileId) {
      fetchMLData();
    }
  }, [fileId, selectedTimeframe]);

  const fetchMLData = async () => {
    setLoading(true);
    try {
      // Fetch predictions
      const predResponse = await fetch(
        `http://localhost:8000/ml/predict?file_id=${fileId}&days=${selectedTimeframe}`
      );
      const predData = await predResponse.json();
      setPredictions(predData);

      // Fetch anomalies
      const anomResponse = await fetch(
        `http://localhost:8000/ml/anomalies?file_id=${fileId}`
      );
      const anomData = await anomResponse.json();
      setAnomalies(anomData.anomalies || []);

      // Fetch health score
      const healthResponse = await fetch(
        `http://localhost:8000/ml/health-score?file_id=${fileId}`
      );
      const healthData = await healthResponse.json();
      setHealthScore(healthData);

      // Fetch insights
      const insightsResponse = await fetch(
        `http://localhost:8000/ml/insights?file_id=${fileId}`
      );
      const insightsData = await insightsResponse.json();
      setInsights(insightsData.insights || []);
    } catch (error) {
      console.error('Error fetching ML data:', error);
    } finally {
      setLoading(false);
    }
  };

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(value);
  };

  const getGradeColor = (grade) => {
    const gradeColors = {
      'A+': COLORS.secondary,
      'A': COLORS.secondary,
      'B+': COLORS.primary,
      'B': COLORS.primary,
      'C+': COLORS.warning,
      'C': COLORS.warning,
      'D': COLORS.danger
    };
    return gradeColors[grade] || COLORS.dark;
  };

  const renderHealthScore = () => {
    if (!healthScore) return null;

    const scoreData = [
      {
        name: 'Overall Score',
        value: healthScore.overall_score,
        fill: getGradeColor(healthScore.grade)
      }
    ];

    const componentData = Object.entries(healthScore.components).map(([key, value]) => ({
      name: key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
      value: value,
      fill: value >= 80 ? COLORS.secondary : value >= 60 ? COLORS.primary : COLORS.warning
    }));

    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="health-score-section"
      >
        <h3>Financial Health Score</h3>
        <div className="health-score-container">
          <div className="overall-score">
            <ResponsiveContainer width="100%" height={200}>
              <RadialBarChart cx="50%" cy="50%" innerRadius="60%" outerRadius="90%" data={scoreData}>
                <RadialBar dataKey="value" cornerRadius={10} fill={scoreData[0].fill} />
                <text x="50%" y="50%" textAnchor="middle" dominantBaseline="middle" className="score-text">
                  <tspan x="50%" dy="-10" fontSize="36" fontWeight="bold">{healthScore.grade}</tspan>
                  <tspan x="50%" dy="30" fontSize="18">{Math.round(healthScore.overall_score)}%</tspan>
                </text>
              </RadialBarChart>
            </ResponsiveContainer>
          </div>
          
          <div className="score-components">
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={componentData} layout="horizontal">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 100]} />
                <YAxis dataKey="name" type="category" width={120} />
                <Tooltip formatter={(value) => `${Math.round(value)}%`} />
                <Bar dataKey="value" radius={[0, 5, 5, 0]}>
                  {componentData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        
        {healthScore.recommendations && healthScore.recommendations.length > 0 && (
          <div className="recommendations">
            <h4>Recommendations</h4>
            <ul>
              {healthScore.recommendations.map((rec, index) => (
                <motion.li
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                >
                  {rec}
                </motion.li>
              ))}
            </ul>
          </div>
        )}
      </motion.div>
    );
  };

  const renderPredictions = () => {
    if (!predictions || !predictions.metadata?.daily_predictions) return null;

    const chartData = predictions.metadata.daily_predictions.map(pred => ({
      date: new Date(pred.ds).toLocaleDateString(),
      predicted: pred.yhat,
      lower: pred.yhat_lower,
      upper: pred.yhat_upper
    }));

    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
        className="predictions-section"
      >
        <h3>Spending Forecast</h3>
        <div className="timeframe-selector">
          {[7, 14, 30, 60, 90].map(days => (
            <button
              key={days}
              className={selectedTimeframe === days ? 'active' : ''}
              onClick={() => setSelectedTimeframe(days)}
            >
              {days}d
            </button>
          ))}
        </div>
        
        <div className="prediction-summary">
          <div className="summary-card">
            <h5>Total Predicted</h5>
            <p className="amount">{formatCurrency(predictions.value)}</p>
            <span className="confidence">Confidence: {(predictions.confidence * 100).toFixed(0)}%</span>
          </div>
          <div className="summary-card">
            <h5>Daily Average</h5>
            <p className="amount">{formatCurrency(predictions.value / selectedTimeframe)}</p>
          </div>
        </div>

        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={chartData}>
            <defs>
              <linearGradient id="predictionGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={COLORS.primary} stopOpacity={0.8}/>
                <stop offset="95%" stopColor={COLORS.primary} stopOpacity={0.1}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis />
            <Tooltip formatter={(value) => formatCurrency(value)} />
            <Area
              type="monotone"
              dataKey="upper"
              stroke="none"
              fill={COLORS.primary}
              fillOpacity={0.2}
            />
            <Area
              type="monotone"
              dataKey="lower"
              stroke="none"
              fill="#fff"
            />
            <Line
              type="monotone"
              dataKey="predicted"
              stroke={COLORS.primary}
              strokeWidth={3}
              dot={false}
            />
          </AreaChart>
        </ResponsiveContainer>

        {predictions.insights && predictions.insights.length > 0 && (
          <div className="prediction-insights">
            {predictions.insights.map((insight, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: index * 0.1 }}
                className="insight-pill"
              >
                {insight}
              </motion.div>
            ))}
          </div>
        )}
      </motion.div>
    );
  };

  const renderAnomalies = () => {
    if (!anomalies || anomalies.length === 0) return null;

    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.4 }}
        className="anomalies-section"
      >
        <h3>Unusual Transactions Detected</h3>
        <div className="anomalies-list">
          <AnimatePresence>
            {anomalies.map((anomaly, index) => (
              <motion.div
                key={`${anomaly.date}-${index}`}
                initial={{ opacity: 0, x: -50 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 50 }}
                transition={{ duration: 0.3, delay: index * 0.05 }}
                className="anomaly-card"
              >
                <div className="anomaly-header">
                  <span className="anomaly-date">{anomaly.date}</span>
                  <span className="anomaly-score" style={{
                    backgroundColor: anomaly.anomaly_score > 0.8 ? COLORS.danger : COLORS.warning
                  }}>
                    Score: {(anomaly.anomaly_score * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="anomaly-details">
                  <p className="description">{anomaly.description}</p>
                  <p className="amount">{formatCurrency(anomaly.amount)}</p>
                  <p className="reason">{anomaly.reason}</p>
                </div>
                <div className="anomaly-actions">
                  <button className="mark-normal">Mark as Normal</button>
                  <button className="investigate">Investigate</button>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </motion.div>
    );
  };

  const renderSmartInsights = () => {
    if (!insights || insights.length === 0) return null;

    const insightIcons = {
      spending_velocity: '📈',
      category_analysis: '📊',
      recurring_payments: '🔄',
      savings_opportunity: '💰',
      anomaly_pattern: '🎯'
    };

    const insightColors = {
      info: COLORS.info,
      warning: COLORS.warning,
      success: COLORS.secondary,
      danger: COLORS.danger
    };

    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.6 }}
        className="smart-insights-section"
      >
        <h3>Smart Insights</h3>
        <div className="insights-grid">
          {insights.map((insight, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: index * 0.1 }}
              className={`insight-card ${insight.severity}`}
              style={{ borderLeftColor: insightColors[insight.severity] }}
            >
              <div className="insight-header">
                <span className="insight-icon">{insightIcons[insight.type] || '💡'}</span>
                <h4>{insight.title}</h4>
              </div>
              <p className="insight-description">{insight.description}</p>
              {insight.value && (
                <div className="insight-value">
                  {typeof insight.value === 'number' && insight.value > 0 
                    ? formatCurrency(insight.value)
                    : `${insight.value.toFixed(1)}%`}
                </div>
              )}
              {insight.details && (
                <div className="insight-details">
                  {insight.details.map((detail, idx) => (
                    <div key={idx} className="detail-item">
                      <span>{detail.description}</span>
                      <span>{formatCurrency(detail.amount)}</span>
                    </div>
                  ))}
                </div>
              )}
            </motion.div>
          ))}
        </div>
      </motion.div>
    );
  };

  if (loading) {
    return (
      <div className="ml-insights-loading">
        <div className="loading-spinner"></div>
        <p>Analyzing your financial data with AI...</p>
      </div>
    );
  }

  return (
    <div className="ml-insights-container">
      <div className="ml-insights-header">
        <h2>AI-Powered Financial Intelligence</h2>
        <div className="connection-status">
          <span className={`status-dot ${wsConnected ? 'connected' : 'disconnected'}`}></span>
          {wsConnected ? 'Real-time updates active' : 'Connecting...'}
        </div>
      </div>
      
      <div className="ml-insights-content">
        {renderHealthScore()}
        {renderPredictions()}
        {renderAnomalies()}
        {renderSmartInsights()}
      </div>
    </div>
  );
};

export default MLInsights; 