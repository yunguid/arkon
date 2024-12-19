import React, { useState, useEffect } from 'react';
import './StockAnalysisHistory.css';

function StockAnalysisHistory({ symbol }) {
    const [history, setHistory] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [expandedItems, setExpandedItems] = useState(new Set());

    const toggleItem = (index) => {
        setExpandedItems(prev => {
            const newSet = new Set(prev);
            if (newSet.has(index)) {
                newSet.delete(index);
            } else {
                newSet.add(index);
            }
            return newSet;
        });
    };

    const fetchHistory = async () => {
        try {
            const response = await fetch(`http://localhost:8000/stock/${symbol}/analysis_history`);
            if (!response.ok) throw new Error('Failed to fetch history');
            const data = await response.json();
            setHistory(data.history);
            setError(null);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        if (symbol) {
            fetchHistory();
        }
    }, [symbol]);

    if (loading) return <div>Loading history...</div>;
    if (error) return <div className="error">{error}</div>;
    if (!history.length) return <div>No analysis history available</div>;

    return (
        <div className="analysis-history">
            <h3>Analysis History</h3>
            <div className="history-list">
                {history.map((item, index) => (
                    <div key={index} className={`history-item ${expandedItems.has(index) ? 'expanded' : ''}`}>
                        <div className="history-header" onClick={() => toggleItem(index)}>
                            <div className="header-main">
                                <span className="date">
                                    {new Date(item.date).toLocaleDateString()}
                                </span>
                                <span className={`sentiment ${
                                    item.sentiment > 0 ? 'positive' : 
                                    item.sentiment < 0 ? 'negative' : 'neutral'
                                }`}>
                                    {item.sentiment > 0 ? 'Positive' : 
                                     item.sentiment < 0 ? 'Negative' : 'Neutral'}
                                </span>
                            </div>
                            <span className="expand-icon">
                                {expandedItems.has(index) ? '▼' : '▶'}
                            </span>
                        </div>
                        {expandedItems.has(index) && (
                            <div className="history-content">
                                <p className="summary">{item.summary.summary}</p>
                                <div className="key-points">
                                    <strong>Key Developments:</strong>
                                    <p>{item.summary.key_developments}</p>
                                    
                                    <strong>Price Impact:</strong>
                                    <p>{item.summary.price_impact}</p>
                                    
                                    {item.summary.risks.length > 0 && (
                                        <div className="risks">
                                            <strong>Risks:</strong>
                                            <ul>
                                                {item.summary.risks.map((risk, i) => (
                                                    <li key={i}>{risk}</li>
                                                ))}
                                            </ul>
                                        </div>
                                    )}
                                </div>
                                <div className="sources">
                                    <small>Sources: {item.sources.length} articles analyzed</small>
                                </div>
                            </div>
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
}

export default StockAnalysisHistory; 