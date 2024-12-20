import React, { useState, useEffect } from 'react';
import './Watchlist.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function Watchlist({ onSelectStock, onAnalyzeAll, isAnalyzing }) {
    const [watchlist, setWatchlist] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [expandedItem, setExpandedItem] = useState(null);

    const fetchWatchlist = async () => {
        try {
            const response = await fetch(`${API_URL}/watchlist`);
            if (!response.ok) throw new Error('Failed to fetch watchlist');
            const data = await response.json();
            setWatchlist(data.watchlist || []);
        } catch (err) {
            console.error('Watchlist Error:', err);
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const removeFromWatchlist = async (symbol) => {
        try {
            const response = await fetch(`${API_URL}/watchlist/${symbol}`, {
                method: 'DELETE'
            });
            if (!response.ok) throw new Error('Failed to remove from watchlist');
            await fetchWatchlist();
        } catch (err) {
            console.error('Remove Error:', err);
            setError(err.message);
        }
    };

    useEffect(() => {
        fetchWatchlist();
    }, []);

    return (
        <div className="watchlist">
            <h3>Watchlist</h3>
            {loading ? (
                <div>Loading...</div>
            ) : error ? (
                <div className="error">{error}</div>
            ) : (
                <ul className="watchlist-items">
                    {watchlist.map(item => (
                        <li 
                            key={item.symbol} 
                            className={expandedItem === item.symbol ? 'expanded' : ''}
                        >
                            <div 
                                className="stock-header"
                                onClick={() => setExpandedItem(
                                    expandedItem === item.symbol ? null : item.symbol
                                )}
                            >
                                <span className="symbol">{item.symbol}</span>
                                <span className="company-name">{item.company_name}</span>
                            </div>
                            
                            {expandedItem === item.symbol && (
                                <div className="stock-details">
                                    <div className="sector">{item.sector}</div>
                                    <div className="actions">
                                        <button 
                                            onClick={() => onSelectStock(item.symbol)}
                                            className="research-btn"
                                        >
                                            Research
                                        </button>
                                        <button 
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                removeFromWatchlist(item.symbol);
                                            }}
                                            className="remove-btn"
                                        >
                                            Remove
                                        </button>
                                    </div>
                                    <div className="status-indicators">
                                        {item.has_filings && (
                                            <span className="has-filings">SEC Filings Available</span>
                                        )}
                                        {item.metrics_available && (
                                            <span className="has-metrics">Metrics Available</span>
                                        )}
                                    </div>
                                </div>
                            )}
                        </li>
                    ))}
                </ul>
            )}
        </div>
    );
}

export default Watchlist; 