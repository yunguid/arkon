import React, { useState, useEffect } from 'react';
import './Watchlist.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function Watchlist({ onSelectStock, onAnalyzeAll, isAnalyzing }) {
    const [watchlist, setWatchlist] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

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
            <h3>
                Watchlist
                <button 
                    className="analyze-all-btn"
                    onClick={onAnalyzeAll}
                    disabled={isAnalyzing || watchlist.length === 0}
                >
                    {isAnalyzing ? 'Analyzing...' : 'Analyze All'}
                </button>
            </h3>
            {loading ? (
                <div>Loading...</div>
            ) : error ? (
                <div className="error">{error}</div>
            ) : (
                <ul className="watchlist-items">
                    {watchlist.map(item => (
                        <li key={item.symbol} onClick={() => onSelectStock(item.symbol)}>
                            <span className="symbol">{item.symbol}</span>
                            <span className="last-analysis">
                                {item.last_analysis ? new Date(item.last_analysis).toLocaleDateString() : 'Never'}
                            </span>
                            <button onClick={(e) => {
                                e.stopPropagation();
                                removeFromWatchlist(item.symbol);
                            }}>×</button>
                        </li>
                    ))}
                </ul>
            )}
        </div>
    );
}

export default Watchlist; 