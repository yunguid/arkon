import React, { useState, useEffect } from 'react';
import './StockNews.css';

function StockNews({ symbol }) {
    const [news, setNews] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [status, setStatus] = useState('');
    const [isCollecting, setIsCollecting] = useState(false);

    const fetchNews = async () => {
        try {
            setStatus('Fetching news data...');
            const response = await fetch(`http://localhost:8000/stock/${symbol}/news`);
            if (!response.ok) throw new Error('Failed to fetch news');
            const data = await response.json();
            console.log('Fetched News Data:', data);
            console.log('News Array:', data.news);
            setNews(data.news);
            setError(null);
            setStatus('');
        } catch (err) {
            console.error('Fetch Error:', err);
            setError(err.message);
            setStatus('Error fetching news');
        } finally {
            setLoading(false);
        }
    };

    const triggerNewsCollection = async () => {
        try {
            setIsCollecting(true);
            setStatus('Initiating news collection...');
            
            const response = await fetch(`http://localhost:8000/stock/${symbol}/collect_news`, {
                method: 'POST'
            });
            
            const data = await response.json();
            console.log('Collection Response:', data);
            
            if (!response.ok) {
                throw new Error(data.detail || 'Failed to trigger news collection');
            }
            
            setStatus('News collection completed. Fetching results...');
            await fetchNews();
            
        } catch (err) {
            console.error('Collection Error:', err);
            setError(`Collection failed: ${err.message}`);
            setStatus('Error during news collection. Please try again.');
        } finally {
            setIsCollecting(false);
        }
    };

    useEffect(() => {
        console.log('Symbol changed:', symbol);
        if (symbol) {
            fetchNews();
        }
    }, [symbol]);

    console.log('Current State:', {
        news,
        loading,
        error,
        status,
        isCollecting
    });

    return (
        <div className="stock-news">
            <div className="news-header">
                <h3>Latest News Analysis</h3>
                <button 
                    onClick={triggerNewsCollection}
                    disabled={isCollecting}
                    className="collect-news-btn"
                >
                    {isCollecting ? 'Collecting...' : 'Analyze Latest News'}
                </button>
            </div>
            
            {status && <div className="status-message">{status}</div>}
            {error && <div className="error-message">{error}</div>}
            
            {loading ? (
                <div className="loading">Loading news analysis...</div>
            ) : !news || news.length === 0 ? (
                <div className="no-news">
                    <p>No news analysis available.</p>
                    <p>Click the button above to analyze latest news.</p>
                </div>
            ) : (
                <div>
                    {news.map((item, index) => (
                        <div key={index} className="news-item">
                            <div className="news-header">
                                <div className="news-date">
                                    {new Date(item.date).toLocaleDateString()}
                                </div>
                                <div className="sentiment">
                                    Sentiment: {item.sentiment > 0 ? 'Positive' : 
                                              item.sentiment < 0 ? 'Negative' : 'Neutral'}
                                </div>
                            </div>
                            <div className="news-content">
                                <p className="summary">
                                    {item.summary?.summary || 
                                     item.summary?.key_developments || 
                                     "No analysis available"}
                                </p>
                                <div className="key-points">
                                    {item.summary?.risks?.length > 0 && (
                                        <div className="risks">
                                            <strong>Risks:</strong>
                                            <ul>
                                                {item.summary.risks.map((risk, i) => (
                                                    <li key={i}>{risk}</li>
                                                ))}
                                            </ul>
                                        </div>
                                    )}
                                    {item.summary?.price_impact && (
                                        <div className="impact">
                                            <strong>Impact:</strong> {item.summary.price_impact}
                                        </div>
                                    )}
                                </div>
                                <div className="sources">
                                    <div className="source-tooltip-container">
                                        <small>Sources: {item.sources?.length || 0} articles analyzed</small>
                                        {item.sources?.length > 0 && (
                                            <div className="source-tooltip">
                                                <ul>
                                                    {item.sources.map((url, i) => (
                                                        <li key={i}>
                                                            <a href={url} target="_blank" rel="noopener noreferrer">
                                                                {new URL(url).hostname.replace('www.', '')}
                                                            </a>
                                                        </li>
                                                    ))}
                                                </ul>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}

export default StockNews; 