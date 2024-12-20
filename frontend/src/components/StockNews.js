import React, { useState } from 'react';
import './StockNews.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function StockNews({ symbol }) {
    const [news, setNews] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);

    const analyzeLatestNews = async () => {
        try {
            setIsAnalyzing(true);
            setLoading(true);
            
            const response = await fetch(`${API_URL}/stock/${symbol}/collect_news`, {
                method: 'POST'
            });
            
            if (!response.ok) {
                throw new Error('Failed to analyze news');
            }
            
            const data = await response.json();
            setNews(data.news);
            
        } catch (err) {
            setError(err.message);
        } finally {
            setIsAnalyzing(false);
            setLoading(false);
        }
    };

    return (
        <div className="stock-news">
            <div className="news-header">
                <h3>Latest News Analysis</h3>
                <button 
                    onClick={analyzeLatestNews}
                    disabled={isAnalyzing}
                    className="collect-news-btn"
                >
                    {isAnalyzing ? 'Analyzing...' : 'Analyze Latest News'}
                </button>
            </div>
            
            {error && <div className="error-message">{error}</div>}
            
            {loading ? (
                <div className="loading">Analyzing latest news...</div>
            ) : !news ? (
                <div className="no-news">
                    <p>Click above to analyze latest news</p>
                </div>
            ) : (
                <div className="news-item">
                    <div className="news-content">
                        <p>{news.summary.summary}</p>
                        <div className="key-points">
                            <strong>Key Developments:</strong>
                            <p>{news.summary.key_developments}</p>
                            <strong>Price Impact:</strong>
                            <p>{news.summary.price_impact}</p>
                            {news.summary.risks && news.summary.risks.length > 0 && (
                                <div>
                                    <strong>Risks:</strong>
                                    <ul>
                                        {news.summary.risks.map((risk, i) => (
                                            <li key={i}>{risk}</li>
                                        ))}
                                    </ul>
                                </div>
                            )}
                        </div>
                        {news.source_urls && news.source_urls.length > 0 && (
                            <div className="source-tooltip-container">
                                <div className="sources">
                                    <small>Sources: {news.source_urls.length} articles analyzed</small>
                                </div>
                                <div className="source-tooltip">
                                    <ul>
                                        {news.source_urls.map((url, i) => (
                                            <li key={i}>
                                                <a href={url} target="_blank" rel="noopener noreferrer">
                                                    Source {i + 1}
                                                </a>
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}

export default StockNews; 