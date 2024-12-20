import React, { useState, useEffect, useCallback } from 'react';
import './ResearchView.css';
import StockFinancials from './StockFinancials';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function ResearchView({ symbol }) {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [stockData, setStockData] = useState(null);
    const [activeTab, setActiveTab] = useState('overview');

    const fetchStockData = useCallback(async () => {
        setLoading(true);
        try {
            const response = await fetch(`${API_URL}/stock/${symbol}/research`);
            if (!response.ok) throw new Error('Failed to fetch stock data');
            const data = await response.json();
            setStockData(data);
            setError(null);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    }, [symbol]);

    const fetchSECFilings = async () => {
        setLoading(true);
        try {
            const response = await fetch(`${API_URL}/stock/${symbol}/fetch-latest-10k`, {
                method: 'POST'
            });
            if (!response.ok) throw new Error('Failed to fetch SEC filings');
            const data = await response.json();
            console.log('SEC Filing Response:', data);  // Debug log
            await fetchStockData(); // Refresh data after fetching filings
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const fetchMetrics = async () => {
        try {
            const response = await fetch(`${API_URL}/stock/${symbol}/metrics`);
            if (!response.ok) throw new Error('Failed to fetch metrics');
            const data = await response.json();
            setStockData(prev => ({
                ...prev,
                metrics: data
            }));
        } catch (err) {
            setError(err.message);
        }
    };

    const fetchLatest10K = async () => {
        setLoading(true);
        try {
            const response = await fetch(`${API_URL}/stock/${symbol}/fetch-latest-10k`, {
                method: 'POST'
            });
            if (!response.ok) throw new Error('Failed to fetch latest 10-K');
            const data = await response.json();
            await fetchStockData(); // Refresh data after fetching filing
            return data;
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchStockData();
    }, [symbol, fetchStockData]);

    if (loading) return <div className="loading">Loading research data...</div>;
    if (error) return <div className="error">{error}</div>;
    if (!stockData) return null;

    return (
        <div className="research-view">
            <header className="research-header">
                <h2>{symbol} Research Dashboard</h2>
                <div className="company-info">
                    <h3>{stockData.company_name}</h3>
                    <span className="sector">{stockData.sector}</span>
                </div>
            </header>

            <nav className="research-tabs">
                <button 
                    className={`tab ${activeTab === 'overview' ? 'active' : ''}`}
                    onClick={() => setActiveTab('overview')}
                >
                    Overview
                </button>
                <button 
                    className={`tab ${activeTab === 'filings' ? 'active' : ''}`}
                    onClick={() => setActiveTab('filings')}
                >
                    SEC Filings
                </button>
                <button 
                    className={`tab ${activeTab === 'metrics' ? 'active' : ''}`}
                    onClick={() => setActiveTab('metrics')}
                >
                    Financial Metrics
                </button>
            </nav>

            <div className="tab-content">
                {activeTab === 'filings' && (
                    <div className="filings-section">
                        <button 
                            className="fetch-filings-btn"
                            onClick={fetchSECFilings}
                            disabled={loading}
                        >
                            {loading ? 'Fetching...' : 'Fetch Latest 10-K Filing'}
                        </button>
                        {stockData.filings && stockData.filings.length > 0 ? (
                            <ul className="filings-list">
                                {stockData.filings.map(filing => (
                                    <li key={filing.url} className="filing-item">
                                        <span className="filing-type">{filing.type}</span>
                                        <span className="filing-date">
                                            {new Date(filing.date).toLocaleDateString()}
                                        </span>
                                        <a 
                                            href={filing.url} 
                                            target="_blank" 
                                            rel="noopener noreferrer"
                                            className="filing-link"
                                        >
                                            View Filing
                                        </a>
                                    </li>
                                ))}
                            </ul>
                        ) : (
                            <p>No SEC filings available. Click the button above to fetch them.</p>
                        )}
                    </div>
                )}
                
                {activeTab === 'overview' && (
                    <div className="overview-section">
                        <div className="overview-grid">
                            <div className="metric-card">
                                <h4>Market Cap</h4>
                                <span className="value">
                                    ${(stockData.metrics?.market_cap / 1e9).toFixed(2)}B
                                </span>
                            </div>
                            <div className="metric-card">
                                <h4>P/E Ratio</h4>
                                <span className="value">
                                    {stockData.metrics?.pe_ratio?.toFixed(2) || 'N/A'}
                                </span>
                            </div>
                            <div className="metric-card">
                                <h4>Forward P/E</h4>
                                <span className="value">
                                    {stockData.metrics?.forward_pe?.toFixed(2) || 'N/A'}
                                </span>
                            </div>
                            <div className="metric-card">
                                <h4>Dividend Yield</h4>
                                <span className="value">
                                    {stockData.metrics?.dividend_yield 
                                        ? `${(stockData.metrics.dividend_yield * 100).toFixed(2)}%` 
                                        : 'N/A'}
                                </span>
                            </div>
                            <div className="metric-card">
                                <h4>Beta</h4>
                                <span className="value">
                                    {stockData.metrics?.beta?.toFixed(2) || 'N/A'}
                                </span>
                            </div>
                            <div className="metric-card">
                                <h4>52-Week Range</h4>
                                <span className="value">
                                    ${stockData.metrics?.['52_week_low']?.toFixed(2)} - 
                                    ${stockData.metrics?.['52_week_high']?.toFixed(2)}
                                </span>
                            </div>
                        </div>
                        <button 
                            className="refresh-metrics-btn"
                            onClick={fetchMetrics}
                        >
                            Refresh Metrics
                        </button>
                    </div>
                )}
                
                {activeTab === 'metrics' && (
                    <StockFinancials 
                        metrics={stockData.metrics} 
                        onRefresh={async () => {
                            try {
                                const response = await fetch(`${API_URL}/stock/${symbol}/metrics`);
                                if (!response.ok) throw new Error('Failed to fetch metrics');
                                const data = await response.json();
                                setStockData(prev => ({
                                    ...prev,
                                    metrics: data
                                }));
                            } catch (err) {
                                setError(err.message);
                            }
                        }}
                    />
                )}
            </div>
        </div>
    );
}

export default ResearchView; 