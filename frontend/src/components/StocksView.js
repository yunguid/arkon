import React, { useState, useEffect, useCallback, useRef } from 'react';
import { ResponsiveContainer, AreaChart, Area, CartesianGrid, XAxis, YAxis, Tooltip } from 'recharts';
import './StocksView.css';
import StockNews from './StockNews';
import Watchlist from './Watchlist';
import StockAnalysisHistory from './StockAnalysisHistory';
import ResearchView from './ResearchView';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function StocksView() {
  const [ticker, setTicker] = useState('');
  const [livePrice, setLivePrice] = useState(null);
  const [error, setError] = useState(null);
  const [stockDataSummary, setStockDataSummary] = useState(null);
  const [loadingUpload, setLoadingUpload] = useState(false);
  const fileInputRef = useRef(null);
  const [isAnalyzingAll, setIsAnalyzingAll] = useState(false);
  const [watchlistKey, setWatchlistKey] = useState(0);
  const [selectedStock, setSelectedStock] = useState(null);

  const fetchPrice = async () => {
    if (!ticker) return;
    try {
      const response = await fetch(`${API_URL}/stock/${ticker}/price`);
      if (!response.ok) throw new Error('Failed to fetch price');
      const data = await response.json();
      setLivePrice(data.price);
      setError(null);
    } catch (err) {
      setError(err.message);
    }
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    const file = fileInputRef.current?.files[0];
    if (!file) return;

    setLoadingUpload(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_URL}/upload_stock_data`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Upload failed');
      setStockDataSummary(data.summary);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingUpload(false);
    }
  };

  const handleSelectStock = (symbol) => {
    setTicker(symbol);
    setSelectedStock(symbol);
  };

  const handleAnalyzeAll = async () => {
    setIsAnalyzingAll(true);
    try {
      const response = await fetch(`${API_URL}/watchlist/analyze`, {
        method: 'POST'
      });
      if (!response.ok) throw new Error('Failed to analyze watchlist');
      const data = await response.json();
      if (data.status === 'complete') {
        setError(null);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setIsAnalyzingAll(false);
    }
  };

  const addToWatchlist = async () => {
    if (!ticker) return;
    try {
      const response = await fetch(`${API_URL}/watchlist/${ticker}`, {
        method: 'POST'
      });
      if (!response.ok) throw new Error('Failed to add to watchlist');
      setError(null);
      setWatchlistKey(prev => prev + 1);
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div className="stocks-view">
      <div className="stocks-layout">
        <aside className="stocks-sidebar">
          <Watchlist 
            onSelectStock={handleSelectStock}
            onAnalyzeAll={handleAnalyzeAll}
            isAnalyzing={isAnalyzingAll}
            key={watchlistKey}
          />
        </aside>
        
        <main className="stocks-main">
          {/* Stock input section */}
          <div className="stock-input-section">
            <h2>STOCK PERFORMANCE</h2>
            <div className="input-group">
              <input 
                type="text" 
                placeholder="Enter ticker (e.g. AAPL)" 
                value={ticker}
                onChange={(e) => setTicker(e.target.value.toUpperCase())}
              />
              <button onClick={fetchPrice}>Get Price</button>
              <button onClick={addToWatchlist}>Add to Watchlist</button>
            </div>
            {error && <div className="error-message">{error}</div>}
            {livePrice && (
              <div className="live-price">
                {ticker}: ${livePrice.toFixed(2)}
              </div>
            )}
            {ticker && !selectedStock && (
              <>
                <StockNews symbol={ticker} />
                <StockAnalysisHistory symbol={ticker} />
              </>
            )}
            {selectedStock && (
              <ResearchView symbol={selectedStock} />
            )}
          </div>
        </main>
      </div>

      {/* Upload section */}
      <div className="upload-section">
        <h3>Upload Stock Data</h3>
        <form onSubmit={handleUpload}>
          <input type="file" accept=".csv" ref={fileInputRef} />
          <button type="submit" disabled={loadingUpload}>
            {loadingUpload ? 'Processing...' : 'Upload & Analyze'}
          </button>
        </form>
      </div>

      {/* Charts section */}
      {stockDataSummary && (
        <div className="charts-section">
          {stockDataSummary.daily_prices?.length > 0 && (
            <div className="chart-container">
              <h4>Price History</h4>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={stockDataSummary.daily_prices}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <Area type="monotone" dataKey="price" stroke="#2c3e50" fill="#3498db" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default StocksView; 