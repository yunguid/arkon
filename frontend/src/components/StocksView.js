import React, { useState, useEffect, useRef } from 'react';
import { ResponsiveContainer, AreaChart, Area, CartesianGrid, XAxis, YAxis, Tooltip, BarChart, Bar, Legend } from 'recharts';
import './StocksView.css';
import StockNews from './StockNews';

function StocksView() {
  const [ticker, setTicker] = useState('');
  const [livePrice, setLivePrice] = useState(null);
  const [error, setError] = useState(null);
  const [intervalId, setIntervalId] = useState(null);
  const [stockDataSummary, setStockDataSummary] = useState(null);
  const [loadingUpload, setLoadingUpload] = useState(false);
  const fileInputRef = useRef(null);

  const fetchPrice = async () => {
    if (!ticker) return;
    try {
      const res = await fetch(`http://localhost:8000/stock_price?symbol=${encodeURIComponent(ticker)}`);
      if (!res.ok) throw new Error('Failed to fetch price');
      const data = await res.json();
      setLivePrice(data.price);
      setError(null);
    } catch (err) {
      setError(err.message);
    }
  };

  const startUpdates = () => {
    if (intervalId) clearInterval(intervalId);
    const newId = setInterval(fetchPrice, 120000);
    setIntervalId(newId);
  };

  const stopUpdates = () => {
    if (intervalId) {
      clearInterval(intervalId);
      setIntervalId(null);
    }
  };

  useEffect(() => {
    return () => stopUpdates();
  }, []);

  const handleUpload = async (e) => {
    e.preventDefault();
    const file = fileInputRef.current?.files[0];
    if (!file) return;

    setLoadingUpload(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('http://localhost:8000/upload_stock_data', {
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

  return (
    <div className="stocks-view">
      {/* Stock input section */}
      <div className="stock-input-section">
        <h2>Stock Performance</h2>
        <div className="input-group">
          <input 
            type="text" 
            placeholder="Enter ticker (e.g. AAPL)" 
            value={ticker}
            onChange={(e) => setTicker(e.target.value.toUpperCase())}
          />
          <button onClick={fetchPrice}>Get Price</button>
          <button onClick={startUpdates} disabled={!ticker}>Auto-Refresh</button>
          <button onClick={stopUpdates}>Stop Refresh</button>
        </div>
        {error && <div className="error-message">{error}</div>}
        {livePrice && (
          <div className="live-price">
            {ticker}: ${livePrice.toFixed(2)}
          </div>
        )}
        {ticker && <StockNews symbol={ticker} />}
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