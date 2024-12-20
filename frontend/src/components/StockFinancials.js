import React from 'react';
import './StockFinancials.css';
import { ResponsiveContainer, LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip } from 'recharts';

function StockFinancials({ metrics, onRefresh }) {
    const formatLargeNumber = (num) => {
        if (!num) return 'N/A';
        if (num >= 1e9) return `$${(num / 1e9).toFixed(2)}B`;
        if (num >= 1e6) return `$${(num / 1e6).toFixed(2)}M`;
        return `$${num.toFixed(2)}`;
    };

    const formatPercentage = (num) => {
        if (!num) return 'N/A';
        return `${(num * 100).toFixed(2)}%`;
    };

    return (
        <div className="financials-section">
            <div className="metrics-header">
                <h3>Financial Metrics</h3>
                <button onClick={onRefresh} className="refresh-btn">
                    Refresh Data
                </button>
            </div>

            <div className="metrics-grid">
                <div className="metric-group">
                    <h4>Valuation Metrics</h4>
                    <div className="metric-cards">
                        <div className="metric-card">
                            <span className="label">Market Cap</span>
                            <span className="value">{formatLargeNumber(metrics?.market_cap)}</span>
                        </div>
                        <div className="metric-card">
                            <span className="label">Enterprise Value</span>
                            <span className="value">{formatLargeNumber(metrics?.enterprise_value)}</span>
                        </div>
                        <div className="metric-card">
                            <span className="label">P/E Ratio</span>
                            <span className="value">{metrics?.pe_ratio?.toFixed(2) || 'N/A'}</span>
                        </div>
                    </div>
                </div>

                <div className="metric-group">
                    <h4>Profitability</h4>
                    <div className="metric-cards">
                        <div className="metric-card">
                            <span className="label">Profit Margin</span>
                            <span className="value">{formatPercentage(metrics?.profit_margins)}</span>
                        </div>
                        <div className="metric-card">
                            <span className="label">Operating Margin</span>
                            <span className="value">{formatPercentage(metrics?.operating_margins)}</span>
                        </div>
                        <div className="metric-card">
                            <span className="label">ROE</span>
                            <span className="value">{formatPercentage(metrics?.return_on_equity)}</span>
                        </div>
                    </div>
                </div>

                <div className="metric-group">
                    <h4>Financial Health</h4>
                    <div className="metric-cards">
                        <div className="metric-card">
                            <span className="label">Current Ratio</span>
                            <span className="value">{metrics?.current_ratio?.toFixed(2) || 'N/A'}</span>
                        </div>
                        <div className="metric-card">
                            <span className="label">Debt/Equity</span>
                            <span className="value">{metrics?.debt_to_equity?.toFixed(2) || 'N/A'}</span>
                        </div>
                        <div className="metric-card">
                            <span className="label">Quick Ratio</span>
                            <span className="value">{metrics?.quick_ratio?.toFixed(2) || 'N/A'}</span>
                        </div>
                    </div>
                </div>
            </div>

            {metrics?.historical_prices && (
                <div className="price-chart">
                    <h4>Historical Price Performance</h4>
                    <ResponsiveContainer width="100%" height={300}>
                        <LineChart data={metrics.historical_prices}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="date" />
                            <YAxis />
                            <Tooltip />
                            <Line 
                                type="monotone" 
                                dataKey="close" 
                                stroke="#3498db" 
                                dot={false}
                            />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            )}
        </div>
    );
}

export default StockFinancials; 