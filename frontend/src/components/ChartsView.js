import React, { useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, PieChart, Pie, Cell, Legend, RadarChart, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, Radar, AreaChart, Area
} from 'recharts';
import './UploadForm.css';

// Utility: Currency formatting
const currencyFormatter = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD'
});
const formatCurrency = (val) => (typeof val === 'number' ? currencyFormatter.format(val) : '$0.00');

// Utility: Shorten long labels
const shortLabel = (label) => {
  if (!label) return '';
  return label.length > 15 ? label.slice(0, 15) + '...' : label;
};

// Color palette for charts
const COLORS = ['#2c3e50', '#34495e', '#446CB3', '#4B77BE', '#1F3A93', '#26A65B', '#1E824C', '#22313F', '#6C7A89'];

/**
 * Aggregate categories, sort by amount, and limit the number shown.
 * Returns a list of categories and values, plus a total of "Other" if needed.
 */
function processCategories(rawData, limit = 8) {
  if (!rawData || !Array.isArray(rawData)) return [];
  
  // Group by main_category and sum amounts
  const categoryMap = rawData.reduce((acc, cat) => {
    const mainCat = cat.main_category || 'Other';
    acc[mainCat] = (acc[mainCat] || 0) + Math.abs(cat.amount);
    return acc;
  }, {});

  // Convert to array and sort
  const categories = Object.entries(categoryMap)
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => b.value - a.value);

  // Handle limit
  if (categories.length > limit) {
    const topCategories = categories.slice(0, limit - 1);
    const otherSum = categories.slice(limit - 1).reduce((sum, cat) => sum + cat.value, 0);
    topCategories.push({ name: 'Other', value: otherSum });
    return topCategories;
  }
  
  return categories;
}

// Add this function to process AI categories
const processAICategories = (aiCategories) => {
  if (!aiCategories || !aiCategories.length) return [];
  
  return aiCategories.map(cat => ({
    name: `${cat.main_category}/${cat.sub_category}`,
    value: Math.abs(cat.amount),
    count: cat.count,
    detail: cat.detail_category
  })).sort((a, b) => b.value - a.value);
};

// Add this section to the ChartsView component
const AICategories = ({ data }) => {
  const aiData = useMemo(() => {
    if (!data || !Array.isArray(data)) {
      console.warn('Invalid AI categories data:', data);
      return [];
    }
    
    return data.map(cat => ({
      name: `${cat.main_category}/${cat.sub_category}`,
      value: Math.abs(parseFloat(cat.amount)),
      count: parseInt(cat.count),
      detail: cat.detail_category
    })).sort((a, b) => b.value - a.value);
  }, [data]);

  console.log('Processed AI data:', aiData); // Debug log

  return (
    <div className="section">
      <h3>AI-Enhanced Categories</h3>
      {aiData.length > 0 ? (
        <>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={aiData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="name" 
                tick={{ fill: '#2c3e50' }}
                angle={-45}
                textAnchor="end"
                height={100}
              />
              <YAxis tick={{ fill: '#2c3e50' }} />
              <Tooltip 
                formatter={(value) => formatCurrency(value)}
                labelFormatter={(label) => `Category: ${label}`}
              />
              <Bar dataKey="value" fill="#2c3e50">
                {aiData.map((entry, index) => (
                  <Cell key={index} fill={COLORS[index % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>

          <div className="ai-categories-table">
            <h4>Detailed Breakdown</h4>
            <table>
              <thead>
                <tr>
                  <th>Category</th>
                  <th>Subcategory</th>
                  <th>Detail</th>
                  <th>Amount</th>
                  <th>Count</th>
                </tr>
              </thead>
              <tbody>
                {data.map((cat, idx) => (
                  <tr key={idx}>
                    <td>{cat.main_category}</td>
                    <td>{cat.sub_category}</td>
                    <td>{cat.detail_category}</td>
                    <td>{formatCurrency(cat.amount)}</td>
                    <td>{cat.count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : (
        <p>No AI categorized data available</p>
      )}
    </div>
  );
};

const DonutChart = ({ data }) => {
  const chartData = useMemo(() => processCategories(data), [data]);
  const total = chartData.reduce((sum, item) => sum + item.value, 0);

  return (
    <div className="section">
      <h3>Where Your Money Goes</h3>
      <div className="donut-chart-container">
        {/* Chart and Table side by side */}
        <div className="chart-wrapper">
          <ResponsiveContainer width="100%" height={400}>
            <PieChart>
              <Pie
                data={chartData}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="50%"
                innerRadius="60%"
                outerRadius="80%"
                paddingAngle={2}
              >
                {chartData.map((entry, index) => (
                  <Cell key={entry.name} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip
                formatter={(value) => formatCurrency(value)}
                labelFormatter={(label) => `Category: ${label}`}
              />
              <Legend
                formatter={(value, entry) => `${value} (${((entry.payload.value / total) * 100).toFixed(1)}%)`}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Centered breakdown table */}
        <div className="category-breakdown" style={{display: 'flex', justifyContent: 'center', width: '100%'}}>
          <table style={{maxWidth: '600px', width: '100%'}}>
            <thead>
              <tr>
                <th style={{textAlign: 'center'}}>Category</th>
                <th style={{textAlign: 'center'}}>Amount</th>
                <th style={{textAlign: 'center'}}>Percentage</th>
              </tr>
            </thead>
            <tbody>
              {chartData.map((cat) => (
                <tr key={cat.name}>
                  <td style={{textAlign: 'center'}}>{cat.name}</td>
                  <td style={{textAlign: 'center'}}>{formatCurrency(cat.value)}</td>
                  <td style={{textAlign: 'center'}}>{((cat.value / total) * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

const ChartsView = ({ summary }) => {
  // Daily expenses
  const dailyData = (summary?.daily_expenses || []).map(d => ({
    date: d.date,
    amount: d.amount
  }));

  // Cumulative spending computation
  const cumulativeData = useMemo(() => {
    if (dailyData.length === 0) return [];
    let cumulative = 0;
    return dailyData.map(item => {
      cumulative += item.amount;
      return { ...item, cumulative };
    });
  }, [dailyData]);

  // Top 5 largest expenses
  const topExpensesData = (summary?.top_5_expenses || []).map(d => ({
    description: d.description,
    amount: d.amount
  }));

  // Recurring transactions
  const recurringData = (summary?.recurring_transactions || []).map(d => ({
    description: d.description,
    count: d.count,
    totalAmount: d.totalamount,
    averageAmount: d.averageamount
  }));

  // Process categories for Donut & Radar charts
  const rawCategories = summary?.category_breakdown || [];
  const processedCategories = useMemo(() => processCategories(rawCategories, 8), [rawCategories]);

  // Radar chart data: same categories as donut
  const radarData = processedCategories.map(d => ({
    category: d.name,
    amount: d.value
  }));
  const multipleCategories = radarData.length > 1;

  return (
    <div className="insights-container">
      {/* FINANCIAL OVERVIEW */}
      <div className="section">
        <h2>Overall Spending Snapshot</h2>
        <p><strong>Total Expenses:</strong> {formatCurrency(summary?.total_expenses)}</p>
        <p><strong>Average Daily Spending:</strong> {formatCurrency(summary?.average_daily)}</p>
        <p><strong>Estimated Monthly Spending:</strong> {formatCurrency(summary?.average_monthly)}</p>
      </div>

      {/* SPENDING OVER TIME (Area Chart) */}
      <div className="section">
        <h3>Daily Spending Trend</h3>
        <p>Track how your spending changes day by day:</p>
        {dailyData.length > 0 ? (
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={dailyData} margin={{ top: 20, right: 30, left: 30, bottom: 20 }}>
              <defs>
                <linearGradient id="colorAmount" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#2c3e50" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#2c3e50" stopOpacity={0.1}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#aaa" />
              <XAxis dataKey="date" tick={{ fill: '#2c3e50' }} />
              <YAxis tick={{ fill: '#2c3e50' }} />
              <Tooltip formatter={(value) => formatCurrency(value)} />
              <Area 
                type="monotone"
                dataKey="amount"
                stroke="#2c3e50"
                fill="url(#colorAmount)"
                fillOpacity={0.7}
              />
            </AreaChart>
          </ResponsiveContainer>
        ) : <p>No daily expense data available.</p>}
      </div>

      {/* CUMULATIVE SPENDING (Line Chart) */}
      <div className="section">
        <h3>Cumulative Spending Over Time</h3>
        <p>See how your total spending accumulates as time passes:</p>
        {cumulativeData.length > 0 ? (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={cumulativeData} margin={{ top: 20, right: 30, left: 30, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#aaa" />
              <XAxis dataKey="date" tick={{ fill: '#2c3e50' }}/>
              <YAxis tick={{ fill: '#2c3e50' }}/>
              <Tooltip formatter={(value) => formatCurrency(value)} />
              <Line 
                type="monotone" 
                dataKey="cumulative" 
                stroke="#4B77BE" 
                strokeWidth={3} 
              />
            </LineChart>
          </ResponsiveContainer>
        ) : <p>No cumulative data available.</p>}
      </div>

      {/* TOP 5 LARGEST EXPENSES (Table) */}
      <div className="section">
        <h3>Top 5 Biggest Expenses</h3>
        <p>These are your top individual expenses that made the biggest dent:</p>
        {topExpensesData.length > 0 ? (
          <table className="top-expenses-table">
            <thead>
              <tr>
                <th>Description</th>
                <th>Total Amount</th>
              </tr>
            </thead>
            <tbody>
              {topExpensesData.map((item, i) => (
                <tr key={i}>
                  <td>{item.description}</td>
                  <td>{formatCurrency(item.amount)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : <p>No large expenses found.</p>}
      </div>

      {/* SPENDING DISTRIBUTION (Radar Chart) */}
      <div className="section">
        <h3>Category Distribution (Radar Chart)</h3>
        <p>This chart compares your top spending categories to each other:</p>
        {multipleCategories ? (
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={radarData} margin={{ top: 40, right: 40, bottom: 40, left: 40 }}>
              <PolarGrid gridType="circle" stroke="#ccc" strokeDasharray="3 3"/>
              <PolarAngleAxis 
                dataKey="category" 
                tick={{ fill: '#2C3E50', fontSize: 12 }}
              />
              <PolarRadiusAxis 
                angle={30} 
                domain={[0, 'auto']} 
                stroke="#2C3E50" 
                tick={{ fill: '#2C3E50', fontSize: 10 }}
              />
              <Radar
                name="Spending"
                dataKey="amount"
                stroke="#2C3E50"
                fill="#2C3E50"
                fillOpacity={0.5}
              />
              <Tooltip formatter={(value) => formatCurrency(value)} />
            </RadarChart>
          </ResponsiveContainer>
        ) : (
          <p>Not enough category variety for a radar chart. As you diversify your spending, more meaningful insights will appear here.</p>
        )}
      </div>

      {/* MOST RECURRING TRANSACTIONS (Bar Chart) */}
      <div className="section recurring-section">
        <h3>Most Frequent Transactions</h3>
        <p>These are the transactions that occur most often:</p>
        {recurringData.length > 0 ? (
          <div className="recurring-chart-container">
            <ResponsiveContainer width="100%" height={400}>
              <BarChart 
                data={recurringData} 
                margin={{ top: 20, right: 30, left: 60, bottom: 100 }}
                barSize={40}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(187, 10, 30, 0.2)" />
                <XAxis 
                  dataKey="description" 
                  interval={0}
                  tickFormatter={shortLabel}
                  height={100}
                  tick={{ 
                    fill: 'var(--military-gray-dark)', 
                    fontSize: 11,
                    width: 80,
                    fontFamily: 'Share Tech Mono'
                  }}
                  tickLine={false}
                  axisLine={{ stroke: 'var(--military-red)', strokeWidth: 1 }}
                />
                <YAxis 
                  tick={{ 
                    fill: 'var(--military-gray-dark)',
                    fontFamily: 'Share Tech Mono',
                    fontSize: 11
                  }}
                  tickFormatter={(value) => formatCurrency(value)}
                  axisLine={{ stroke: 'var(--military-red)', strokeWidth: 1 }}
                />
                <Tooltip 
                  formatter={(value) => formatCurrency(value)}
                  contentStyle={{
                    background: 'var(--military-gray-light)',
                    border: '1px solid var(--military-red)',
                    borderRadius: 0,
                    fontFamily: 'Share Tech Mono'
                  }}
                />
                <Legend 
                  verticalAlign="top"
                  height={36}
                  wrapperStyle={{
                    fontFamily: 'Share Tech Mono',
                    textTransform: 'uppercase'
                  }}
                />
                <Bar 
                  dataKey="totalAmount" 
                  name="Total Amount" 
                  fill="var(--military-red)"
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : <p>No recurring transactions found.</p>}
      </div>

      {/* Add AI Categories section */}
      {summary.ai_categories && summary.ai_categories.length > 0 && (
        <DonutChart data={summary.ai_categories} />
      )}

      {/* Budget Alerts Section */}
      {summary.budget_alerts && summary.budget_alerts.length > 0 && (
        <div className="budget-alerts-section">
          <h3>⚠️ Budget Alerts</h3>
          <div className="budget-alerts-container">
            {summary.budget_alerts.map((alert, index) => (
              <div key={index} className="budget-alert-card">
                <div className="alert-header">
                  <span className="category-name">{alert.category}</span>
                  <span className="percentage-badge">{alert.percentage.toFixed(0)}%</span>
                </div>
                <div className="alert-details">
                  <div className="amount-bar">
                    <div 
                      className="amount-spent" 
                      style={{ width: `${Math.min(alert.percentage, 100)}%` }}
                    />
                  </div>
                  <div className="amount-text">
                    <span>Spent: ${alert.spent.toFixed(2)}</span>
                    <span>Limit: ${alert.limit.toFixed(2)}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Export Section */}
      <div className="export-section">
        <button 
          className="export-btn"
          onClick={() => handleExport('csv')}
        >
          Export as CSV
        </button>
        <button 
          className="export-btn"
          onClick={() => handleExport('json')}
        >
          Export as JSON
        </button>
      </div>

      {/* Add more statistics */}
      <div className="statistics-grid">
        <div className="stat-card">
          <h4>Median Transaction</h4>
          <p className="stat-value">${summary.statistics?.median_transaction?.toFixed(2) || '0.00'}</p>
        </div>
        <div className="stat-card">
          <h4>Standard Deviation</h4>
          <p className="stat-value">${summary.statistics?.std_deviation?.toFixed(2) || '0.00'}</p>
        </div>
        <div className="stat-card">
          <h4>Min Transaction</h4>
          <p className="stat-value">${summary.statistics?.min_transaction?.toFixed(2) || '0.00'}</p>
        </div>
        <div className="stat-card">
          <h4>Max Transaction</h4>
          <p className="stat-value">${summary.statistics?.max_transaction?.toFixed(2) || '0.00'}</p>
        </div>
      </div>
    </div>
  );
};

// Add export handler
const handleExport = async (format) => {
  try {
    // Get current file ID from props or state
    const fileId = summary.file_id || 1; // You'll need to pass this from parent
    
    const response = await fetch(`http://localhost:8000/export/${fileId}?format=${format}`);
    if (!response.ok) throw new Error('Export failed');
    
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `financial_export.${format}`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  } catch (error) {
    console.error('Export error:', error);
  }
};

export default ChartsView;