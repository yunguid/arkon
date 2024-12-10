import React, { useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, PieChart, Pie, Cell, Legend, RadarChart, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, Radar, AreaChart, Area
} from 'recharts';
import './UploadForm.css';

const currencyFormatter = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD'
});
const formatCurrency = (val) => (typeof val === 'number' ? currencyFormatter.format(val) : '$0.00');
const shortLabel = (label) => (!label ? '' : label.length > 15 ? label.slice(0, 15) + '...' : label);

const ChartsView = ({ summary }) => {
  const COLORS = ['#2c3e50', '#34495e', '#446CB3', '#4B77BE', '#1F3A93', '#26A65B', '#1E824C', '#22313F', '#6C7A89'];

  const categoryData = (summary?.category_breakdown || []).map(d => ({
    name: d.category,
    value: d.amount
  }));

  const dailyData = (summary?.daily_expenses || []).map(d => ({
    date: d.date,
    amount: d.amount
  }));

  const cumulativeData = useMemo(() => {
    if (!dailyData || dailyData.length === 0) return [];
    let cumulative = 0;
    return dailyData.map(item => {
      cumulative += item.amount;
      return { ...item, cumulative };
    });
  }, [dailyData]);

  const topExpensesData = summary?.top_5_expenses?.map(d => ({
    description: d.description,
    amount: d.amount
  })) || [];

  const recurringData = summary?.recurring_transactions?.map(d => ({
    description: d.description,
    count: d.count,
    totalAmount: d.totalamount,
    averageAmount: d.averageamount
  })) || [];

  const radarData = categoryData.map(d => ({
    category: d.name,
    amount: d.value
  }));
  const multipleCategories = radarData.length > 1;

  return (
    <div className="insights-container">
      <div className="section">
        <h2>Financial Overview</h2>
        <p><strong>Total Expenses:</strong> {formatCurrency(summary.total_expenses)}</p>
        <p><strong>Average Daily Spending:</strong> {formatCurrency(summary.average_daily)}</p>
        <p><strong>Estimated Monthly Spending:</strong> {formatCurrency(summary.average_monthly)}</p>
      </div>

      <div className="section">
        <h3>Spending Over Time (Area Chart)</h3>
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
              <XAxis dataKey="date" tick={{ fill: '#2c3e50' }}/>
              <YAxis tick={{ fill: '#2c3e50' }}/>
              <Tooltip formatter={(value) => formatCurrency(value)} />
              <Area type="monotone" dataKey="amount" stroke="#2c3e50" fill="url(#colorAmount)" fillOpacity={0.7} />
            </AreaChart>
          </ResponsiveContainer>
        ) : <p>No daily expense data available.</p>}
      </div>

      <div className="section">
        <h3>Cumulative Spending Over Time</h3>
        {cumulativeData.length > 0 ? (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={cumulativeData} margin={{ top: 20, right: 30, left: 30, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#aaa" />
              <XAxis dataKey="date" tick={{ fill: '#2c3e50' }}/>
              <YAxis tick={{ fill: '#2c3e50' }}/>
              <Tooltip formatter={(value) => formatCurrency(value)} />
              <Line type="monotone" dataKey="cumulative" stroke="#4B77BE" strokeWidth={3} />
            </LineChart>
          </ResponsiveContainer>
        ) : <p>No data to show cumulative spending.</p>}
      </div>

      <div className="section">
        <h3>Top 5 Largest Expenses</h3>
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

      <div className="section">
        <h3>Expense Category Breakdown (Donut Chart)</h3>
        {categoryData.length > 0 ? (
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={categoryData}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={100}
                fill="#2C3E50"
                paddingAngle={4}
              >
                {categoryData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip formatter={(value, name) => [formatCurrency(value), name]} />
              <Legend verticalAlign="bottom" height={36} />
            </PieChart>
          </ResponsiveContainer>
        ) : <p>No categorized expenses available.</p>}
      </div>

      <div className="section">
        <h3>Spending Distribution (Radar Chart)</h3>
        {multipleCategories ? (
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={radarData} margin={{ top: 40, right: 40, bottom: 40, left: 40 }}>
              <PolarGrid gridType="circle" stroke="#ccc" strokeDasharray="3 3"/>
              <PolarAngleAxis 
                dataKey="category" 
                tick={{ fill: '#2C3E50', fontSize: 12 }}
                tickLine={{ stroke: '#2C3E50' }}
              />
              <PolarRadiusAxis angle={30} domain={[0, 'auto']} stroke="#2C3E50" tick={{ fill: '#2C3E50', fontSize: 10 }}/>
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
          <p>Not enough categories for a radar chart. Diversify your spending for a richer view.</p>
        )}
      </div>

      <div className="section recurring-section">
        <h3>Most Recurring Transactions</h3>
        {recurringData.length > 0 ? (
          <div className="recurring-chart-container">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={recurringData} margin={{ top: 20, right: 30, left: 30, bottom: 50 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#aaa" />
                <XAxis 
                  dataKey="description" 
                  angle={-30} 
                  textAnchor="end" 
                  interval={0}
                  tickFormatter={shortLabel}
                  tick={{ fill: '#2C3E50', fontSize: 12 }}
                />
                <YAxis tick={{ fill: '#2C3e50' }}/>
                <Tooltip 
                  formatter={(value, name) => 
                    name === 'totalAmount' ? formatCurrency(value) : value
                  } 
                />
                <Legend />
                <Bar dataKey="totalAmount" fill="#26A65B" name="Total Amount" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : <p>No recurring transactions found.</p>}
      </div>
    </div>
  );
};

export default ChartsView;