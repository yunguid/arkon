import React, { useState, useEffect } from 'react';
import './BudgetManager.css';

const BudgetManager = () => {
  const [budgets, setBudgets] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [showAddForm, setShowAddForm] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [newBudget, setNewBudget] = useState({
    category: '',
    monthly_limit: ''
  });
  const [editingBudget, setEditingBudget] = useState(null);

  // Available categories based on backend
  const categories = [
    'Shopping', 'Food', 'Transport', 'Housing', 
    'Entertainment', 'Utilities', 'Healthcare', 
    'Investment', 'Other'
  ];

  useEffect(() => {
    fetchBudgets();
    fetchAlerts();
  }, []);

  const fetchBudgets = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/budgets');
      if (!response.ok) throw new Error('Failed to fetch budgets');
      const data = await response.json();
      setBudgets(data.budgets || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchAlerts = async () => {
    try {
      const response = await fetch('http://localhost:8000/alerts?unread_only=true');
      if (!response.ok) throw new Error('Failed to fetch alerts');
      const data = await response.json();
      setAlerts(data.alerts || []);
    } catch (err) {
      console.error('Error fetching alerts:', err);
    }
  };

  const handleAddBudget = async (e) => {
    e.preventDefault();
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/budgets', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          category: newBudget.category,
          monthly_limit: parseFloat(newBudget.monthly_limit)
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to create budget');
      }

      await fetchBudgets();
      setShowAddForm(false);
      setNewBudget({ category: '', monthly_limit: '' });
    } catch (err) {
      setError(err.message);
    }
  };

  const handleUpdateBudget = async (budgetId) => {
    if (!editingBudget || !editingBudget.newLimit) return;

    try {
      const response = await fetch(`http://localhost:8000/budgets/${budgetId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          monthly_limit: parseFloat(editingBudget.newLimit)
        })
      });

      if (!response.ok) throw new Error('Failed to update budget');

      await fetchBudgets();
      setEditingBudget(null);
    } catch (err) {
      setError(err.message);
    }
  };

  const handleDeleteBudget = async (budgetId) => {
    if (!window.confirm('Are you sure you want to delete this budget?')) return;

    try {
      const response = await fetch(`http://localhost:8000/budgets/${budgetId}`, {
        method: 'DELETE'
      });

      if (!response.ok) throw new Error('Failed to delete budget');

      await fetchBudgets();
    } catch (err) {
      setError(err.message);
    }
  };

  const markAlertAsRead = async (alertId) => {
    try {
      await fetch(`http://localhost:8000/alerts/${alertId}/read`, {
        method: 'PUT'
      });
      fetchAlerts();
    } catch (err) {
      console.error('Error marking alert as read:', err);
    }
  };

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(amount);
  };

  return (
    <div className="budget-manager">
      <div className="budget-header">
        <h2>Budget Management</h2>
        <button 
          className="add-budget-btn"
          onClick={() => setShowAddForm(!showAddForm)}
        >
          {showAddForm ? 'Cancel' : '+ Add Budget'}
        </button>
      </div>

      {error && (
        <div className="error-message">
          {error}
          <button onClick={() => setError(null)}>×</button>
        </div>
      )}

      {/* Alerts Section */}
      {alerts.length > 0 && (
        <div className="alerts-section">
          <h3>Budget Alerts</h3>
          {alerts.map(alert => (
            <div key={alert.id} className={`alert ${alert.type}`}>
              <div className="alert-content">
                <span className="alert-icon">⚠️</span>
                <p>{alert.message}</p>
              </div>
              <button 
                className="dismiss-alert"
                onClick={() => markAlertAsRead(alert.id)}
              >
                Dismiss
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Add Budget Form */}
      {showAddForm && (
        <form className="add-budget-form" onSubmit={handleAddBudget}>
          <div className="form-group">
            <label>Category</label>
            <select
              value={newBudget.category}
              onChange={(e) => setNewBudget({ ...newBudget, category: e.target.value })}
              required
            >
              <option value="">Select a category</option>
              {categories.map(cat => (
                <option key={cat} value={cat}>{cat}</option>
              ))}
            </select>
          </div>
          <div className="form-group">
            <label>Monthly Limit</label>
            <input
              type="number"
              step="0.01"
              min="0"
              value={newBudget.monthly_limit}
              onChange={(e) => setNewBudget({ ...newBudget, monthly_limit: e.target.value })}
              placeholder="Enter amount"
              required
            />
          </div>
          <div className="form-actions">
            <button type="submit" className="save-btn">Save Budget</button>
            <button 
              type="button" 
              className="cancel-btn"
              onClick={() => {
                setShowAddForm(false);
                setNewBudget({ category: '', monthly_limit: '' });
              }}
            >
              Cancel
            </button>
          </div>
        </form>
      )}

      {/* Budgets List */}
      {loading ? (
        <div className="loading">Loading budgets...</div>
      ) : (
        <div className="budgets-list">
          {budgets.length === 0 ? (
            <div className="no-budgets">
              <p>No budgets set yet. Click "Add Budget" to get started!</p>
            </div>
          ) : (
            budgets.map(budget => (
              <div key={budget.id} className="budget-item">
                <div className="budget-info">
                  <h4>{budget.category}</h4>
                  {editingBudget?.id === budget.id ? (
                    <div className="edit-form">
                      <input
                        type="number"
                        step="0.01"
                        min="0"
                        value={editingBudget.newLimit}
                        onChange={(e) => setEditingBudget({ 
                          ...editingBudget, 
                          newLimit: e.target.value 
                        })}
                        className="edit-input"
                      />
                      <button 
                        className="save-btn small"
                        onClick={() => handleUpdateBudget(budget.id)}
                      >
                        Save
                      </button>
                      <button 
                        className="cancel-btn small"
                        onClick={() => setEditingBudget(null)}
                      >
                        Cancel
                      </button>
                    </div>
                  ) : (
                    <p className="budget-limit">
                      Limit: {formatCurrency(budget.monthly_limit)}
                    </p>
                  )}
                </div>
                <div className="budget-actions">
                  {editingBudget?.id !== budget.id && (
                    <>
                      <button
                        className="edit-btn"
                        onClick={() => setEditingBudget({ 
                          id: budget.id, 
                          newLimit: budget.monthly_limit 
                        })}
                      >
                        Edit
                      </button>
                      <button
                        className="delete-btn"
                        onClick={() => handleDeleteBudget(budget.id)}
                      >
                        Delete
                      </button>
                    </>
                  )}
                </div>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
};

export default BudgetManager; 