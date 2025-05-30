import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import VoiceAssistant from './VoiceAssistant';
import InteractiveTutorial from './InteractiveTutorial';
import HelpCenter from './HelpCenter';
import { useNotification } from './NotificationSystem';
import './Dashboard.css';

const Dashboard = ({ user }) => {
  const navigate = useNavigate();
  const { showSuccess, showInfo, showWarning, showAchievement } = useNotification();
  
  const [showTutorial, setShowTutorial] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const [dashboardData, setDashboardData] = useState({
    balance: 0,
    recentTransactions: [],
    budgets: [],
    insights: []
  });

  useEffect(() => {
    // Check if user is new and needs tutorial
    const isNewUser = localStorage.getItem('tutorialCompleted') !== 'true';
    if (isNewUser && user) {
      setTimeout(() => {
        setShowTutorial(true);
        showInfo(
          'Welcome to Arkon!',
          'Let me show you around your financial dashboard.',
          { duration: 8000 }
        );
      }, 1000);
    }

    // Load dashboard data
    loadDashboardData();

    // Show voice assistant tip occasionally
    const tipTimer = setTimeout(() => {
      if (Math.random() > 0.7) {
        showInfo(
          'Pro Tip',
          'Try saying "Hey Arkon, what\'s my balance?" to use voice commands!',
          { duration: 10000, icon: '💡' }
        );
      }
    }, 30000);

    return () => clearTimeout(tipTimer);
  }, [user]);

  const loadDashboardData = async () => {
    try {
      const response = await fetch('/api/dashboard', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        setDashboardData(data);
        
        // Check for achievements
        checkAchievements(data);
      }
    } catch (error) {
      console.error('Failed to load dashboard:', error);
    }
  };

  const checkAchievements = (data) => {
    // Example achievement checks
    if (data.balance > 10000 && !localStorage.getItem('10k_achievement')) {
      showAchievement(
        'Milestone Reached!',
        'Your total balance exceeded $10,000! Keep up the great work!',
        {
          actions: [
            { label: 'View Details', variant: 'primary' },
            { label: 'Share', variant: 'secondary' }
          ]
        }
      );
      localStorage.setItem('10k_achievement', 'true');
    }
  };

  const handleTutorialComplete = () => {
    setShowTutorial(false);
    localStorage.setItem('tutorialCompleted', 'true');
    showSuccess(
      'Tutorial Completed!',
      'You\'re ready to start managing your finances like a pro!'
    );
  };

  const handleAddTransaction = async () => {
    // Navigate to add transaction page or open modal
    navigate('/add-transaction');
  };

  const handleSetBudget = () => {
    navigate('/budgets');
  };

  return (
    <div className="dashboard-container">
      {/* Header */}
      <header className="dashboard-header">
        <div className="header-content">
          <motion.h1
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            Welcome back, {user?.name?.split(' ')[0]}!
          </motion.h1>
          <div className="header-actions">
            <button className="help-button" onClick={() => setShowHelp(true)}>
              <span className="help-icon">❓</span>
              Help
            </button>
          </div>
        </div>
      </header>

      {/* Main Dashboard Grid */}
      <div className="dashboard-grid">
        {/* Balance Overview */}
        <motion.div
          className="dashboard-widget balance-widget dashboard-overview"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, delay: 0.1 }}
        >
          <h2>Total Balance</h2>
          <div className="balance-amount">
            ${dashboardData.balance.toLocaleString('en-US', { minimumFractionDigits: 2 })}
          </div>
          <div className="balance-change positive">
            +$523.45 (2.4%) this month
          </div>
        </motion.div>

        {/* Quick Actions */}
        <motion.div
          className="dashboard-widget quick-actions"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <h2>Quick Actions</h2>
          <div className="action-buttons">
            <button className="action-btn" onClick={handleAddTransaction}>
              <span className="action-icon">➕</span>
              Add Transaction
            </button>
            <button className="action-btn" onClick={handleSetBudget}>
              <span className="action-icon">📊</span>
              Set Budget
            </button>
            <button className="action-btn">
              <span className="action-icon">📸</span>
              Scan Receipt
            </button>
            <button className="action-btn">
              <span className="action-icon">🎯</span>
              Set Goal
            </button>
          </div>
        </motion.div>

        {/* Recent Transactions */}
        <motion.div
          className="dashboard-widget transactions-widget"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, delay: 0.3 }}
        >
          <h2>Recent Transactions</h2>
          <div className="transactions-list">
            {dashboardData.recentTransactions.slice(0, 5).map((transaction, index) => (
              <div key={transaction.id || index} className="transaction-item">
                <div className="transaction-info">
                  <div className="transaction-merchant">{transaction.merchant}</div>
                  <div className="transaction-category">{transaction.category}</div>
                </div>
                <div className="transaction-amount negative">
                  -${transaction.amount}
                </div>
              </div>
            ))}
          </div>
          <button className="view-all-btn">View All Transactions →</button>
        </motion.div>

        {/* AI Insights */}
        <motion.div
          className="dashboard-widget insights-panel"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, delay: 0.4 }}
        >
          <h2>AI Insights</h2>
          <div className="insights-list">
            <div className="insight-item">
              <span className="insight-icon">💡</span>
              <p>You could save $45/month by switching to a different coffee routine</p>
            </div>
            <div className="insight-item">
              <span className="insight-icon">📈</span>
              <p>Your grocery spending is 15% lower than last month - great job!</p>
            </div>
            <div className="insight-item">
              <span className="insight-icon">⚡</span>
              <p>Cancel unused subscription to Netflix to save $15.99/month</p>
            </div>
          </div>
        </motion.div>

        {/* Budget Status */}
        <motion.div
          className="dashboard-widget budget-widget"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, delay: 0.5 }}
        >
          <h2>Budget Status</h2>
          <div className="budget-list">
            <div className="budget-item">
              <div className="budget-header">
                <span>Food & Dining</span>
                <span>$320 / $500</span>
              </div>
              <div className="budget-progress">
                <div className="progress-bar" style={{ width: '64%' }}></div>
              </div>
            </div>
            <div className="budget-item warning">
              <div className="budget-header">
                <span>Shopping</span>
                <span>$450 / $400</span>
              </div>
              <div className="budget-progress">
                <div className="progress-bar" style={{ width: '112%' }}></div>
              </div>
            </div>
            <div className="budget-item">
              <div className="budget-header">
                <span>Transport</span>
                <span>$120 / $200</span>
              </div>
              <div className="budget-progress">
                <div className="progress-bar" style={{ width: '60%' }}></div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Spending Trends Chart */}
        <motion.div
          className="dashboard-widget chart-widget"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, delay: 0.6 }}
        >
          <h2>Spending Trends</h2>
          <div className="chart-placeholder">
            {/* Add actual chart component here */}
            <p>Interactive spending chart</p>
          </div>
        </motion.div>
      </div>

      {/* Voice Assistant */}
      <VoiceAssistant user={user} />

      {/* Interactive Tutorial */}
      <InteractiveTutorial
        isActive={showTutorial}
        onComplete={handleTutorialComplete}
        onSkip={() => setShowTutorial(false)}
      />

      {/* Help Center */}
      <HelpCenter
        isOpen={showHelp}
        onClose={() => setShowHelp(false)}
      />

      {/* Floating Tutorial Button */}
      {!showTutorial && (
        <button
          className="tutorial-trigger"
          onClick={() => setShowTutorial(true)}
          title="Start Tutorial"
        >
          <span className="tutorial-icon">🎓</span>
        </button>
      )}
    </div>
  );
};

export default Dashboard; 