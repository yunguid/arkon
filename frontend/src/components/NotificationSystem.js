import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './NotificationSystem.css';

const NotificationContext = React.createContext();

export const useNotification = () => {
  const context = React.useContext(NotificationContext);
  if (!context) {
    throw new Error('useNotification must be used within NotificationProvider');
  }
  return context;
};

export const NotificationProvider = ({ children }) => {
  const [notifications, setNotifications] = useState([]);

  const addNotification = (notification) => {
    const id = Date.now();
    const newNotification = {
      id,
      ...notification,
      timestamp: new Date()
    };
    
    setNotifications(prev => [...prev, newNotification]);

    // Auto-dismiss after duration (default 5 seconds)
    if (notification.autoDismiss !== false) {
      setTimeout(() => {
        removeNotification(id);
      }, notification.duration || 5000);
    }
  };

  const removeNotification = (id) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  };

  const showSuccess = (title, message, options = {}) => {
    addNotification({
      type: 'success',
      title,
      message,
      icon: '✅',
      ...options
    });
  };

  const showError = (title, message, options = {}) => {
    addNotification({
      type: 'error',
      title,
      message,
      icon: '❌',
      ...options
    });
  };

  const showWarning = (title, message, options = {}) => {
    addNotification({
      type: 'warning',
      title,
      message,
      icon: '⚠️',
      ...options
    });
  };

  const showInfo = (title, message, options = {}) => {
    addNotification({
      type: 'info',
      title,
      message,
      icon: 'ℹ️',
      ...options
    });
  };

  const showAchievement = (title, message, options = {}) => {
    addNotification({
      type: 'achievement',
      title,
      message,
      icon: '🏆',
      duration: 7000,
      ...options
    });
  };

  const value = {
    notifications,
    addNotification,
    removeNotification,
    showSuccess,
    showError,
    showWarning,
    showInfo,
    showAchievement
  };

  return (
    <NotificationContext.Provider value={value}>
      {children}
      <NotificationContainer 
        notifications={notifications} 
        onRemove={removeNotification} 
      />
    </NotificationContext.Provider>
  );
};

const NotificationContainer = ({ notifications, onRemove }) => {
  return (
    <div className="notification-container">
      <AnimatePresence>
        {notifications.map((notification) => (
          <Notification
            key={notification.id}
            notification={notification}
            onRemove={() => onRemove(notification.id)}
          />
        ))}
      </AnimatePresence>
    </div>
  );
};

const Notification = ({ notification, onRemove }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [progress, setProgress] = useState(100);

  useEffect(() => {
    if (notification.autoDismiss !== false && notification.duration) {
      const interval = setInterval(() => {
        setProgress(prev => {
          if (prev <= 0) {
            clearInterval(interval);
            return 0;
          }
          return prev - (100 / (notification.duration / 100));
        });
      }, 100);

      return () => clearInterval(interval);
    }
  }, [notification]);

  const handleAction = (action) => {
    if (action.onClick) {
      action.onClick();
    }
    if (action.dismissOnClick !== false) {
      onRemove();
    }
  };

  return (
    <motion.div
      className={`notification notification-${notification.type}`}
      initial={{ x: 400, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      exit={{ x: 400, opacity: 0 }}
      transition={{ type: 'spring', damping: 25, stiffness: 300 }}
      layout
    >
      <div className="notification-header">
        <div className="notification-icon">{notification.icon}</div>
        <div className="notification-content">
          <h4 className="notification-title">{notification.title}</h4>
          {notification.message && (
            <p className={`notification-message ${isExpanded ? 'expanded' : ''}`}>
              {notification.message}
            </p>
          )}
          {notification.message && notification.message.length > 100 && (
            <button 
              className="expand-btn"
              onClick={() => setIsExpanded(!isExpanded)}
            >
              {isExpanded ? 'Show less' : 'Show more'}
            </button>
          )}
        </div>
        <button className="notification-close" onClick={onRemove}>×</button>
      </div>

      {notification.actions && (
        <div className="notification-actions">
          {notification.actions.map((action, index) => (
            <button
              key={index}
              className={`notification-action ${action.variant || 'default'}`}
              onClick={() => handleAction(action)}
            >
              {action.label}
            </button>
          ))}
        </div>
      )}

      {notification.autoDismiss !== false && (
        <div className="notification-progress">
          <div 
            className="progress-bar" 
            style={{ width: `${progress}%` }}
          />
        </div>
      )}
    </motion.div>
  );
};

// Example usage helpers
export const notificationExamples = {
  // Success notifications
  transactionAdded: () => ({
    type: 'success',
    title: 'Transaction Added',
    message: 'Your transaction has been recorded successfully.',
    icon: '💰'
  }),
  
  budgetCreated: (budgetName) => ({
    type: 'success',
    title: 'Budget Created',
    message: `Your "${budgetName}" budget is now active and tracking.`,
    icon: '📊'
  }),
  
  // Warning notifications
  budgetExceeded: (category, percentage) => ({
    type: 'warning',
    title: 'Budget Alert',
    message: `You've used ${percentage}% of your ${category} budget this month.`,
    icon: '⚠️',
    actions: [
      { label: 'View Budget', variant: 'primary' },
      { label: 'Adjust Limit', variant: 'secondary' }
    ]
  }),
  
  // Error notifications
  syncError: () => ({
    type: 'error',
    title: 'Sync Failed',
    message: 'Unable to sync your transactions. Please check your connection.',
    icon: '🔄',
    autoDismiss: false,
    actions: [
      { label: 'Retry', variant: 'primary' },
      { label: 'Dismiss', variant: 'secondary' }
    ]
  }),
  
  // Achievement notifications
  savingsGoalReached: (goalName, amount) => ({
    type: 'achievement',
    title: 'Goal Achieved! 🎉',
    message: `Congratulations! You've reached your "${goalName}" savings goal of $${amount}.`,
    icon: '🏆',
    duration: 10000,
    actions: [
      { label: 'View Achievement', variant: 'primary' },
      { label: 'Set New Goal', variant: 'secondary' }
    ]
  }),
  
  // Info notifications
  voiceCommandTip: () => ({
    type: 'info',
    title: 'Pro Tip',
    message: 'You can say "Hey Arkon, what did I spend on food this week?" to get instant insights.',
    icon: '💡',
    duration: 8000
  })
};

export default NotificationProvider; 