import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import './Onboarding.css';

const Onboarding = ({ user, onComplete }) => {
  const navigate = useNavigate();
  const [currentStep, setCurrentStep] = useState(0);
  const [formData, setFormData] = useState({
    financialGoals: [],
    monthlyIncome: '',
    mainCategories: [],
    preferredCurrency: 'USD',
    enableVoice: true,
    enableNotifications: true
  });

  const steps = [
    {
      id: 'welcome',
      title: `Welcome to Arkon, ${user?.name?.split(' ')[0] || 'there'}!`,
      subtitle: "Let's set up your financial dashboard in just a few steps",
      content: <WelcomeStep user={user} />
    },
    {
      id: 'goals',
      title: 'What are your financial goals?',
      subtitle: 'Select all that apply',
      content: <GoalsStep formData={formData} onChange={setFormData} />
    },
    {
      id: 'income',
      title: 'Set your monthly income',
      subtitle: 'This helps us create better budgets for you',
      content: <IncomeStep formData={formData} onChange={setFormData} />
    },
    {
      id: 'categories',
      title: 'Choose your main spending categories',
      subtitle: 'We\'ll track these automatically',
      content: <CategoriesStep formData={formData} onChange={setFormData} />
    },
    {
      id: 'features',
      title: 'Enable smart features',
      subtitle: 'Customize your experience',
      content: <FeaturesStep formData={formData} onChange={setFormData} />
    },
    {
      id: 'complete',
      title: 'You\'re all set!',
      subtitle: 'Your personalized dashboard is ready',
      content: <CompleteStep />
    }
  ];

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      completeOnboarding();
    }
  };

  const handleBack = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const completeOnboarding = async () => {
    try {
      // Save preferences to backend
      await fetch('/api/user/preferences', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify(formData)
      });

      // Mark onboarding as complete
      onComplete();
      navigate('/dashboard');
    } catch (error) {
      console.error('Failed to save preferences:', error);
    }
  };

  const progress = ((currentStep + 1) / steps.length) * 100;

  return (
    <div className="onboarding-container">
      <div className="onboarding-background">
        <div className="bg-gradient"></div>
        <div className="bg-pattern"></div>
      </div>
      
      <div className="onboarding-content">
        {/* Progress Bar */}
        <div className="progress-container">
          <div className="progress-bar">
            <motion.div 
              className="progress-fill"
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.5, ease: "easeInOut" }}
            />
          </div>
          <div className="progress-steps">
            {steps.map((step, index) => (
              <motion.div
                key={step.id}
                className={`progress-step ${index <= currentStep ? 'active' : ''}`}
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: index * 0.1 }}
              >
                {index < currentStep ? '✓' : index + 1}
              </motion.div>
            ))}
          </div>
        </div>

        {/* Step Content */}
        <AnimatePresence mode="wait">
          <motion.div
            key={currentStep}
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -50 }}
            transition={{ duration: 0.3 }}
            className="step-content"
          >
            <h2>{steps[currentStep].title}</h2>
            <p className="step-subtitle">{steps[currentStep].subtitle}</p>
            
            <div className="step-body">
              {steps[currentStep].content}
            </div>
          </motion.div>
        </AnimatePresence>

        {/* Navigation */}
        <div className="onboarding-nav">
          {currentStep > 0 && (
            <button 
              className="btn-secondary"
              onClick={handleBack}
            >
              Back
            </button>
          )}
          
          <button 
            className="btn-primary"
            onClick={handleNext}
          >
            {currentStep === steps.length - 1 ? 'Get Started' : 'Continue'}
          </button>
        </div>
        
        {currentStep === 0 && (
          <button 
            className="skip-btn"
            onClick={completeOnboarding}
          >
            Skip for now
          </button>
        )}
      </div>
    </div>
  );
};

// Step Components
const WelcomeStep = ({ user }) => (
  <motion.div 
    className="welcome-step"
    initial={{ scale: 0.8, opacity: 0 }}
    animate={{ scale: 1, opacity: 1 }}
    transition={{ duration: 0.5 }}
  >
    <div className="welcome-avatar">
      {user?.picture ? (
        <img src={user.picture} alt={user.name} />
      ) : (
        <div className="avatar-placeholder">
          {user?.name?.charAt(0) || 'U'}
        </div>
      )}
    </div>
    
    <div className="welcome-features">
      <div className="feature-item">
        <span className="feature-icon">🤖</span>
        <div>
          <h4>AI-Powered Insights</h4>
          <p>Get personalized financial advice</p>
        </div>
      </div>
      <div className="feature-item">
        <span className="feature-icon">🎙️</span>
        <div>
          <h4>Voice Assistant</h4>
          <p>Manage finances with voice commands</p>
        </div>
      </div>
      <div className="feature-item">
        <span className="feature-icon">📊</span>
        <div>
          <h4>Smart Analytics</h4>
          <p>Track spending patterns automatically</p>
        </div>
      </div>
    </div>
  </motion.div>
);

const GoalsStep = ({ formData, onChange }) => {
  const goals = [
    { id: 'save_money', label: 'Save Money', icon: '💰' },
    { id: 'reduce_debt', label: 'Reduce Debt', icon: '📉' },
    { id: 'invest', label: 'Start Investing', icon: '📈' },
    { id: 'budget_better', label: 'Budget Better', icon: '📊' },
    { id: 'retire_early', label: 'Retire Early', icon: '🏖️' },
    { id: 'buy_home', label: 'Buy a Home', icon: '🏠' }
  ];

  const toggleGoal = (goalId) => {
    const currentGoals = formData.financialGoals;
    const newGoals = currentGoals.includes(goalId)
      ? currentGoals.filter(g => g !== goalId)
      : [...currentGoals, goalId];
    
    onChange({ ...formData, financialGoals: newGoals });
  };

  return (
    <div className="goals-grid">
      {goals.map((goal, index) => (
        <motion.div
          key={goal.id}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.1 }}
          className={`goal-card ${formData.financialGoals.includes(goal.id) ? 'selected' : ''}`}
          onClick={() => toggleGoal(goal.id)}
        >
          <span className="goal-icon">{goal.icon}</span>
          <span className="goal-label">{goal.label}</span>
        </motion.div>
      ))}
    </div>
  );
};

const IncomeStep = ({ formData, onChange }) => {
  const incomeRanges = [
    { id: 'under_30k', label: 'Under $30,000' },
    { id: '30k_50k', label: '$30,000 - $50,000' },
    { id: '50k_75k', label: '$50,000 - $75,000' },
    { id: '75k_100k', label: '$75,000 - $100,000' },
    { id: '100k_150k', label: '$100,000 - $150,000' },
    { id: 'over_150k', label: 'Over $150,000' }
  ];

  return (
    <div className="income-selector">
      <p className="income-note">
        This information stays private and helps us provide better recommendations
      </p>
      
      <div className="income-options">
        {incomeRanges.map((range, index) => (
          <motion.button
            key={range.id}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.05 }}
            className={`income-option ${formData.monthlyIncome === range.id ? 'selected' : ''}`}
            onClick={() => onChange({ ...formData, monthlyIncome: range.id })}
          >
            {range.label}
          </motion.button>
        ))}
      </div>
      
      <div className="prefer-not">
        <button 
          className="text-btn"
          onClick={() => onChange({ ...formData, monthlyIncome: 'prefer_not' })}
        >
          Prefer not to say
        </button>
      </div>
    </div>
  );
};

const CategoriesStep = ({ formData, onChange }) => {
  const categories = [
    { id: 'food', label: 'Food & Dining', icon: '🍕' },
    { id: 'transport', label: 'Transportation', icon: '🚗' },
    { id: 'shopping', label: 'Shopping', icon: '🛍️' },
    { id: 'entertainment', label: 'Entertainment', icon: '🎬' },
    { id: 'bills', label: 'Bills & Utilities', icon: '📱' },
    { id: 'health', label: 'Health & Fitness', icon: '💪' },
    { id: 'education', label: 'Education', icon: '📚' },
    { id: 'travel', label: 'Travel', icon: '✈️' }
  ];

  const toggleCategory = (categoryId) => {
    const currentCategories = formData.mainCategories;
    const newCategories = currentCategories.includes(categoryId)
      ? currentCategories.filter(c => c !== categoryId)
      : [...currentCategories, categoryId];
    
    onChange({ ...formData, mainCategories: newCategories });
  };

  return (
    <div className="categories-grid">
      {categories.map((category, index) => (
        <motion.div
          key={category.id}
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: index * 0.05 }}
          className={`category-card ${formData.mainCategories.includes(category.id) ? 'selected' : ''}`}
          onClick={() => toggleCategory(category.id)}
        >
          <span className="category-icon">{category.icon}</span>
          <span className="category-label">{category.label}</span>
        </motion.div>
      ))}
    </div>
  );
};

const FeaturesStep = ({ formData, onChange }) => {
  return (
    <div className="features-list">
      <motion.div 
        className="feature-toggle"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <div className="feature-info">
          <h4>🎙️ Voice Assistant</h4>
          <p>Control your finances with natural voice commands</p>
        </div>
        <label className="toggle-switch">
          <input
            type="checkbox"
            checked={formData.enableVoice}
            onChange={(e) => onChange({ ...formData, enableVoice: e.target.checked })}
          />
          <span className="toggle-slider"></span>
        </label>
      </motion.div>
      
      <motion.div 
        className="feature-toggle"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <div className="feature-info">
          <h4>🔔 Smart Notifications</h4>
          <p>Get alerts for unusual spending and budget limits</p>
        </div>
        <label className="toggle-switch">
          <input
            type="checkbox"
            checked={formData.enableNotifications}
            onChange={(e) => onChange({ ...formData, enableNotifications: e.target.checked })}
          />
          <span className="toggle-slider"></span>
        </label>
      </motion.div>
      
      <motion.div 
        className="currency-selector"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <h4>💱 Preferred Currency</h4>
        <select
          value={formData.preferredCurrency}
          onChange={(e) => onChange({ ...formData, preferredCurrency: e.target.value })}
          className="currency-select"
        >
          <option value="USD">USD ($)</option>
          <option value="EUR">EUR (€)</option>
          <option value="GBP">GBP (£)</option>
          <option value="JPY">JPY (¥)</option>
          <option value="CAD">CAD ($)</option>
          <option value="AUD">AUD ($)</option>
        </select>
      </motion.div>
    </div>
  );
};

const CompleteStep = () => (
  <motion.div 
    className="complete-step"
    initial={{ scale: 0.8, opacity: 0 }}
    animate={{ scale: 1, opacity: 1 }}
    transition={{ duration: 0.5 }}
  >
    <div className="success-animation">
      <motion.div
        className="success-circle"
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        <motion.div
          className="success-check"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 0.5, delay: 0.5 }}
        >
          ✓
        </motion.div>
      </motion.div>
    </div>
    
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.8 }}
      className="complete-message"
    >
      <p>Your personalized financial dashboard is ready!</p>
      <p className="sub-message">
        We've customized everything based on your preferences. 
        You can always change these settings later.
      </p>
    </motion.div>
  </motion.div>
);

export default Onboarding; 