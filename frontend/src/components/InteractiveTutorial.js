import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './InteractiveTutorial.css';

const InteractiveTutorial = ({ isActive, onComplete, onSkip }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [targetElement, setTargetElement] = useState(null);

  const tutorialSteps = [
    {
      id: 'welcome',
      title: 'Welcome to Your Financial Dashboard!',
      content: 'Let me show you around. This won\'t take long!',
      target: null,
      position: 'center'
    },
    {
      id: 'voice-assistant',
      title: 'Meet Your AI Assistant',
      content: 'Click this button to talk to me anytime. Just say "What\'s my balance?" or "Show me my spending"',
      target: '.voice-button',
      position: 'top-left'
    },
    {
      id: 'dashboard-overview',
      title: 'Your Financial Overview',
      content: 'Here you can see your total balance, recent transactions, and spending trends at a glance',
      target: '.dashboard-overview',
      position: 'bottom'
    },
    {
      id: 'quick-actions',
      title: 'Quick Actions',
      content: 'Add transactions, set budgets, or scan receipts with just one click',
      target: '.quick-actions',
      position: 'left'
    },
    {
      id: 'insights',
      title: 'AI-Powered Insights',
      content: 'Get personalized recommendations based on your spending patterns',
      target: '.insights-panel',
      position: 'right'
    },
    {
      id: 'help',
      title: 'Need Help?',
      content: 'Click the help button anytime for tutorials, FAQs, or to contact support',
      target: '.help-button',
      position: 'top'
    },
    {
      id: 'complete',
      title: 'You\'re All Set!',
      content: 'Start managing your finances like a pro. Remember, I\'m always here to help!',
      target: null,
      position: 'center'
    }
  ];

  const currentStepData = tutorialSteps[currentStep];

  useEffect(() => {
    if (!isActive) return;

    if (currentStepData.target) {
      const element = document.querySelector(currentStepData.target);
      setTargetElement(element);
    } else {
      setTargetElement(null);
    }
  }, [currentStep, isActive, currentStepData.target]);

  const handleNext = () => {
    if (currentStep < tutorialSteps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      onComplete();
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const getTooltipPosition = () => {
    if (!targetElement) return { top: '50%', left: '50%', transform: 'translate(-50%, -50%)' };

    const rect = targetElement.getBoundingClientRect();
    const position = currentStepData.position;

    switch (position) {
      case 'top':
        return {
          top: rect.top - 20,
          left: rect.left + rect.width / 2,
          transform: 'translate(-50%, -100%)'
        };
      case 'bottom':
        return {
          top: rect.bottom + 20,
          left: rect.left + rect.width / 2,
          transform: 'translate(-50%, 0)'
        };
      case 'left':
        return {
          top: rect.top + rect.height / 2,
          left: rect.left - 20,
          transform: 'translate(-100%, -50%)'
        };
      case 'right':
        return {
          top: rect.top + rect.height / 2,
          left: rect.right + 20,
          transform: 'translate(0, -50%)'
        };
      case 'top-left':
        return {
          top: rect.top - 20,
          left: rect.left,
          transform: 'translate(0, -100%)'
        };
      default:
        return {
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)'
        };
    }
  };

  if (!isActive) return null;

  return (
    <AnimatePresence>
      <div className="tutorial-overlay">
        {/* Dark overlay with spotlight */}
        <motion.div
          className="tutorial-backdrop"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onSkip}
        />

        {/* Spotlight effect */}
        {targetElement && (
          <motion.div
            className="tutorial-spotlight"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            style={{
              position: 'fixed',
              top: targetElement.getBoundingClientRect().top - 10,
              left: targetElement.getBoundingClientRect().left - 10,
              width: targetElement.getBoundingClientRect().width + 20,
              height: targetElement.getBoundingClientRect().height + 20,
              borderRadius: '8px',
              pointerEvents: 'none',
              boxShadow: '0 0 0 9999px rgba(0, 0, 0, 0.7)'
            }}
          />
        )}

        {/* Tutorial tooltip */}
        <motion.div
          className={`tutorial-tooltip ${currentStepData.position || 'center'}`}
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.9 }}
          style={targetElement ? getTooltipPosition() : {}}
        >
          <div className="tutorial-header">
            <h3>{currentStepData.title}</h3>
            <button className="tutorial-close" onClick={onSkip}>×</button>
          </div>

          <div className="tutorial-content">
            <p>{currentStepData.content}</p>
          </div>

          <div className="tutorial-footer">
            <div className="tutorial-progress">
              {tutorialSteps.map((_, index) => (
                <div
                  key={index}
                  className={`progress-dot ${index <= currentStep ? 'active' : ''}`}
                />
              ))}
            </div>

            <div className="tutorial-actions">
              {currentStep > 0 && (
                <button className="btn-text" onClick={handlePrevious}>
                  Back
                </button>
              )}
              <button className="btn-primary" onClick={handleNext}>
                {currentStep === tutorialSteps.length - 1 ? 'Finish' : 'Next'}
              </button>
            </div>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
};

export default InteractiveTutorial; 