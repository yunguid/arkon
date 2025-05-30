import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './HelpCenter.css';

const HelpCenter = ({ isOpen, onClose }) => {
  const [activeCategory, setActiveCategory] = useState('getting-started');
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedFAQ, setExpandedFAQ] = useState(null);

  const helpCategories = [
    {
      id: 'getting-started',
      title: 'Getting Started',
      icon: '🚀',
      articles: [
        {
          id: 'first-steps',
          title: 'Your First Steps with Arkon',
          content: 'Learn how to set up your account, connect banks, and start tracking expenses.'
        },
        {
          id: 'dashboard-overview',
          title: 'Understanding Your Dashboard',
          content: 'A complete guide to navigating your financial dashboard and key features.'
        },
        {
          id: 'voice-commands',
          title: 'Using Voice Commands',
          content: 'Master voice commands to check balances, add transactions, and more.'
        }
      ]
    },
    {
      id: 'features',
      title: 'Features',
      icon: '⚡',
      articles: [
        {
          id: 'ai-insights',
          title: 'AI-Powered Financial Insights',
          content: 'How our AI analyzes your spending and provides personalized recommendations.'
        },
        {
          id: 'budgeting',
          title: 'Smart Budgeting Tools',
          content: 'Create and manage budgets that adapt to your spending patterns.'
        },
        {
          id: 'investments',
          title: 'Investment Tracking',
          content: 'Monitor your portfolio and get AI-driven investment suggestions.'
        }
      ]
    },
    {
      id: 'security',
      title: 'Security & Privacy',
      icon: '🔒',
      articles: [
        {
          id: 'data-protection',
          title: 'How We Protect Your Data',
          content: 'Learn about our bank-level security and encryption standards.'
        },
        {
          id: 'privacy-controls',
          title: 'Privacy Controls',
          content: 'Manage your privacy settings and data sharing preferences.'
        }
      ]
    },
    {
      id: 'troubleshooting',
      title: 'Troubleshooting',
      icon: '🔧',
      articles: [
        {
          id: 'common-issues',
          title: 'Common Issues & Solutions',
          content: 'Quick fixes for the most common problems users encounter.'
        },
        {
          id: 'sync-problems',
          title: 'Fixing Sync Issues',
          content: 'Troubleshoot bank connection and transaction sync problems.'
        }
      ]
    }
  ];

  const faqs = [
    {
      question: 'Is my financial data secure?',
      answer: 'Yes! We use bank-level 256-bit encryption and never store your bank credentials. All data is encrypted both in transit and at rest.'
    },
    {
      question: 'How does the AI assistant work?',
      answer: 'Our AI analyzes your spending patterns, identifies trends, and provides personalized recommendations. It learns from your habits to give increasingly accurate insights.'
    },
    {
      question: 'Can I use Arkon offline?',
      answer: 'Yes! The mobile app works offline and syncs when you\'re back online. Voice commands require an internet connection.'
    },
    {
      question: 'How do I cancel my subscription?',
      answer: 'You can cancel anytime from Account Settings. You\'ll continue to have access until the end of your billing period.'
    },
    {
      question: 'Which banks are supported?',
      answer: 'We support over 10,000 banks and financial institutions worldwide. Check our full list in the settings.'
    }
  ];

  const videoTutorials = [
    {
      id: 'intro',
      title: 'Introduction to Arkon',
      duration: '2:30',
      thumbnail: '/images/tutorial-intro.jpg'
    },
    {
      id: 'voice',
      title: 'Mastering Voice Commands',
      duration: '4:15',
      thumbnail: '/images/tutorial-voice.jpg'
    },
    {
      id: 'budgets',
      title: 'Creating Smart Budgets',
      duration: '3:45',
      thumbnail: '/images/tutorial-budgets.jpg'
    }
  ];

  const filteredArticles = () => {
    if (!searchQuery) return [];
    
    const query = searchQuery.toLowerCase();
    const results = [];
    
    helpCategories.forEach(category => {
      category.articles.forEach(article => {
        if (article.title.toLowerCase().includes(query) || 
            article.content.toLowerCase().includes(query)) {
          results.push({ ...article, category: category.title });
        }
      });
    });
    
    return results;
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        className="help-center-overlay"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={onClose}
      >
        <motion.div
          className="help-center-modal"
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          onClick={(e) => e.stopPropagation()}
        >
          <div className="help-header">
            <h2>Help Center</h2>
            <button className="close-btn" onClick={onClose}>×</button>
          </div>

          <div className="help-search">
            <input
              type="text"
              placeholder="Search for help..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="search-input"
            />
            <span className="search-icon">🔍</span>
          </div>

          {searchQuery ? (
            <div className="search-results">
              <h3>Search Results</h3>
              {filteredArticles().length > 0 ? (
                <div className="results-list">
                  {filteredArticles().map(article => (
                    <div key={article.id} className="search-result">
                      <h4>{article.title}</h4>
                      <p>{article.content}</p>
                      <span className="result-category">{article.category}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="no-results">No results found. Try different keywords.</p>
              )}
            </div>
          ) : (
            <div className="help-content">
              <div className="help-sidebar">
                <h3>Categories</h3>
                {helpCategories.map(category => (
                  <button
                    key={category.id}
                    className={`category-btn ${activeCategory === category.id ? 'active' : ''}`}
                    onClick={() => setActiveCategory(category.id)}
                  >
                    <span className="category-icon">{category.icon}</span>
                    {category.title}
                  </button>
                ))}
              </div>

              <div className="help-main">
                {activeCategory === 'getting-started' && (
                  <div className="video-tutorials">
                    <h3>Video Tutorials</h3>
                    <div className="tutorials-grid">
                      {videoTutorials.map(video => (
                        <div key={video.id} className="video-card">
                          <div className="video-thumbnail">
                            <img src={video.thumbnail} alt={video.title} />
                            <div className="play-overlay">▶</div>
                          </div>
                          <h4>{video.title}</h4>
                          <span className="video-duration">{video.duration}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div className="articles-section">
                  <h3>Articles</h3>
                  <div className="articles-list">
                    {helpCategories
                      .find(cat => cat.id === activeCategory)
                      ?.articles.map(article => (
                        <div key={article.id} className="article-card">
                          <h4>{article.title}</h4>
                          <p>{article.content}</p>
                          <button className="read-more">Read more →</button>
                        </div>
                      ))}
                  </div>
                </div>

                <div className="faq-section">
                  <h3>Frequently Asked Questions</h3>
                  <div className="faq-list">
                    {faqs.map((faq, index) => (
                      <div key={index} className="faq-item">
                        <button
                          className="faq-question"
                          onClick={() => setExpandedFAQ(expandedFAQ === index ? null : index)}
                        >
                          {faq.question}
                          <span className="faq-toggle">
                            {expandedFAQ === index ? '−' : '+'}
                          </span>
                        </button>
                        <AnimatePresence>
                          {expandedFAQ === index && (
                            <motion.div
                              className="faq-answer"
                              initial={{ height: 0, opacity: 0 }}
                              animate={{ height: 'auto', opacity: 1 }}
                              exit={{ height: 0, opacity: 0 }}
                              transition={{ duration: 0.3 }}
                            >
                              <p>{faq.answer}</p>
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          <div className="help-footer">
            <div className="contact-options">
              <h3>Still need help?</h3>
              <div className="contact-buttons">
                <button className="contact-btn">
                  <span className="contact-icon">💬</span>
                  Live Chat
                </button>
                <button className="contact-btn">
                  <span className="contact-icon">📧</span>
                  Email Support
                </button>
                <button className="contact-btn">
                  <span className="contact-icon">📞</span>
                  Call Us
                </button>
              </div>
            </div>
            <p className="response-time">Average response time: 2 minutes</p>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default HelpCenter; 