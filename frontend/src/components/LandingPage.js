import React, { useState, useEffect } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { GoogleLogin } from '@react-oauth/google';
import './LandingPage.css';

const LandingPage = () => {
  const navigate = useNavigate();
  const [isScrolled, setIsScrolled] = useState(false);
  const { scrollY } = useScroll();
  const opacity = useTransform(scrollY, [0, 300], [1, 0]);
  const scale = useTransform(scrollY, [0, 300], [1, 0.8]);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleGoogleSuccess = (credentialResponse) => {
    // Send token to backend for verification
    fetch('/api/auth/google', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token: credentialResponse.credential })
    })
    .then(res => res.json())
    .then(data => {
      localStorage.setItem('token', data.token);
      navigate('/dashboard');
    })
    .catch(err => console.error('Google auth failed:', err));
  };

  const features = [
    {
      icon: '🤖',
      title: 'AI-Powered Insights',
      description: 'Get personalized financial advice powered by advanced machine learning algorithms'
    },
    {
      icon: '🎙️',
      title: 'Voice Assistant',
      description: 'Talk to Arkon naturally with our voice-enabled AI assistant'
    },
    {
      icon: '🔐',
      title: 'Bank-Level Security',
      description: 'Your financial data is protected with military-grade encryption'
    },
    {
      icon: '📊',
      title: 'Real-Time Analytics',
      description: 'Track your spending and investments with beautiful, interactive visualizations'
    },
    {
      icon: '🌐',
      title: 'DeFi Integration',
      description: 'Connect to decentralized finance protocols and earn yields on your assets'
    },
    {
      icon: '📱',
      title: 'Mobile First',
      description: 'Access your finances anywhere with our powerful mobile app'
    }
  ];

  const testimonials = [
    {
      name: 'Sarah Chen',
      role: 'Entrepreneur',
      image: '/images/testimonial1.jpg',
      content: 'Arkon helped me save $5,000 in just 3 months by identifying spending patterns I never noticed.'
    },
    {
      name: 'Michael Rodriguez',
      role: 'Software Developer',
      image: '/images/testimonial2.jpg',
      content: 'The AI insights are incredible. It predicted my budget overruns before they happened!'
    },
    {
      name: 'Emma Thompson',
      role: 'Marketing Manager',
      image: '/images/testimonial3.jpg',
      content: 'Voice commands make managing finances so easy. I can check my budget while cooking!'
    }
  ];

  return (
    <div className="landing-page">
      {/* Navigation */}
      <nav className={`landing-nav ${isScrolled ? 'scrolled' : ''}`}>
        <div className="nav-container">
          <motion.div 
            className="logo"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
          >
            <span className="logo-icon">💎</span>
            <span className="logo-text">Arkon</span>
          </motion.div>
          
          <div className="nav-links">
            <a href="#features">Features</a>
            <a href="#how-it-works">How it Works</a>
            <a href="#testimonials">Testimonials</a>
            <a href="#pricing">Pricing</a>
          </div>
          
          <div className="nav-actions">
            <button className="btn-secondary" onClick={() => navigate('/login')}>
              Sign In
            </button>
            <button className="btn-primary" onClick={() => navigate('/signup')}>
              Get Started
            </button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-background">
          <div className="gradient-orb orb-1"></div>
          <div className="gradient-orb orb-2"></div>
          <div className="gradient-orb orb-3"></div>
        </div>
        
        <motion.div 
          className="hero-content"
          style={{ opacity, scale }}
        >
          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            Your AI-Powered
            <span className="gradient-text"> Financial Assistant</span>
          </motion.h1>
          
          <motion.p
            className="hero-subtitle"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
          >
            Take control of your finances with intelligent insights, voice commands, 
            and real-time analytics. Join thousands who've transformed their financial future.
          </motion.p>
          
          <motion.div
            className="hero-actions"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
          >
            <GoogleLogin
              onSuccess={handleGoogleSuccess}
              onError={() => console.log('Login Failed')}
              theme="filled_blue"
              size="large"
              text="continue_with"
              shape="pill"
            />
            
            <button className="btn-outline">
              <span className="btn-icon">▶</span>
              Watch Demo
            </button>
          </motion.div>
          
          <motion.div
            className="hero-stats"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1, delay: 1 }}
          >
            <div className="stat">
              <div className="stat-value">$2.5M+</div>
              <div className="stat-label">Saved by Users</div>
            </div>
            <div className="stat">
              <div className="stat-value">50K+</div>
              <div className="stat-label">Active Users</div>
            </div>
            <div className="stat">
              <div className="stat-value">4.9★</div>
              <div className="stat-label">App Rating</div>
            </div>
          </motion.div>
        </motion.div>
        
        <motion.div
          className="hero-visual"
          initial={{ opacity: 0, x: 100 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 1, delay: 0.5 }}
        >
          <div className="floating-dashboard">
            <img src="/images/dashboard-preview.png" alt="Arkon Dashboard" />
          </div>
          <div className="floating-card card-1">
            <div className="mini-chart"></div>
          </div>
          <div className="floating-card card-2">
            <div className="voice-wave"></div>
          </div>
        </motion.div>
      </section>

      {/* Features Section */}
      <section id="features" className="features-section">
        <div className="section-container">
          <motion.div
            className="section-header"
            initial={{ opacity: 0, y: 50 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <h2>Everything you need to master your finances</h2>
            <p>Powerful features designed to give you complete control</p>
          </motion.div>
          
          <div className="features-grid">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                className="feature-card"
                initial={{ opacity: 0, y: 50 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
                whileHover={{ y: -10, transition: { duration: 0.2 } }}
              >
                <div className="feature-icon">{feature.icon}</div>
                <h3>{feature.title}</h3>
                <p>{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section id="how-it-works" className="how-it-works-section">
        <div className="section-container">
          <motion.div
            className="section-header"
            initial={{ opacity: 0, y: 50 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <h2>Getting started is easy</h2>
            <p>Three simple steps to financial freedom</p>
          </motion.div>
          
          <div className="steps-container">
            <motion.div
              className="step"
              initial={{ opacity: 0, x: -50 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5 }}
              viewport={{ once: true }}
            >
              <div className="step-number">1</div>
              <div className="step-content">
                <h3>Connect Your Accounts</h3>
                <p>Securely link your bank accounts, credit cards, and investment portfolios</p>
              </div>
            </motion.div>
            
            <motion.div
              className="step"
              initial={{ opacity: 0, x: -50 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              viewport={{ once: true }}
            >
              <div className="step-number">2</div>
              <div className="step-content">
                <h3>AI Analyzes Your Finances</h3>
                <p>Our AI engine processes your data to find patterns and opportunities</p>
              </div>
            </motion.div>
            
            <motion.div
              className="step"
              initial={{ opacity: 0, x: -50 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.4 }}
              viewport={{ once: true }}
            >
              <div className="step-number">3</div>
              <div className="step-content">
                <h3>Get Personalized Insights</h3>
                <p>Receive actionable recommendations to optimize your financial health</p>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Voice Assistant Demo */}
      <section className="voice-demo-section">
        <div className="section-container">
          <motion.div
            className="voice-demo-content"
            initial={{ opacity: 0, y: 50 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <h2>Meet your AI financial assistant</h2>
            <p>Just ask Arkon anything about your finances</p>
            
            <div className="voice-examples">
              <div className="voice-bubble">"What's my spending this month?"</div>
              <div className="voice-bubble">"Set a budget for groceries"</div>
              <div className="voice-bubble">"Show me investment opportunities"</div>
            </div>
            
            <button className="btn-primary btn-large">
              <span className="btn-icon">🎙️</span>
              Try Voice Demo
            </button>
          </motion.div>
        </div>
      </section>

      {/* Testimonials */}
      <section id="testimonials" className="testimonials-section">
        <div className="section-container">
          <motion.div
            className="section-header"
            initial={{ opacity: 0, y: 50 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <h2>Loved by thousands of users</h2>
            <p>See what our users have to say</p>
          </motion.div>
          
          <div className="testimonials-grid">
            {testimonials.map((testimonial, index) => (
              <motion.div
                key={index}
                className="testimonial-card"
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
              >
                <div className="testimonial-content">
                  <p>"{testimonial.content}"</p>
                </div>
                <div className="testimonial-author">
                  <img src={testimonial.image} alt={testimonial.name} />
                  <div>
                    <div className="author-name">{testimonial.name}</div>
                    <div className="author-role">{testimonial.role}</div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="cta-section">
        <motion.div
          className="cta-content"
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          <h2>Ready to transform your financial future?</h2>
          <p>Join 50,000+ users who are already saving smarter</p>
          
          <div className="cta-actions">
            <GoogleLogin
              onSuccess={handleGoogleSuccess}
              onError={() => console.log('Login Failed')}
              theme="outline"
              size="large"
              text="signup_with"
              shape="pill"
            />
            <button className="btn-secondary">
              Explore Features
            </button>
          </div>
          
          <p className="cta-note">
            Free forever • No credit card required • 5-minute setup
          </p>
        </motion.div>
      </section>

      {/* Footer */}
      <footer className="landing-footer">
        <div className="footer-container">
          <div className="footer-brand">
            <div className="logo">
              <span className="logo-icon">💎</span>
              <span className="logo-text">Arkon</span>
            </div>
            <p>Your AI-powered financial assistant</p>
          </div>
          
          <div className="footer-links">
            <div className="footer-column">
              <h4>Product</h4>
              <a href="#features">Features</a>
              <a href="#pricing">Pricing</a>
              <a href="/docs">Documentation</a>
              <a href="/api">API</a>
            </div>
            
            <div className="footer-column">
              <h4>Company</h4>
              <a href="/about">About</a>
              <a href="/blog">Blog</a>
              <a href="/careers">Careers</a>
              <a href="/contact">Contact</a>
            </div>
            
            <div className="footer-column">
              <h4>Legal</h4>
              <a href="/privacy">Privacy Policy</a>
              <a href="/terms">Terms of Service</a>
              <a href="/security">Security</a>
            </div>
          </div>
        </div>
        
        <div className="footer-bottom">
          <p>&copy; 2024 Arkon Financial. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage; 