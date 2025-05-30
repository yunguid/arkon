#!/usr/bin/env python3
"""
Upgrade script for Arkon Financial Analyzer
This script helps upgrade from the original version to the enhanced version
"""

import os
import sys
import shutil
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_environment():
    """Check if the environment is properly set up"""
    logger.info("Checking environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False
    
    # Check if we're in the project root
    if not os.path.exists("backend") or not os.path.exists("frontend"):
        logger.error("Please run this script from the project root directory")
        return False
    
    return True

def backup_database():
    """Create a backup of the existing database"""
    logger.info("Backing up database...")
    
    db_path = Path("backend/financial_docs.db")
    if db_path.exists():
        backup_path = Path(f"backend/financial_docs_backup_{int(os.path.getmtime(db_path))}.db")
        shutil.copy2(db_path, backup_path)
        logger.info(f"Database backed up to: {backup_path}")
    else:
        logger.warning("No existing database found")

def upgrade_backend():
    """Upgrade backend components"""
    logger.info("Upgrading backend...")
    
    # Install new dependencies
    logger.info("Installing backend dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "backend/requirements.txt"], check=True)
    
    # Run migrations
    logger.info("Running database migrations...")
    os.chdir("backend")
    try:
        subprocess.run([sys.executable, "migrations/add_new_tables.py"], check=True)
        logger.info("Database migrations completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Migration failed: {e}")
        return False
    finally:
        os.chdir("..")
    
    return True

def upgrade_frontend():
    """Upgrade frontend components"""
    logger.info("Upgrading frontend...")
    
    os.chdir("frontend")
    try:
        # Install dependencies
        logger.info("Installing frontend dependencies...")
        subprocess.run(["npm", "install"], check=True)
        logger.info("Frontend dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Frontend upgrade failed: {e}")
        return False
    finally:
        os.chdir("..")
    
    return True

def create_env_file():
    """Create or update .env file"""
    logger.info("Setting up environment variables...")
    
    env_path = Path("backend/.env")
    env_example = """# Arkon Financial Analyzer Environment Variables
ANTHROPIC_API_KEY=your_anthropic_key_here
PERPLEXITY_API_KEY=your_perplexity_key_here
DATABASE_URL=sqlite:///./financial_docs.db

# Optional: Yahoo Finance credentials
# YAHOO_FINANCE_APP_ID=your_app_id
# YAHOO_FINANCE_CLIENT_ID=your_client_id
# YAHOO_FINANCE_CLIENT_SECRET=your_client_secret
"""
    
    if not env_path.exists():
        env_path.write_text(env_example)
        logger.info("Created .env file - please update with your API keys")
    else:
        logger.info(".env file already exists - please ensure all required keys are set")

def print_next_steps():
    """Print instructions for next steps"""
    print("\n" + "="*60)
    print("✅ UPGRADE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n📋 Next Steps:")
    print("1. Update your API keys in backend/.env")
    print("2. Start the enhanced backend:")
    print("   cd backend && python main_improved.py")
    print("3. Start the frontend:")
    print("   cd frontend && npm start")
    print("\n🚀 New Features Available:")
    print("- Budget Management System")
    print("- Enhanced Data Export (CSV/JSON)")
    print("- AI Category Caching")
    print("- Advanced Statistics")
    print("- Rate Limiting & Performance Improvements")
    print("\n📖 See README_IMPROVEMENTS.md for detailed documentation")
    print("="*60)

def main():
    """Main upgrade process"""
    logger.info("Starting Arkon Financial Analyzer upgrade...")
    
    if not check_environment():
        logger.error("Environment check failed. Aborting upgrade.")
        return 1
    
    # Backup database
    backup_database()
    
    # Upgrade components
    if not upgrade_backend():
        logger.error("Backend upgrade failed")
        return 1
    
    if not upgrade_frontend():
        logger.error("Frontend upgrade failed")
        return 1
    
    # Setup environment
    create_env_file()
    
    # Print success message and next steps
    print_next_steps()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 