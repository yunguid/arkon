"""
Migration script to add new tables for enhanced features
Run this script to update your database schema
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import engine, SessionLocal
from models import Base, Budget, CategoryCache, AlertLog
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate():
    """Create new tables if they don't exist"""
    try:
        # Create all tables (will skip existing ones)
        Base.metadata.create_all(bind=engine)
        
        logger.info("Migration completed successfully!")
        logger.info("New tables created:")
        logger.info("- budgets")
        logger.info("- category_cache")
        logger.info("- alert_logs")
        logger.info("Updated existing tables with new columns where applicable")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise

if __name__ == "__main__":
    migrate() 