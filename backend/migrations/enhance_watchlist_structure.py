import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import engine, Base, WatchlistStock, StockFiling, FinancialMetrics

def run_migration():
    print("Running database migration...")
    
    # Drop existing tables if they exist
    Base.metadata.drop_all(bind=engine)
    
    # Create all tables with new schema
    Base.metadata.create_all(bind=engine)
    
    print("Migration complete!")

if __name__ == "__main__":
    run_migration() 