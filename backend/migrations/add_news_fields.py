import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import engine, Base, StockNews
from sqlalchemy import Column, JSON, Text

def run_migration():
    # Add new columns to existing table
    with engine.connect() as conn:
        print("Adding new columns to stock_news table...")
        conn.execute('ALTER TABLE stock_news ADD COLUMN key_developments JSON')
        conn.execute('ALTER TABLE stock_news ADD COLUMN price_impact TEXT')
        conn.execute('ALTER TABLE stock_news ADD COLUMN risks JSON')
        print("Migration complete!")

if __name__ == "__main__":
    run_migration() 