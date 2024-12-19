from database import engine, Base, StockAnalysisHistory

def run_migration():
    Base.metadata.create_all(bind=engine, tables=[StockAnalysisHistory.__table__])

if __name__ == "__main__":
    run_migration() 