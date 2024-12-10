# database.py
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, func
from sqlalchemy.orm import sessionmaker, declarative_base
import json

Base = declarative_base()

class FinancialDocument(Base):
    __tablename__ = 'financial_documents'
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    upload_date = Column(DateTime(timezone=True), server_default=func.now())
    summary_json = Column(Text)  # store the summary as a JSON string

engine = create_engine('sqlite:///./financial_data.db', connect_args={"check_same_thread": False})
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)