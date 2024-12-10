import io
import os
import json
import logging
import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import polars as pl
from dotenv import load_dotenv
import ell
from ell.types import Message
from database import SessionLocal, FinancialDocument
from sqlalchemy.orm import Session

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ell.init(store='./logdir', autocommit=True, verbose=False)

def categorize_expenses_expression():
    return (
        pl.when(pl.col("Description").str.contains("GROCERY|SUPERMARKET|WHOLE FOODS|WALMART|COSTCO", literal=True))
        .then(pl.lit("groceries"))
        .when(pl.col("Description").str.contains("RESTAURANT|CAFE|DINER|EATERY", literal=True))
        .then(pl.lit("dining out"))
        .when(pl.col("Description").str.contains("UBER|LYFT|TRANSPORT|GAS|PETROL|SUBWAY", literal=True))
        .then(pl.lit("transport"))
        .when(pl.col("Description").str.contains("NETFLIX|HULU|SPOTIFY|AMAZON PRIME", literal=True))
        .then(pl.lit("entertainment"))
        .when(pl.col("Description").str.contains("RENT|MORTGAGE|UTILITIES|ELECTRIC|WATER|INTERNET|CABLE", literal=True))
        .then(pl.lit("housing/utilities"))
        .otherwise(pl.lit("other"))
    )

def process_financial_data(contents: bytes):
    df = pl.read_csv(io.BytesIO(contents))
    logger.info(f"Loaded DataFrame with columns: {df.columns}")

    required_cols = {"Date", "Amount", "Description"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns: {', '.join(required_cols)}")

    df = df.with_columns([
        pl.col("Date").str.strptime(pl.Date, format="%m/%d/%Y"),
        pl.col("Amount").cast(pl.Float64),
        pl.col("Description").str.to_uppercase().str.strip_chars()
    ])

    df = df.with_columns([
        categorize_expenses_expression().alias("Category")
    ])

    date_series = df["Date"]
    max_date = date_series.max()
    min_date = date_series.min()
    num_days = max((max_date - min_date).days + 1, 1)

    total_expenses = df["Amount"].sum()

    daily_expenses = (
        df.group_by("Date")
        .agg(pl.col("Amount").sum().alias("amount"))
        .sort("Date")
    ).to_dicts()
    for d in daily_expenses:
        d["date"] = d.pop("Date").strftime("%Y-%m-%d")

    recurring = (
        df.group_by("Description")
        .agg([
            pl.col("Amount").count().alias("count"),
            pl.col("Amount").sum().alias("totalamount"),
            (pl.col("Amount").sum() / pl.col("Amount").count()).alias("averageamount")
        ])
        .filter(pl.col("count") > 1)
        .sort("totalamount", descending=True)
        .to_dicts()
    )
    for r in recurring:
        r["description"] = r.pop("Description")

    cat_breakdown = (
        df.group_by("Category")
        .agg(pl.col("Amount").sum().alias("amount"))
        .sort("amount", descending=True)
        .to_dicts()
    )
    for c in cat_breakdown:
        c["category"] = c.pop("Category")

    top_5 = (
        df.group_by("Description")
        .agg(pl.col("Amount").sum().alias("amount"))
        .sort("amount", descending=True)
        .head(5)
        .to_dicts()
    )
    for t in top_5:
        t["description"] = t.pop("Description")

    avg_daily = float(total_expenses / num_days)
    avg_monthly = float(total_expenses / (num_days / 30))

    summary = {
        "total_expenses": float(total_expenses),
        "average_daily": avg_daily,
        "average_monthly": avg_monthly,
        "daily_expenses": daily_expenses,
        "category_breakdown": cat_breakdown,
        "top_5_expenses": top_5,
        "recurring_transactions": recurring,
        "date_range": {
            "start": min_date.strftime("%Y-%m-%d"),
            "end": max_date.strftime("%Y-%m-%d")
        }
    }
    return summary

class DateJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        return super().default(obj)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    logger.info(f"Received file upload: {file.filename}")

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    # If filename exists, rename it
    base_name, ext = os.path.splitext(file.filename)
    new_filename = file.filename
    doc_count = db.query(FinancialDocument).filter(FinancialDocument.filename == file.filename).count()
    if doc_count > 0:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"{base_name}_{timestamp}{ext}"

    summary = process_financial_data(contents)

    doc = FinancialDocument(
        filename=new_filename,
        summary_json=json.dumps(summary, cls=DateJSONEncoder)
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    return {
        "status": "success",
        "file_id": doc.id,
        "summary": summary,
        "filename": new_filename
    }

@app.get("/files")
def list_files(db: Session = Depends(get_db)):
    docs = db.query(FinancialDocument).order_by(FinancialDocument.upload_date.desc()).all()
    return [
        {
            "id": doc.id,
            "filename": doc.filename,
            "upload_date": doc.upload_date.isoformat() if doc.upload_date else None
        }
        for doc in docs
    ]

@app.get("/file/{file_id}")
def get_file(file_id: int, db: Session = Depends(get_db)):
    doc = db.query(FinancialDocument).filter(FinancialDocument.id == file_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="File not found")

    summary = json.loads(doc.summary_json)
    return {
        "status": "success",
        "summary": summary
    }

if __name__ == "__main__":
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)