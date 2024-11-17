from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import polars as pl
import io
import ell as ell
import anthropic
import os
from dotenv import load_dotenv
import logging
from ell.types import Message

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Validate API key presence
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

app = FastAPI()

# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],  # More permissive for development
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Initialize Anthropic client once
client = anthropic.Client(api_key=ANTHROPIC_API_KEY)

# ELL initialization (if needed)
ell.init(
    store='./logdir',
    autocommit=True,
    verbose=False
)

@ell.simple(
    model="claude-3-5-sonnet-20241022",
    client=client,
    max_tokens=8000
)
def analyze_financial_data(financial_data: str) -> list[Message]:
    """Analyze financial data and provide insights."""
    return [
        Message(
            role="system",
            content=
            """
            You will be analyzing financial data provided in CSV format. The data contains information about transactions, including dates, descriptions, and amounts in USD. Your task is to analyze this data, identify patterns, and provide insights to help the user understand their spending habits.

            Follow these steps to analyze the data and provide insights:

            1. Data Processing:
            - Parse the CSV data, ensuring you correctly interpret the Date, Description, and Amount columns.
            - Convert the Amount column to numerical values for calculations.

            2. Spending Analysis:
            - Calculate total spending for the given period.
            - Identify the top 5 largest expenses.
            - Determine average daily and monthly spending.
            - Categorize expenses into broad categories (e.g., food, entertainment, utilities) based on the descriptions.

            3. Provide Quantitative Insights:
            - Present your findings in a clear, concise manner.
            - Use specific numbers and percentages to support your insights.

            4. Recommendations:
            - After providing the quantitative analysis, adopt the tone of an elderly financier with extensive wisdom (e.g., Charlie Munger, Warren Buffet, or Jim Simons).
            - Offer advice on how the user can save money and improve their financial habits.
            - Base your recommendations on the patterns and insights you've identified in the data.

            Present your analysis and recommendations in the following format:

            

            
            [Provide overall spending analysis, including total spent, average daily/monthly spending, and top expenses]
            

            
            [List 3-5 key insights derived from the data analysis]
            

            
            [Provide recommendations and financial advice in the style of an experienced, wise investor]
            
            

            Remember to be thorough in your analysis, provide specific numbers to support your insights, and offer practical, wise advice in the tone of an elderly financier.
            """
        ),
        Message(
            role="user",
            content=f"""[Your detailed user prompt here with {financial_data} included]"""
        )
    ]

def process_financial_data(contents: bytes):
    """
    Process the CSV data using Polars and compute summary statistics.

    Returns:
        summary (dict): A dictionary containing summaries like total expenses,
                        expenses by category, daily expenses, etc.
    """
    # Read CSV data into Polars DataFrame
    df = pl.read_csv(io.BytesIO(contents))

    # Ensure columns are correctly typed
    df = df.with_columns([
        pl.col("Date").str.strptime(pl.Date, format="%m/%d/%Y"),
        pl.col("Amount").cast(pl.Float64),
        pl.col("Description").cast(pl.Utf8),
    ])

    # Total expenses
    total_expenses = df.select(pl.col("Amount").sum()).item()
    logger.info(f"Total expenses: {total_expenses}\n\n")

    # Expenses by day
    daily_expenses = df.group_by("Date").agg(pl.col("Amount").sum()).sort("Date")
    logger.info(f"Daily expenses: {daily_expenses}\n\n")

    # Expenses by description
    expenses_by_description = df.group_by("Description").agg(pl.col("Amount").sum()).sort("Amount", descending=True)
    logger.info(f"Expenses by description: {expenses_by_description}\n\n")

    # Add recurring transactions analysis
    recurring_transactions = (
        df.group_by("Description")
        .agg([
            pl.col("Amount").count().alias("count"),
            pl.col("Amount").abs().sum().alias("totalAmount"),
            (pl.col("Amount").abs().sum() / pl.col("Amount").count()).alias("averageAmount")
        ])
        .filter(pl.col("count") > 1)  # Only keep items occurring more than once
        .sort("count", descending=True)
        .head(10)  # Top 10 recurring transactions
    )
    logger.info(f"Recurring transactions: {recurring_transactions}\n\n")

    # Update summary dictionary
    summary = {
        "total_expenses": total_expenses,
        "daily_expenses": daily_expenses.to_dicts(),
        "expenses_by_description": expenses_by_description.to_dicts(),
        "recurring_transactions": recurring_transactions.to_dicts()  # Add new data
    }

    return summary

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    logger.info(f"Received file upload: {file.filename}")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        contents = await file.read()
        
        # Immediately process data for visualization
        summary = process_financial_data(contents)
        
        # Return initial response with summary data
        return {
            "status": "processing",
            "summary": summary,
            "message": "Charts data ready. AI analysis in progress..."
        }
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/analyze")
# async def analyze_file(file: UploadFile = File(...)):
#     try:
#         contents = await file.read()
#         df = pl.read_csv(io.BytesIO(contents))
#         csv_data = df.write_csv()
        
#         # Perform AI analysis
#         insights = analyze_financial_data(csv_data)
        
#         return {
#             "status": "complete",
#             "insights": insights
#         }
        
#     except Exception as e:
#         logger.error(f"Error analyzing file: {str(e)}", exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
