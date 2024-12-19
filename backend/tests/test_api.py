import pytest
from fastapi.testclient import TestClient
from main import app
import io

client = TestClient(app)

def test_get_stock_price():
    response = client.get("/stock_price?symbol=AAPL")
    assert response.status_code == 200
    data = response.json()
    assert "symbol" in data
    assert "price" in data
    assert data["symbol"] == "AAPL"
    assert isinstance(data["price"], float)

def test_get_stock_price_invalid():
    response = client.get("/stock_price?symbol=")
    assert response.status_code == 400

def test_upload_stock_data():
    # Create sample CSV content
    csv_content = """Date,Price,Volume
2024-01-01,150.00,1000000
2024-01-02,155.00,1200000
2024-01-03,153.00,900000"""
    
    files = {
        'file': ('test.csv', io.StringIO(csv_content), 'text/csv')
    }
    
    response = client.post("/upload_stock_data", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "daily_prices" in data["summary"]
    assert len(data["summary"]["daily_prices"]) == 3

def test_upload_invalid_stock_data():
    files = {
        'file': ('test.csv', io.StringIO("invalid,csv"), 'text/csv')
    }
    response = client.post("/upload_stock_data", files=files)
    assert response.status_code == 400 