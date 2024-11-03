import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_upload_file():
    with open('tests/test_data.csv', 'rb') as f:
        response = client.post("/upload", files={"file": ("test_data.csv", f, "text/csv")})
    assert response.status_code == 200
    assert "insights" in response.json() 