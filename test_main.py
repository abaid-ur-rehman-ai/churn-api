from fastapi.testclient import TestClient
from app.main import app   # adjust import path if your app object lives elsewhere

client = TestClient(app)

def test_predict_valid_input():
    response = client.post("/predict", json={
        "CreditScore": 650,
        "Geography": "France",
        "Gender": "Male",
        "Age": 40,
        "Tenure": 3,
        "Balance": 60000,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 50000
    })
    assert response.status_code == 200
    assert "churn_probability" in response.json()

def test_predict_invalid_age_type():
    response = client.post("/predict", json={
        "CreditScore": 650,
        "Geography": "France",
        "Gender": "Male",
        "Age": "Thirty",   # invalid — should be rejected
        "Tenure": 3,
        "Balance": 60000,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 50000
    })
    assert response.status_code == 422

def test_predict_missing_field():
    response = client.post("/predict", json={
        "CreditScore": 650,
        "Geography": "France",
        "Gender": "Male",
        # "Age" missing entirely
        "Tenure": 3,
        "Balance": 60000,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 50000
    })
    assert response.status_code == 422