# Customer Churn Prediction API

This project deploys a Customer Churn prediction model using FastAPI.

This project predicts whether a bank customer will leave (churn) or stay using a Machine Learning model deployed with FastAPI.

## Live Demo link 
https://churn-api-hk05.onrender.com/docs

## Input Example
{
  "CreditScore": 650,
  "Geography": "France",
  "Gender": "Male",
  "Age": 35,
  "Tenure": 5,
  "Balance": 50000.0,
  "NumOfProducts": 1,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 80000.0
}

## Features
- Real-time churn prediction
- FastAPI REST API
- Input validation
- Swagger UI documentation

## Tech Stack
- Python
- FastAPI
- TensorFlow / Keras
- Uvicorn
- Scikit-learn

## How to Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload





