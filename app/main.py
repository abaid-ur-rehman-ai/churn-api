from fastapi import FastAPI
from pydantic import BaseModel
from app.preprocess import preprocess_input
from app.model import predict_churn

app = FastAPI(title="Customer Churn Prediction API")

class CustomerData(BaseModel):
    CreditScore: int
    Geography: str          # France / Germany / Spain
    Gender: str             # Male / Female
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

@app.get("/")
def home():
    return {"message": "Customer Churn API is running"}

@app.post("/predict")
def predict(data: CustomerData):
    input_dict = data.dict()
    
    # Preprocess
    features = preprocess_input(input_dict)
    
    # Predict
    probability = predict_churn(features)
    
    result = "Churn" if probability > 0.5 else "No Churn"
    
    return {
        "churn_probability": round(probability, 4),
        "prediction": result
    }