import pandas as pd
import numpy as np
import joblib

scaler = joblib.load('model/scaler.pkl')  # load once, at import time

def preprocess_input(data: dict):
    df = pd.DataFrame([data])

    df['Geography_Germany'] = 1 if data.get('Geography') == 'Germany' else 0
    df['Geography_Spain'] = 1 if data.get('Geography') == 'Spain' else 0
    df['Gender_Male'] = 1 if data.get('Gender') == 'Male' else 0

    final_columns = [
        'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
        'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
        'Geography_Germany', 'Geography_Spain', 'Gender_Male'
    ]

    df = df[final_columns]

    scaled = scaler.transform(df)   # <-- the missing step
    return scaled.astype('float32')