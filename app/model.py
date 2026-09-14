from tensorflow.keras.models import load_model
import joblib

MODEL_PATH = "model/churn_model.keras"
SCALER_PATH = "model/scaler.pkl"

model = None
scaler = None

def load_churn_model():
    global model, scaler
    if model is None:
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Model and scaler loaded successfully")
    return model, scaler

def predict_churn(features):
    model, scaler = load_churn_model()
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return float(prediction[0][0])