from fastapi import FastAPI
from pydantic import BaseModel
from src.data import load
from src.model import AttritionModel
import pandas as pd

app = FastAPI(title="Attrition Predictor API")
model = AttritionModel()
model.fit(load())

class Employee(BaseModel):
    age: int
    tenure_years: float
    salary: float
    performance_score: float
    engagement_score: float
    overtime_hours: float
    promotions: int
    manager_changes: int
    training_hours: float
    distance_km: float
    satisfaction: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(emp: Employee):
    df = pd.DataFrame([emp.dict()])
    prob = model.predict_proba(df)[0]
    level = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.4 else "LOW"
    return {"risk_probability": float(prob), "risk_level": level}
