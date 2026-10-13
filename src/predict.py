import joblib
import pandas as pd

class Predictor:
    def __init__(self, path="models/attrition.pkl"):
        self.model = joblib.load(path)

    def predict(self, df: pd.DataFrame):
        probs = self.model.predict_proba(df)
        return [{"employee_id": i, "risk": float(p), "risk_level": self._level(p)}
                for i, p in zip(df.index, probs)]

    def _level(self, p):
        if p > 0.7: return "HIGH"
        if p > 0.4: return "MEDIUM"
        return "LOW"
