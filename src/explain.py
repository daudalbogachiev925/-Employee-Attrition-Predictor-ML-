import shap
import joblib
import pandas as pd
from src.features import build_features

class Explainer:
    def __init__(self, model_path="models/attrition.pkl"):
        self.model = joblib.load(model_path)
        self.explainer = shap.TreeExplainer(self.model.model)

    def explain(self, df):
        X = build_features(df)[self.model.features]
        shap_values = self.explainer.shap_values(X)
        return pd.DataFrame({
            "feature": X.columns,
            "importance": abs(shap_values).mean(axis=0)
        }).sort_values("importance", ascending=False)
