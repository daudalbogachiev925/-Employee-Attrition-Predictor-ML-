import joblib
import os
from src.data import load
from src.model import AttritionModel

def main():
    df = load()
    model = AttritionModel()
    metrics = model.fit(df)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/attrition.pkl")
    print(f"Model saved. AUC = {metrics['auc']:.4f}")
    print(metrics["report"])

if __name__ == "__main__":
    main()
