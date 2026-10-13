import pandas as pd
import numpy as np

def generate_hr_dataset(n=3000, seed=42):
    np.random.seed(seed)
    df = pd.DataFrame({
        "age": np.random.randint(22, 60, n),
        "tenure_years": np.random.exponential(3, n).round(1),
        "salary": np.random.normal(90000, 30000, n).round(0),
        "performance_score": np.random.normal(3.5, 0.7, n).round(2),
        "engagement_score": np.random.uniform(1, 10, n).round(1),
        "overtime_hours": np.random.exponential(5, n).round(1),
        "promotions": np.random.poisson(1.2, n),
        "manager_changes": np.random.poisson(0.8, n),
        "training_hours": np.random.normal(20, 10, n).round(0),
        "distance_km": np.random.exponential(15, n).round(1),
        "satisfaction": np.random.uniform(1, 5, n).round(1),
    })
    # Attrition как функция признаков
    logit = (
        -0.05 * df["engagement_score"]
        - 0.02 * df["tenure_years"]
        + 0.1 * df["overtime_hours"]
        - 0.3 * df["promotions"]
        + 0.15 * df["manager_changes"]
        - 0.5 * df["satisfaction"]
        + np.random.normal(0, 1, n)
    )
    df["attrition"] = (1 / (1 + np.exp(-logit)) > 0.5).astype(int)
    return df

def load(path=None):
    if path:
        return pd.read_csv(path)
    return generate_hr_dataset()
