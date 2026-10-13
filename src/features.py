import pandas as pd

FEATURES = [
    "age", "tenure_years", "salary", "performance_score",
    "engagement_score", "overtime_hours", "promotions",
    "manager_changes", "training_hours", "distance_km", "satisfaction"
]

def build_features(df):
    X = df[FEATURES].copy()
    X["salary_per_year_tenure"] = X["salary"] / (X["tenure_years"] + 1)
    X["satisfaction_per_engagement"] = X["satisfaction"] / (X["engagement_score"] + 1e-6)
    return X

def build_target(df):
    return df["attrition"]
