from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from src.features import build_features, build_target

class AttritionModel:
    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            use_label_encoder=False, eval_metric="logloss")
        self.features = None

    def fit(self, df):
        X = build_features(df)
        y = build_target(df)
        self.features = X.columns.tolist()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        preds = self.model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds)
        return {"auc": auc, "report": classification_report(y_test, (preds > 0.5).astype(int))}

    def predict_proba(self, df):
        X = build_features(df)[self.features]
        return self.model.predict_proba(X)[:, 1]
