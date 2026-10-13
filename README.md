# -Employee-Attrition-Predictor-ML-
ML-модель предсказания ухода сотрудников (XGBoost + SHAP), API через FastAPI.
# Employee Attrition Predictor

ML-модель предсказания ухода сотрудников + SHAP-объяснения + FastAPI.

## Обучение
```bash
python -m src.train
```

## API
```bash
uvicorn api.main:app --reload
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{...}'
```
