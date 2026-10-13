from src.data import generate_hr_dataset
from src.features import build_features, build_target

def test_features_shape():
    df = generate_hr_dataset(100)
    X = build_features(df)
    y = build_target(df)
    assert len(X) == 100
    assert len(y) == 100
