import os
import joblib
import pandas as pd

BASE = os.path.dirname(os.path.dirname(__file__))

MODEL = os.path.join(BASE, "model", "rf_model.pkl")
FEATURES = os.path.join(BASE, "model", "feature_names.pkl")

TEST = os.path.join(BASE, "data", "test_features.csv")

OUTPUT = os.path.join(BASE, "predictions.csv")

print("=" * 60)
print("Loading model...")
print("=" * 60)

model = joblib.load(MODEL)
feature_names = joblib.load(FEATURES)

test = pd.read_csv(TEST)

X = test[feature_names]

print("Predicting probabilities...")

probs = model.predict_proba(X)[:, 1]

test["truth_score"] = probs

test.to_csv(OUTPUT, index=False)

print()
print("Prediction file saved:")
print(OUTPUT)