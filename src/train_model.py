import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

BASE = os.path.dirname(os.path.dirname(__file__))

TRAIN = os.path.join(BASE, "data", "train_features.csv")

MODEL_DIR = os.path.join(BASE, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

print("="*60)
print("Loading Training Features")
print("="*60)

df = pd.read_csv(TRAIN)

drop_cols = [
    "fact_uri",
    "head",
    "relation",
    "tail",
    "truth"
]

X = df.drop(columns=drop_cols)
y = df["truth"]

print("Training Samples :", len(X))
print("Features :", len(X.columns))
print()

print("Cross Validating...")

model = RandomForestClassifier(
    n_estimators=700,
    max_depth=16,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    X,
    y,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1
)

print()
print("Fold ROC-AUC")

for i,s in enumerate(scores):
    print(i+1,":",round(s,4))

print()

print("Average ROC-AUC :",round(scores.mean(),4))

print()

print("Training Final Model...")

model.fit(X,y)

joblib.dump(
    model,
    os.path.join(MODEL_DIR,"rf_model.pkl")
)

joblib.dump(
    list(X.columns),
    os.path.join(MODEL_DIR,"feature_names.pkl")
)

print()
print("Model Saved.")