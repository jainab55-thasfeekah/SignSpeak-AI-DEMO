"""Train a starter classifier from sign_landmarks.csv.
Run after collecting samples: python train_model.py
"""
import os, sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib

DATA = "sign_landmarks.csv"
if not os.path.exists(DATA):
    raise SystemExit("No sign_landmarks.csv found. Run collect_data.py first.")
df = pd.read_csv(DATA).dropna()
if "label" not in df.columns or df["label"].nunique() < 2:
    raise SystemExit("Collect samples for at least 2 different labels.")
counts = df["label"].value_counts()
print("Samples per class:\n", counts)
if counts.min() < 10:
    print("WARNING: Some classes have fewer than 10 samples. Collect more for better results.")
X = df.drop(columns=["label"]).astype(float)
y = df["label"].astype(str)
stratify = y if y.value_counts().min() >= 2 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify
)
model = RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(f"Holdout accuracy: {accuracy_score(y_test, pred):.3f}")
print(classification_report(y_test, pred, zero_division=0))
joblib.dump(model, "sign_model.joblib")
with open("sign_labels.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(sorted(y.unique())))
print("Saved sign_model.joblib and sign_labels.txt")
print("Important: test with people and sessions not used in collection; a random frame split can overestimate real-world accuracy.")
