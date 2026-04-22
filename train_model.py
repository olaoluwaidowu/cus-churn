"""
MIT807: Artificial Intelligence & Its Business Applications
Group 1 — Customer Churn Prediction
Model Training Script: Trains a Random Forest pipeline and saves it to disk.
"""

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score

# ── 1. Load data ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("📂  Loading data …")
df = pd.read_csv(DATA_PATH)

# ── 2. Basic cleaning ────────────────────────────────────────────────────────
df.drop(columns=["customerID"], inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ── 3. Feature / target split ────────────────────────────────────────────────
X = df.drop(columns=["Churn"])
y = df["Churn"]

# ── 4. Column groups ─────────────────────────────────────────────────────────
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

# ── 5. Pre-processor ─────────────────────────────────────────────────────────
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_COLS),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
         CATEGORICAL_COLS),
    ]
)

# ── 6. Full pipeline ─────────────────────────────────────────────────────────
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=500,
        max_leaf_nodes=30,
        max_features="sqrt",
        oob_score=True,
        n_jobs=-1,
        random_state=50,
    )),
])

# ── 7. Train / test split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=40, stratify=y
)

# ── 8. Fit ───────────────────────────────────────────────────────────────────
print("🤖  Training Random Forest pipeline …")
pipeline.fit(X_train, y_train)

# ── 9. Evaluate ──────────────────────────────────────────────────────────────
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅  Test Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

# ── 10. Save ─────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "rf_pipeline.pkl")

joblib.dump(pipeline, MODEL_PATH)
print(f"💾  Model saved → {MODEL_PATH}")
