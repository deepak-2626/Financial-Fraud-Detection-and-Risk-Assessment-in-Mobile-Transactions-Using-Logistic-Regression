"""
train_model.py
--------------
Downloads the PaySim dataset (ealaxi/paysim1), trains a Logistic Regression
model for fraud detection, and saves model + metadata for app.py.

Actual PaySim columns:
  step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
  nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud

Features used (NO demographic info):
  step, amount, type,
  balanceDiffOrig  (engineered: how much the sender balance changed vs expected)
  balanceDiffDest  (engineered: how much the recipient balance changed vs expected)

Target: isFraud

Run once before launching the Streamlit app:
    python train_model.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# ── 1. Download dataset ────────────────────────────────────────────────────────
import kagglehub

print("Downloading dataset...")
path = kagglehub.dataset_download("ealaxi/paysim1")
print(f"Dataset path: {path}")

csv_file = None
for fname in os.listdir(path):
    if fname.endswith(".csv"):
        csv_file = os.path.join(path, fname)
        break

if csv_file is None:
    raise FileNotFoundError(f"No CSV found in {path}")

print(f"Loading: {csv_file}")
df = pd.read_csv(csv_file)
print(f"Shape: {df.shape}")
print("Columns:", df.columns.tolist())

# ── 2. Rename for clarity ──────────────────────────────────────────────────────
df = df.rename(columns={"isFraud": "fraud"})
print(f"Fraud rate: {df['fraud'].mean():.4f}")

# ── 3. Feature engineering ────────────────────────────────────────────────────
# These capture the "balance anomaly" signal without using any PII.
# For a real fraud, the origin balance drops to 0 and dest balance doesn't
# increase as expected — these features encode that pattern.
df["balanceDiffOrig"] = df["newbalanceOrig"] - df["oldbalanceOrg"] + df["amount"]
df["balanceDiffDest"] = df["newbalanceDest"] - df["oldbalanceDest"] - df["amount"]

# ── 4. Feature selection ───────────────────────────────────────────────────────
FEATURES_NUM = ["step", "amount", "oldbalanceOrg", "newbalanceOrig",
                "oldbalanceDest", "newbalanceDest",
                "balanceDiffOrig", "balanceDiffDest"]
FEATURES_CAT = ["type"]
TARGET       = "fraud"

df = df.dropna(subset=FEATURES_NUM + FEATURES_CAT + [TARGET])
df[TARGET] = df[TARGET].astype(int)

X = df[FEATURES_NUM + FEATURES_CAT]
y = df[TARGET]

# ── 5. Build pipeline ─────────────────────────────────────────────────────────
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), FEATURES_NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FEATURES_CAT),
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )),
])

# ── 6. Train / test split ─────────────────────────────────────────────────────
# Sample for speed — 500k rows is plenty for logistic regression.
# Separate fraud/legit rows then combine to preserve class ratio.
SAMPLE_SIZE = 500_000
fraud_rows  = df[df[TARGET] == 1]
legit_rows  = df[df[TARGET] == 0]

fraud_n = len(fraud_rows)
legit_n = min(SAMPLE_SIZE - fraud_n, len(legit_rows))

df_sample = pd.concat([
    fraud_rows,
    legit_rows.sample(n=legit_n, random_state=42),
]).sample(frac=1, random_state=42).reset_index(drop=True)   # shuffle

X = df_sample[FEATURES_NUM + FEATURES_CAT]
y = df_sample[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── 7. Train ──────────────────────────────────────────────────────────────────
print("Training Logistic Regression...")
pipeline.fit(X_train, y_train)
print("Done.")

# ── 8. Evaluate ───────────────────────────────────────────────────────────────
y_pred  = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("\n── Classification Report ──")
print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

cm  = confusion_matrix(y_test, y_pred)
roc = roc_auc_score(y_test, y_proba)
print("Confusion Matrix:\n", cm)
print(f"ROC-AUC: {roc:.4f}")

# ── 9. Risk metadata per transaction type ─────────────────────────────────────
type_stats = (
    df.groupby("type")["fraud"]
    .agg(fraud_rate="mean", count="count")
    .reset_index()
)
type_stats["risk_tier"] = pd.cut(
    type_stats["fraud_rate"],
    bins=[-1, 0.001, 0.01, 1.0],
    labels=["low", "medium", "high"],
)
type_risk = type_stats.set_index("type")["risk_tier"].astype(str).to_dict()
print("\nTransaction type risk tiers:", type_risk)

# ── 10. Amount stats per type (for anomaly scoring) ───────────────────────────
amount_stats = (
    df.groupby("type")["amount"]
    .agg(mean="mean", std="std")
    .fillna(0)
    .to_dict(orient="index")
)

# ── 11. Valid types + step range ──────────────────────────────────────────────
valid_types = sorted(df["type"].dropna().unique().tolist())
step_max    = int(df["step"].max())
amount_max  = float(df["amount"].quantile(0.99))   # cap at 99th pct for UI

# ── 12. Save everything ───────────────────────────────────────────────────────
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

joblib.dump(pipeline, os.path.join(OUT_DIR, "model.pkl"))
print("\nSaved model.pkl")

metadata = {
    "valid_types":   valid_types,
    "type_risk":     type_risk,
    "amount_stats":  amount_stats,
    "step_max":      step_max,
    "amount_max":    round(amount_max, 2),
    "features_num":  FEATURES_NUM,
    "features_cat":  FEATURES_CAT,
    "eval": {
        "roc_auc":          round(roc, 4),
        "confusion_matrix": cm.tolist(),
        "report": classification_report(
            y_test, y_pred,
            target_names=["Legit", "Fraud"],
            output_dict=True,
        ),
    },
}

with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
print("Saved metadata.json")
print("\nAll done. Run: streamlit run app.py")
