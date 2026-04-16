"""
Customer Churn Prediction - Model Training Script
Generates synthetic telecom data, trains ML models, saves best model + encoders.
Run: python model/train_model.py
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

# ── 1. Generate realistic synthetic dataset ──────────────────────────────────
np.random.seed(42)
N = 7043  # same size as original Telco dataset

def generate_churn_dataset(n=N):
    tenure        = np.random.randint(0, 73, n)
    monthly       = np.round(np.random.uniform(18, 118, n), 2)
    total_charges = np.round(monthly * tenure + np.random.normal(0, 50, n), 2)
    total_charges = np.clip(total_charges, 0, None)

    contract      = np.random.choice(
        ["Month-to-month", "One year", "Two year"],
        n, p=[0.55, 0.24, 0.21]
    )
    internet      = np.random.choice(
        ["DSL", "Fiber optic", "No"],
        n, p=[0.34, 0.44, 0.22]
    )
    payment       = np.random.choice(
        ["Electronic check", "Mailed check",
         "Bank transfer (automatic)", "Credit card (automatic)"],
        n, p=[0.34, 0.23, 0.22, 0.21]
    )
    senior        = np.random.choice([0, 1], n, p=[0.84, 0.16])
    partner       = np.random.choice(["Yes", "No"], n, p=[0.48, 0.52])
    dependents    = np.random.choice(["Yes", "No"], n, p=[0.30, 0.70])
    phone_service = np.random.choice(["Yes", "No"], n, p=[0.90, 0.10])
    paperless     = np.random.choice(["Yes", "No"], n, p=[0.59, 0.41])
    tech_support  = np.random.choice(["Yes", "No", "No internet service"], n)
    online_sec    = np.random.choice(["Yes", "No", "No internet service"], n)

    # Churn probability based on real-world factors
    churn_prob = (
        0.05
        + 0.25 * (contract == "Month-to-month")
        + 0.10 * (internet == "Fiber optic")
        + 0.08 * (monthly > 70)
        - 0.12 * (tenure > 36)
        - 0.08 * (contract == "Two year")
        + 0.05 * (payment == "Electronic check")
        + 0.04 * (senior == 1)
        - 0.05 * (tech_support == "Yes")
        + np.random.normal(0, 0.05, n)
    )
    churn_prob = np.clip(churn_prob, 0.02, 0.95)
    churn      = (np.random.rand(n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customerID":      [f"CUST-{i:05d}" for i in range(n)],
        "gender":          np.random.choice(["Male", "Female"], n),
        "SeniorCitizen":   senior,
        "Partner":         partner,
        "Dependents":      dependents,
        "tenure":          tenure,
        "PhoneService":    phone_service,
        "InternetService": internet,
        "OnlineSecurity":  online_sec,
        "TechSupport":     tech_support,
        "Contract":        contract,
        "PaperlessBilling":paperless,
        "PaymentMethod":   payment,
        "MonthlyCharges":  monthly,
        "TotalCharges":    total_charges,
        "Churn":           churn,
    })
    return df

# ── 2. Load / generate data ──────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
df = generate_churn_dataset()
df.to_csv("data/telco_churn.csv", index=False)
print(f"✅ Dataset created: {df.shape[0]} rows × {df.shape[1]} cols")
print(f"   Churn rate: {df['Churn'].mean()*100:.1f}%")

# ── 3. Preprocessing ─────────────────────────────────────────────────────────
df_model = df.drop(columns=["customerID"])

# Fix TotalCharges (sometimes blank in real dataset)
df_model["TotalCharges"] = pd.to_numeric(df_model["TotalCharges"], errors="coerce")
df_model["TotalCharges"].fillna(df_model["TotalCharges"].median(), inplace=True)

# Encode categoricals
cat_cols = df_model.select_dtypes(include="object").columns.tolist()
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    encoders[col] = le

# Feature / target split
X = df_model.drop("Churn", axis=1)
y = df_model["Churn"]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ── 4. Train models ───────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":        RandomForestClassifier(
                                n_estimators=200, max_depth=12,
                                random_state=42, n_jobs=-1
                            ),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        "model":     model,
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
    }
    print(f"\n── {name} ──")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

# ── 5. Select best model ──────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["f1"])
best_model = results[best_name]["model"]
print(f"\n🏆 Best model: {best_name}  (F1={results[best_name]['f1']:.4f})")

# Feature importance (Random Forest)
feature_importance = {}
if hasattr(best_model, "feature_importances_"):
    fi = best_model.feature_importances_
    feature_importance = dict(zip(X.columns.tolist(), fi.tolist()))
    fi_sorted = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    print("\n📌 Top 10 Feature Importances:")
    for feat, score in fi_sorted[:10]:
        print(f"   {feat:<25} {score:.4f}")

# ── 6. Save artefacts ────────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)

payload = {
    "model":              best_model,
    "scaler":             scaler,
    "encoders":           encoders,
    "feature_columns":    X.columns.tolist(),
    "feature_importance": feature_importance,
    "model_name":         best_name,
    "metrics":            {k: v for k, v in results[best_name].items() if k != "model"},
    "churn_rate":         float(df["Churn"].mean()),
    "total_customers":    int(len(df)),
    "avg_monthly":        float(df["MonthlyCharges"].mean()),
}

with open("model/churn_model.pkl", "wb") as f:
    pickle.dump(payload, f)

print("\n✅ Model saved to model/churn_model.pkl")
print("   Run app.py to start the API server.")
