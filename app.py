"""
Customer Churn Prediction — Flask API
Run: python app.py
Endpoints:
  GET  /             → health check
  GET  /stats        → dataset KPIs
  GET  /features     → feature importances
  POST /predict      → churn prediction
"""

import pickle, os, json
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder="frontend", static_url_path="")

# ── Load model artefacts ─────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "churn_model.pkl")
with open(MODEL_PATH, "rb") as f:
    artefacts = pickle.load(f)

model             = artefacts["model"]
scaler            = artefacts["scaler"]
encoders          = artefacts["encoders"]
feature_columns   = artefacts["feature_columns"]
feature_importance= artefacts["feature_importance"]
metrics           = artefacts["metrics"]
churn_rate        = artefacts["churn_rate"]
total_customers   = artefacts["total_customers"]
avg_monthly       = artefacts["avg_monthly"]

print(f"✅ Model loaded: {artefacts['model_name']}")
print(f"   Accuracy={metrics['accuracy']:.3f}  F1={metrics['f1']:.3f}")

# ── CORS helper ───────────────────────────────────────────────────────────────
def _cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response

@app.after_request
def after_request(resp):
    return _cors(resp)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the frontend dashboard."""
    return send_from_directory("frontend", "index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": artefacts["model_name"]})

@app.route("/stats")
def stats():
    """Return KPI stats for the dashboard."""
    return jsonify({
        "total_customers": total_customers,
        "churn_rate":      round(churn_rate * 100, 1),
        "avg_monthly":     round(avg_monthly, 2),
        "model_accuracy":  round(metrics["accuracy"] * 100, 1),
        "model_f1":        round(metrics["f1"] * 100, 1),
        "model_name":      artefacts["model_name"],
    })

@app.route("/features")
def features():
    """Return sorted feature importances."""
    sorted_fi = sorted(
        feature_importance.items(), key=lambda x: x[1], reverse=True
    )
    return jsonify([{"feature": k, "importance": round(v * 100, 2)} for k, v in sorted_fi[:12]])

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    """
    Predict churn for a single customer.

    Expected JSON body:
    {
      "gender": "Male",
      "SeniorCitizen": 0,
      "Partner": "Yes",
      "Dependents": "No",
      "tenure": 12,
      "PhoneService": "Yes",
      "InternetService": "Fiber optic",
      "OnlineSecurity": "No",
      "TechSupport": "No",
      "Contract": "Month-to-month",
      "PaperlessBilling": "Yes",
      "PaymentMethod": "Electronic check",
      "MonthlyCharges": 79.85,
      "TotalCharges": 958.2
    }
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    # ── Build feature vector ──────────────────────────────────────────────────
    row = {}
    for col in feature_columns:
        val = data.get(col, None)
        if val is None:
            return jsonify({"error": f"Missing field: {col}"}), 400

        if col in encoders:
            le = encoders[col]
            val_str = str(val)
            if val_str not in le.classes_:
                val_str = le.classes_[0]          # fallback
            val = int(le.transform([val_str])[0])
        else:
            val = float(val)
        row[col] = val

    X = np.array([[row[c] for c in feature_columns]])
    X_scaled = scaler.transform(X)

    # ── Predict ───────────────────────────────────────────────────────────────
    pred       = int(model.predict(X_scaled)[0])
    prob_arr   = model.predict_proba(X_scaled)[0]
    churn_prob = float(prob_arr[1])

    if churn_prob >= 0.65:
        risk = "High"
    elif churn_prob >= 0.35:
        risk = "Medium"
    else:
        risk = "Low"

    # ── Business recommendation ───────────────────────────────────────────────
    recommendations = []
    monthly = float(data.get("MonthlyCharges", 0))
    contract = str(data.get("Contract", ""))
    tenure   = int(data.get("tenure", 0))
    internet = str(data.get("InternetService", ""))
    payment  = str(data.get("PaymentMethod", ""))
    tech     = str(data.get("TechSupport", ""))
    security = str(data.get("OnlineSecurity", ""))

    if pred == 1:
        if contract == "Month-to-month":
            recommendations.append("🎯 Offer a discounted annual contract — month-to-month customers churn 3× more often.")
        if monthly > 70:
            recommendations.append("💰 Consider a loyalty discount or bundle upgrade to reduce perceived cost.")
        if tenure < 12:
            recommendations.append("🔑 Early-tenure customers are fragile — trigger a 90-day success programme.")
        if tech == "No":
            recommendations.append("🛠️ Offer complimentary tech-support trial — it significantly reduces churn.")
        if security == "No" and internet != "No":
            recommendations.append("🔒 Provide a free Online Security add-on; it improves retention for Fiber users.")
        if payment == "Electronic check":
            recommendations.append("💳 Encourage auto-pay via credit card — it correlates with lower churn.")
        if not recommendations:
            recommendations.append("📞 Proactively reach out with a personalised retention offer.")
    else:
        recommendations.append("✅ Customer appears stable. Schedule a check-in at 6-month mark to maintain satisfaction.")
        if monthly > 80:
            recommendations.append("⭐ High-value, low-risk customer — ideal candidate for upsell / referral programme.")

    return jsonify({
        "churn_prediction": bool(pred),
        "churn_label":      "Yes" if pred else "No",
        "churn_probability": round(churn_prob * 100, 1),
        "risk_level":       risk,
        "recommendations":  recommendations,
    })

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
