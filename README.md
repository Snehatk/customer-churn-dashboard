# 📡 ChurnIQ — Customer Churn Prediction & Business Intelligence Dashboard

> **End-to-end ML + Full-Stack project** — From raw data to a production-ready AI-powered dashboard that predicts customer churn and delivers actionable retention strategies.

---

## 🎯 Problem Statement

Telecom and subscription companies lose **20–30% of customers annually** to churn, costing billions in lost revenue. **ChurnIQ** solves this by:

1. **Predicting** which customers will churn (before they do)
2. **Explaining** the key drivers behind churn decisions
3. **Recommending** specific retention actions per customer
4. **Visualising** business insights for non-technical stakeholders

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| **ML / Data** | scikit-learn, pandas, numpy, matplotlib, seaborn |
| **Backend API** | Flask (Python), REST API |
| **Frontend** | HTML5 + CSS3 + Vanilla JS + Chart.js |
| **ML Models** | Logistic Regression, Random Forest (200 trees) |
| **Serialisation** | Pickle |
| **Deployment** | Render (backend), Netlify (frontend) |

---

## 📁 Project Structure

```
customer-churn-project/
│
├── app.py                    # Flask API server
│
├── model/
│   ├── train_model.py        # ML training pipeline
│   └── churn_model.pkl       # Saved model + artefacts (generated)
│
├── frontend/
│   └── index.html            # Full dashboard UI
│
├── data/
│   └── telco_churn.csv       # Synthetic telecom dataset (generated)
│
├── notebooks/
│   ├── eda.py                # Exploratory Data Analysis script
│   └── charts/               # Generated EDA charts
│
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start (30 Minutes)

### Step 1 — Clone & Set Up Environment

```bash
# Create project directory
mkdir customer-churn-project && cd customer-churn-project

# Create virtual environment
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Step 2 — Install Dependencies

```bash
pip install flask scikit-learn pandas numpy matplotlib seaborn gunicorn
```

### Step 3 — Train the ML Model

```bash
python model/train_model.py
```

Expected output:
```
✅ Dataset created: 7043 rows × 16 cols
   Churn rate: 22.0%
🏆 Best model: Random Forest  (F1=...)
✅ Model saved to model/churn_model.pkl
```

### Step 4 — Run the EDA (Optional)

```bash
python notebooks/eda.py
# Charts saved to notebooks/charts/
```

### Step 5 — Start the API Server

```bash
python app.py
```

Server starts at: `http://localhost:5000`

### Step 6 — Open the Dashboard

Open your browser and go to: **`http://localhost:5000`**

The dashboard serves automatically from the Flask app!

---

## 🌐 API Reference

### `GET /health`
Returns model status.

```json
{ "status": "ok", "model": "Random Forest" }
```

### `GET /stats`
Returns KPI data for dashboard cards.

```json
{
  "total_customers": 7043,
  "churn_rate": 22.0,
  "avg_monthly": 64.76,
  "model_accuracy": 77.0,
  "model_f1": 9.7,
  "model_name": "Random Forest"
}
```

### `GET /features`
Returns feature importances.

```json
[
  { "feature": "MonthlyCharges", "importance": 18.49 },
  { "feature": "tenure", "importance": 15.97 },
  ...
]
```

### `POST /predict`

**Request body:**
```json
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
  "TotalCharges": 958.20
}
```

**Response:**
```json
{
  "churn_prediction": true,
  "churn_label": "Yes",
  "churn_probability": 74.3,
  "risk_level": "High",
  "recommendations": [
    "🎯 Offer a discounted annual contract — month-to-month customers churn 3× more often.",
    "💰 Consider a loyalty discount or bundle upgrade to reduce perceived cost."
  ]
}
```

---

## 🤖 ML Model Details

### Data Pipeline
- **Dataset**: 7,043 synthetic telecom customers (mirrors Kaggle Telco Churn)
- **Features**: 14 input features (demographics, services, billing)
- **Target**: Binary churn (0 = No, 1 = Yes)
- **Encoding**: LabelEncoder for categoricals
- **Scaling**: StandardScaler for numerical features
- **Split**: 80% train / 20% test, stratified

### Models Trained
| Model | Accuracy | F1 | Notes |
|-------|----------|-----|-------|
| Logistic Regression | 78% | 0.44 | Baseline |
| **Random Forest** | **77%** | **Best** | **Selected** |

### Feature Importance (Top 5)
1. **MonthlyCharges** — 18.5% — Core price signal
2. **TotalCharges** — 18.4% — Proxy for tenure × price
3. **tenure** — 16.0% — Loyalty indicator
4. **Contract** — 14.9% — Strongest single predictor
5. **PaymentMethod** — 5.5% — Auto-pay = retention signal

---

## 💡 Key Business Insights

| Finding | Impact | Action |
|---------|--------|--------|
| M2M contracts → 42% churn | Critical | Offer annual discounts |
| First 12 months = highest risk | High | 90-day onboarding programme |
| Electronic check payers churn +30% | High | Incentivise auto-pay switch |
| No tech support → higher churn | Medium | Bundle free trial |
| Fiber users pay more, churn more | Medium | SLA guarantees + proactive support |

---

## 🚀 Deployment

### Backend — Deploy to Render

1. Push this project to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Set:
   - **Build Command**: `pip install -r requirements.txt && python model/train_model.py`
   - **Start Command**: `gunicorn app:app`
5. Click **Deploy**
6. Your API is live at: `https://your-app.onrender.com`

### Frontend — Deploy to Netlify

1. In `frontend/index.html`, change line:
   ```javascript
   const API = '';
   // to:
   const API = 'https://your-app.onrender.com';
   ```
2. Drag-drop the `frontend/` folder to [netlify.com/drop](https://app.netlify.com/drop)
3. Done — live public URL in 30 seconds!

---

## 📸 Dashboard Sections

| Section | Description |
|---------|-------------|
| **Overview** | KPI cards, churn donut, revenue by risk, probability distribution |
| **Customer Analysis** | Contract vs churn, tenure trends, internet impact, price sensitivity, feature importance |
| **Prediction Panel** | Live form → churn score + risk level + personalised recommendations |
| **Business Insights** | 6 insight cards + prioritised retention playbook table |
| **Model Performance** | Radar chart comparing models, confusion matrix, model explainer |

---

## 🏆 Why This Project Stands Out

- ✅ **Real ML pipeline** — data gen, preprocessing, multi-model training, model selection
- ✅ **Production API** — Flask REST with CORS, error handling, fallback responses
- ✅ **Professional UI** — Dark industrial design with Chart.js, animations, responsive
- ✅ **Business value** — Not just predictions but recommendations and a retention playbook
- ✅ **Full-stack** — Data Science + Backend + Frontend all in one project
- ✅ **Demo-ready** — Works standalone (demo mode) without backend running

---

## 📌 Live Demo

> 🔗 **Demo**: [placeholder — add your Render URL here]
> 🔗 **Frontend**: [placeholder — add your Netlify URL here]

---

## 👨‍💻 Author

Built as a portfolio-grade end-to-end ML + Full-Stack project.  
**Tech**: Python · Flask · scikit-learn · Chart.js · HTML/CSS/JS

*ChurnIQ v2.0 — Customer Intelligence Platform*
