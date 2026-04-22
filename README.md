# 📡 Telecom Customer Churn Predictor

> **MIT807: Artificial Intelligence & Its Business Applications**  
> Group Project · **GROUP 1**

---

## 📌 Overview

This application predicts whether a telecom customer is likely to **churn** (cancel their service) based on their account information, service subscriptions, and billing details.

It is built as a **Streamlit web application** powered by a **Random Forest** machine learning model trained on the IBM Telco Customer Churn dataset. The model is packaged inside a **Docker** container and deployed on **Render**.

---

## 🌐 Live Demo

**[https://churn-predictor-group1.onrender.com](https://churn-predictor-group1.onrender.com)**

> ⚠️ Render free-tier services spin down after inactivity. First load may take ~30 seconds.

---

## 🧠 Model

| Property | Detail |
|---|---|
| Algorithm | Random Forest Classifier |
| Estimators | 500 trees |
| Test Accuracy | ~81.4% |
| Dataset | IBM Telco Customer Churn (7,043 customers) |
| Pipeline | `StandardScaler` + `OrdinalEncoder` + `RandomForestClassifier` |
| Output | Binary prediction (Churn / No Churn) + confidence probability |

### Why Random Forest?

Eight models were evaluated on this dataset. Here is the comparison:

| Model | Accuracy | Churn F1 |
|---|---|---|
| **Voting Classifier** | 81.71% | 0.63 |
| **Random Forest** ✅ | **81.37%** | **0.59** |
| Logistic Regression | 80.90% | 0.62 |
| SVM | 80.76% | 0.58 |
| AdaBoost | 80.76% | 0.60 |
| Gradient Boosting | 80.81% | 0.60 |
| KNN | 77.54% | 0.55 |
| Decision Tree | 72.51% | 0.50 |

Random Forest was chosen for its strong accuracy, interpretability, and robustness — it handles mixed feature types and noisy data well without heavy hyperparameter tuning.

---

## 🗂️ Project Structure

```
cus-churn/
├── app.py                          # Streamlit UI application
├── train_model.py                  # Model training script
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker container definition
├── .dockerignore                   # Files excluded from Docker image
├── render.yaml                     # Render deployment blueprint
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # IBM Telco dataset
├── model/
│   └── rf_pipeline.pkl             # Saved model pipeline (generated at build)
└── README.md
```

---

## 🚀 Running Locally

### Prerequisites
- Python 3.9+
- pip

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/olaoluwaidowu/cus-churn.git
cd cus-churn

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train and save the model
python train_model.py

# 4. Launch the app
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## 🐳 Running with Docker

```bash
# Build the image (also trains the model inside)
docker build -t churn-predictor .

# Run the container
docker run -p 8501:10000 churn-predictor
```

Open **http://localhost:8501** in your browser.

---

## ☁️ Deployment on Render

This project is configured for one-click deployment via **Render Blueprints**.

### Automatic (Blueprint)
1. Fork or push this repo to your GitHub
2. Go to [render.com](https://render.com) → **New → Blueprint**
3. Connect your GitHub repo — Render detects `render.yaml` automatically
4. Click **Apply** — done ✅

### Manual
1. Go to [render.com](https://render.com) → **New → Web Service**
2. Connect the GitHub repo
3. Set runtime to **Docker**
4. Render finds the `Dockerfile` automatically
5. Deploy

### How it works on Render

```
GitHub push (main branch)
       ↓
Render detects change  ← autoDeploy: true in render.yaml
       ↓
docker build
  → python:3.11-slim base image
  → pip install -r requirements.txt
  → python train_model.py   ← model baked into image at build time
       ↓
docker run
  → streamlit run app.py --server.port=10000
       ↓
🌐 Live at https://churn-predictor-group1.onrender.com
```

> The model is trained **at build time** and baked into the Docker image, so runtime startup is fast with no training overhead.

---

## 🖥️ Application Features

- **Prediction form** — enter customer details across 6 sections:
  - 👤 Personal Information
  - 📋 Account Details
  - 💰 Charges
  - 📞 Phone Services
  - 🌐 Internet Services
- **Prediction result** — Churn / No Churn with confidence percentage
- **Probability breakdown** — visual cards showing churn vs retention probability
- **Risk factor insights** — automatic analysis of key churn drivers for the customer
- **Recommended action** — tailored retention strategy based on risk level

---

## 📊 Dataset

**IBM Telco Customer Churn Dataset**  
Source: [IBM Sample Datasets / Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

- 7,043 customers · 21 features
- Target variable: `Churn` (Yes / No)
- Features include: tenure, contract type, internet service, payment method, monthly charges, and more

---

## 👥 Group

| | |
|---|---|
| Course | MIT807: Artificial Intelligence & Its Business Applications |
| Group | GROUP 1 |
| Project | Customer Churn Prediction |

---

## 📄 License

This project is submitted as academic coursework for MIT807. All rights reserved by GROUP 1.
