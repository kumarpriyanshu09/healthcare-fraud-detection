# Healthcare Fraud Detection System

A machine learning-powered system for detecting fraudulent healthcare providers using claims data analysis and interactive visualization.

![Model Performance](data/plots/model_performance.png)

## 🌟 Key Features

- Interactive dashboard for fraud risk assessment
- Real-time predictions with explainable AI
- Model performance:
  - AUC-ROC: 0.947
  - Precision: 0.68
  - Recall: 0.76
  - F1-Score: 0.72

## 🛠️ Tech Stack

- Python 3.11
- Streamlit
- XGBoost
- SHAP (Explainable AI)
- Pandas, NumPy

## 📁 Project Structure

```
├── data/               # Data files
├── models/            # Trained models
├── notebooks/         # Jupyter notebooks
├── reports/          # Analysis reports
└── streamlit_app/    # Interactive dashboard
```

## ⚙️ Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/healthcare-fraud-detection.git
cd healthcare-fraud-detection
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Usage

Run the Streamlit application:
```bash
streamlit run streamlit_app/app2.py
```

The dashboard includes:
- 📦 Dataset Overview
- 🏗️ Feature Analysis
- 🤖 Model Performance
- 📊 SHAP Explainability
- 🕵️ Fraud Risk Predictor
- 💼 Business Impact

## 📊 Model Features

Analyzes 28 provider features across categories:
- Claim Patterns
- Reimbursement Data
- Patient Demographics
- Specialty & Procedures
- Behavioral Metrics

---
Built with ❤️ using Streamlit, XGBoost, and SHAP
