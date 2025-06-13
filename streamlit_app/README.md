# Healthcare Provider Fraud Detection Dashboard 🏥

A comprehensive Streamlit dashboard for detecting healthcare provider fraud using machine learning and explainable AI.

## Features

### 🔍 **Real-Time Fraud Prediction**
- Interactive form with 28 engineered features
- Instant fraud risk assessment with probability scores
- Color-coded risk levels (Low, Moderate, High)
- SHAP explanations for model transparency

### 📊 **Model Performance Insights**
- Comprehensive performance metrics (AUC, Precision, Recall, F1)
- Feature importance visualizations
- ROC curves and confusion matrices
- Validation results and charts

### 🧠 **Explainable AI**
- SHAP force plots for individual predictions
- Feature impact analysis
- Business-friendly explanations
- Transparent decision making

### 📈 **Project Showcase**
- Technical approach documentation
- Key insights and findings
- Portfolio demonstration
- Implementation details

## Quick Start

### Prerequisites
```bash
pip install streamlit pandas numpy scikit-learn xgboost shap matplotlib joblib
```

### Running the App
```bash
# From the project root directory
streamlit run streamlit_app/app2.py

# Or specify a custom port
streamlit run streamlit_app/app2.py --server.port 8502
```

### For Development
```bash
# Install requirements
pip install -r requirements.txt

# Run with auto-reload
streamlit run streamlit_app/app2.py --server.runOnSave true
```

## Data Structure

The app expects the following data structure:
```
streamlit_app/
├── data/
│   ├── models/
│   │   ├── xgb_classifier.pkl        # Trained XGBoost model
│   │   └── xgb_classifier.json       # Model configuration
│   ├── metrics/
│   │   └── XGBost_metrics.json       # Performance metrics
│   ├── validation_sets/
│   │   ├── X_train_full.parquet      # Training features
│   │   ├── shap_feature_importance.csv
│   │   ├── provider13_features.csv   # Example provider data
│   │   └── xgb_val_*.csv             # Validation results
│   ├── plots/
│   │   ├── roc_XGBoost.png          # ROC curve
│   │   ├── conf_matrix_XGBoost.png  # Confusion matrix
│   │   ├── shap_summary_bar.png     # SHAP summary
│   │   └── shap_*.png               # SHAP visualizations
│   └── eda/
│       └── provider_fraud_label_distribution.png
├── app2.py                          # Main Streamlit application
└── README.md                        # This file
```

## Features Overview

### Input Features (28 total)
The model analyzes providers across these categories:

**📊 Claim Patterns**
- `total_claims`, `inpatient_claims`, `outpatient_claims`
- `claims_per_bene`, `inpt_outpt_ratio`

**💰 Reimbursement Data** 
- `total_reimb`, `avg_reimb`, `median_reimb`, `std_reimb`, `max_reimb`
- `total_deductible`, `avg_deductible`

**👥 Patient Demographics**
- `unique_beneficiaries`, `avg_age`, `pct_deceased`, `pct_male`
- `race_diversity`

**🏥 Specialty & Procedures**
- `alzheimer_rate`, `heartfail_rate`, `kidney_rate`, `diabetes_rate`
- `avg_diag_diversity`, `avg_proc_diversity`

**📈 Behavioral Metrics**
- `pct_bene_multiclaim`, `avg_days_between_claims`
- `pct_high_value`, `pct_weekend`, `pct_all_diag_filled`

### Model Performance
- **AUC-ROC**: 0.947 (Excellent discrimination)
- **Precision**: 0.68 (68% of flagged providers are fraudulent)
- **Recall**: 0.76 (Catches 76% of fraudulent providers)
- **F1-Score**: 0.72 (Strong balance)

## Usage Guide

### 1. **Home Tab**
- Overview of the project and dataset
- Mission statement and approach
- Dataset statistics

### 2. **Predict Fraud Tab**
- Input provider data using the interactive form
- Features are grouped by category for easier navigation
- Get instant predictions with risk assessment
- View SHAP explanations for transparency

### 3. **Model Insights Tab**
- Review comprehensive performance metrics
- Explore feature importance rankings
- Analyze ROC curves and confusion matrices
- Understand model validation results

### 4. **Project Showcase Tab**
- Technical implementation details
- Key findings and business insights
- Portfolio demonstration
- Future improvements and next steps

## Technical Implementation

### Model Pipeline
1. **Data Processing**: Cleaned and engineered 28 features from raw claims data
2. **Model Training**: XGBoost classifier with hyperparameter tuning
3. **Validation**: Stratified splits with comprehensive evaluation metrics
4. **Deployment**: Streamlit interface with real-time predictions

### Key Technologies
- **Machine Learning**: XGBoost, Scikit-learn
- **Explainability**: SHAP (TreeExplainer)
- **Frontend**: Streamlit with custom CSS
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Streamlit charts

### Performance Optimizations
- Cached model loading with `@st.cache_resource`
- Efficient data structures using Parquet format
- Lazy loading of visualizations
- Responsive UI design

## Troubleshooting

### Common Issues

**Model Loading Error**
```python
# If you see XGBoost version warnings, this is normal
# The model will still work correctly
```

**Data Directory Not Found**
```bash
# Ensure you're running from the correct directory
cd /path/to/healthcarefraud_detection
streamlit run streamlit_app/app2.py
```

**Missing Dependencies**
```bash
# Install missing packages
pip install streamlit pandas numpy scikit-learn xgboost shap matplotlib joblib
```

**Port Already in Use**
```bash
# Use a different port
streamlit run streamlit_app/app2.py --server.port 8503
```

### Development Mode
```bash
# Run with debugging information
streamlit run streamlit_app/app2.py --logger.level=debug

# Run with auto-reload on file changes
streamlit run streamlit_app/app2.py --server.runOnSave=true
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Test the Streamlit app thoroughly
5. Commit your changes (`git commit -am 'Add new feature'`)
6. Push to the branch (`git push origin feature/improvement`)
7. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions about this project or to discuss machine learning applications in healthcare, feel free to reach out!

---

**Built with ❤️ using Streamlit, XGBoost, and SHAP**

*🔒 All predictions include explainable AI for transparency and regulatory compliance*
