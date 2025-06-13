import numpy as np
import pandas as pd
import os

print("=== SHAP Data Analysis ===")

# Check SHAP values
try:
    shap_vals = np.load('/workspaces/healthcare-fraud-detection/streamlit_app/data/validation_sets/shap_values.npy')
    print(f"SHAP values shape: {shap_vals.shape}")
    print(f"SHAP values range: {shap_vals.min():.3f} to {shap_vals.max():.3f}")
except:
    print("SHAP values file not found")

# Check feature importance
try:
    feature_imp = pd.read_csv('/workspaces/healthcare-fraud-detection/streamlit_app/data/validation_sets/shap_feature_importance.csv')
    print(f"\nFeature importance shape: {feature_imp.shape}")
    print("\nTop 10 features:")
    print(feature_imp.head(10))
except:
    print("Feature importance file not found")

# Check validation data
try:
    X_val = pd.read_parquet('/workspaces/healthcare-fraud-detection/streamlit_app/data/validation_sets/X_val_full.parquet')
    print(f"\nValidation features shape: {X_val.shape}")
    print(f"Feature columns: {list(X_val.columns)}")
except:
    print("Validation features not found")

# Check predictions
try:
    y_proba = pd.read_csv('/workspaces/healthcare-fraud-detection/streamlit_app/data/validation_sets/xgb_val_proba.csv')
    print(f"\nPredictions shape: {y_proba.shape}")
    print(f"Prediction range: {y_proba['xgb_proba'].min():.3f} to {y_proba['xgb_proba'].max():.3f}")
except:
    print("Predictions file not found")

# Check true labels
try:
    y_val = pd.read_csv('/workspaces/healthcare-fraud-detection/streamlit_app/data/validation_sets/y_val.csv')
    print(f"\nLabels shape: {y_val.shape}")
    print(f"Label distribution:\n{y_val['PotentialFraud'].value_counts()}")
except:
    print("Labels file not found")

print("\n=== File Availability ===")
files_to_check = [
    '/workspaces/healthcare-fraud-detection/streamlit_app/data/plots/shap_summary_bar.png',
    '/workspaces/healthcare-fraud-detection/streamlit_app/data/plots/shap_beeswarm.png',
    '/workspaces/healthcare-fraud-detection/streamlit_app/data/plots/shap_force_provider_13.png'
]

for file_path in files_to_check:
    exists = os.path.exists(file_path)
    print(f"{file_path}: {'✓' if exists else '✗'}")
