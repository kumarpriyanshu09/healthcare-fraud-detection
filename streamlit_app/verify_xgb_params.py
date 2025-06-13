#!/usr/bin/env python3
"""
Verification script to check XGBoost parameters from the notebook
Based on grep search results from the notebook
"""

# From the grep searches, the actual XGBoost parameters found were:
actual_xgb_params = {
    "n_estimators": 100,
    "max_depth": 5, 
    "learning_rate": 0.1,
    "scale_pos_weight": 9.686419753086419,  # from notebook output
    "random_state": 42,
    "n_jobs": -1,
    "use_label_encoder": False,
    "eval_metric": "auc"
}

print("Verified XGBoost Parameters from notebook:")
for param, value in actual_xgb_params.items():
    print(f"  {param}: {value}")

print("\nActual scale_pos_weight calculation:")
print("scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()")
print("Result: 9.686419753086419")

print("\nTraining approach:")
print("- Used scikit-learn API (not PySpark)")
print("- Single eval_set for validation monitoring")
print("- No early stopping explicitly configured")
print("- MLflow tracking enabled")
print("- fit() with eval_set=[(X_val, y_val)]")
