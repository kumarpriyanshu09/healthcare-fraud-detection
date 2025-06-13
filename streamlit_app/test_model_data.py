import json
import os

# Test loading metrics files
metrics_dir = "/workspaces/healthcare-fraud-detection/streamlit_app/data/metrics"

for filename in ["lr_metrics.json", "dt_metrics.json", "rf_gridsearch_metrics.json", "XGBost_metrics.json"]:
    filepath = os.path.join(metrics_dir, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
            print(f"{filename}: {data}")
    else:
        print(f"Missing: {filepath}")

# Check plots directory
plots_dir = "/workspaces/healthcare-fraud-detection/streamlit_app/data/plots"
print(f"\nPlots available:")
for filename in os.listdir(plots_dir):
    if filename.endswith('.png'):
        print(f"  - {filename}")
