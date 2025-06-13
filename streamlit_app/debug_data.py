#!/usr/bin/env python3

import json
import os
import pandas as pd

print("=== Debug Data Loading ===")
print(f"Current directory: {os.getcwd()}")

# Check if data directory exists
print(f"Data directory exists: {os.path.exists('data')}")
if os.path.exists('data'):
    print(f"Data directory contents: {os.listdir('data')}")

# Test loading each file individually
files_to_check = [
    "data/features_full.parquet",
    "data/eda/eda_metrics.json", 
    "data/eda/counts_summary.json",
    "data/eda/eda_summary.json",
    "data/validation_sets/shap_feature_importance.csv",
    "data/validation_sets/provider13_features.csv",
    "data/eda/eda_top_outliers.csv"
]

for file_path in files_to_check:
    exists = os.path.exists(file_path)
    print(f"{file_path}: {'✅' if exists else '❌'}")
    
    if exists and file_path.endswith('.json'):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            print(f"  JSON keys: {list(data.keys())}")
        except Exception as e:
            print(f"  Error loading JSON: {e}")

# Test the load_data function logic
print("\n=== Testing Load Data Logic ===")
data = {}
errors = []

try:
    # Load main features data
    if os.path.exists("data/features_full.parquet"):
        data['features'] = pd.read_parquet("data/features_full.parquet")
        data['features']['Provider'] = data['features']['Provider'].astype(str)
        print("✅ Features loaded")
    else:
        errors.append("Missing: data/features_full.parquet")
    
    # Load EDA metrics
    if os.path.exists("data/eda/eda_metrics.json"):
        with open("data/eda/eda_metrics.json", 'r') as f:
            data['eda_metrics'] = json.load(f)
        print("✅ EDA metrics loaded")
            
    # Load counts summary
    if os.path.exists("data/eda/counts_summary.json"):
        with open("data/eda/counts_summary.json", 'r') as f:
            data['counts_summary'] = json.load(f)
        print("✅ Counts summary loaded")
            
    # Load EDA summary
    if os.path.exists("data/eda/eda_summary.json"):
        with open("data/eda/eda_summary.json", 'r') as f:
            data['eda_summary'] = json.load(f)
        print("✅ EDA summary loaded")
    
except Exception as e:
    print(f"❌ Error: {e}")
    errors.append(str(e))

print(f"\nData keys loaded: {list(data.keys())}")
print(f"Errors: {errors}")

# Test the dataset overview logic
if 'eda_summary' in data:
    print("\n=== Testing Dataset Overview Logic ===")
    train_data = data['eda_summary']['Train.csv']
    bene_data = data['eda_summary']['Train_Beneficiarydata.csv'] 
    inp_data = data['eda_summary']['Train_Inpatientdata.csv']
    out_data = data['eda_summary']['Train_Outpatientdata.csv']
    
    print(f"Train: {train_data['num_rows']:,} rows, {train_data['num_columns']} columns")
    print(f"Beneficiary: {bene_data['num_rows']:,} rows, {bene_data['num_columns']} columns")
    print(f"Inpatient: {inp_data['num_rows']:,} rows, {inp_data['num_columns']} columns")
    print(f"Outpatient: {out_data['num_rows']:,} rows, {out_data['num_columns']} columns")
    
    # Create the dataset table
    dataset_overview = pd.DataFrame({
        'Dataset': [
            '🏷️ Training Data',
            '👥 Beneficiary Data', 
            '🏥 Inpatient Claims',
            '🚑 Outpatient Claims'
        ],
        'Description': [
            'Provider fraud labels',
            'Patient demographics',
            'Hospital admissions', 
            'Ambulatory care'
        ],
        'Rows': [
            f"{train_data['num_rows']:,}",
            f"{bene_data['num_rows']:,}",
            f"{inp_data['num_rows']:,}",
            f"{out_data['num_rows']:,}"
        ],
        'Columns': [
            train_data['num_columns'],
            bene_data['num_columns'],
            inp_data['num_columns'],
            out_data['num_columns']
        ],
        'File': [
            'Train.csv',
            'Train_Beneficiarydata.csv',
            'Train_Inpatientdata.csv', 
            'Train_Outpatientdata.csv'
        ]
    })
    
    print("\n=== Dataset Overview Table ===")
    print(dataset_overview.to_string(index=False))
else:
    print("❌ EDA summary not found in data")
