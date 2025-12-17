import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, roc_curve, auc)
from sklearn.utils import estimator_html_repr

import mlflow
import mlflow.sklearn
import dagshub

# KONFIGURASI KONEKSI
DAGSHUB_REPO_OWNER = os.environ.get("DAGSHUB_REPO_OWNER")
DAGSHUB_REPO_NAME = os.environ.get("DAGSHUB_REPO_NAME")

# Cek variabel penting
if not DAGSHUB_REPO_OWNER or not DAGSHUB_REPO_NAME:
    print("Error: Environment variables DAGSHUB_REPO_OWNER and DAGSHUB_REPO_NAME must be set.")
    sys.exit(1)

# LOGIKA AUTENTIKASI
if os.environ.get("MLFLOW_TRACKING_USERNAME") and os.environ.get("MLFLOW_TRACKING_PASSWORD"):
    print("Environment: CI/CD Detected. Using Manual Auth (No Browser).")
    uri = f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow"
    mlflow.set_tracking_uri(uri)
else:
    print("Environment: Local Detected. Using dagshub.init().")
    dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)

print(f"Tracking URI set to: {mlflow.get_tracking_uri()}")

# Set Experiment
mlflow.set_experiment("Diabetes_CI_Pipeline")

def run_advanced_tuning():
    print("Memulai Training...")
    try:
        df = pd.read_csv('diabetes_data_preprocessed.csv')
    except FileNotFoundError:
        print("Error: diabetes_data_preprocessed.csv not found!")
        sys.exit(1)
        
    X = df.drop('class', axis=1)
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Simple Hyperparameter Tuning
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50],
        'max_depth': [10]
    }
    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=2, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    if "MLFLOW_RUN_ID" in os.environ:
        print(f"Clearing existing MLFLOW_RUN_ID: {os.environ['MLFLOW_RUN_ID']} to force new DagsHub run.")
        del os.environ["MLFLOW_RUN_ID"]

    # Start Run Baru di DagsHub
    with mlflow.start_run() as run:
        # Simpan Run ID ke file
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)
        print(f"New DagsHub Run ID ({run.info.run_id}) saved to run_id.txt")

        # Log Params
        mlflow.log_params(best_params)
        
        # Predict & Log Metrics
        y_pred = best_model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred, average='weighted')
        }
        mlflow.log_metrics(metrics)

        # Log Model
        mlflow.sklearn.log_model(best_model, "model")
        
        # ARTEFAK
        # 1. Metric Info JSON
        with open("metric_info.json", "w") as f:
            json.dump(metrics, f, indent=4)
        mlflow.log_artifact("metric_info.json")

        # 2. Confusion Matrix PNG
        plt.figure(figsize=(6, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.savefig("confusion_matrix.png")
        plt.close()
        mlflow.log_artifact("confusion_matrix.png")

        # 3. Estimator HTML
        print("Generating estimator.html...")
        try:
            with open("estimator.html", "w", encoding="utf-8") as f:
                f.write(estimator_html_repr(best_model))
            mlflow.log_artifact("estimator.html")
        except Exception as e:
            print(f"Warning: Could not generate estimator.html: {e}")

        print("Training Selesai.")

if __name__ == "__main__":
    run_advanced_tuning()