import pandas as pd
import numpy as np
import json
import os
import sys
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score

# Get the project root directory (CML jobs run from /home/cdsw)
try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    # In Jupyter notebooks, __file__ is not defined
    BASE_DIR = os.getcwd()

# Add parent directory for imports
script_dir = BASE_DIR
sys.path.append(BASE_DIR)
    
# Import from your project's helper scripts
try:
    from shared_utils import MODEL_CONFIG
    from helpers._training_utils import setup_mlflow
except ImportError:
    print("⚠️ Could not import shared_utils or helpers. Using defaults.")
    # Define minimal config if imports fail
    MODEL_CONFIG = {
        "random_state": 42,
        "categorical_features": ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome'],
        "numerical_features": ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    }
    def setup_mlflow(exp_name):
        mlflow.set_experiment(exp_name)


def main():
    """
    Main retraining pipeline.
    """
    print("--- Starting Retraining Job (Step 3) ---")

    # 1. Setup MLflow experiment
    # We'll use a specific experiment name for retraining jobs
    EXPERIMENT_NAME = "banking_retraining_pipeline"
    setup_mlflow(EXPERIMENT_NAME)
    
    # 2. Load both datasets
    try:
        # CML jobs execute from the project root directory (/home/cdsw)
        original_data_path = os.path.join(BASE_DIR, "module1", "data", "bank-additional", "bank-additional-full.csv")
        new_batch_path = os.path.join(BASE_DIR, "outputs", "new_labeled_batch_01.csv")

        original_data = pd.read_csv(original_data_path, sep=";")
        new_batch = pd.read_csv(new_batch_path)
        print(f"Loaded {len(original_data)} original records and {len(new_batch)} new records.")
    except FileNotFoundError:
        print("❌ ERROR: Could not find 'bank-additional-full.csv' or 'outputs/new_labeled_batch_01.csv'")
        print("Did you run '2_simulate_labeling_job.py' first?")
        sys.exit(1)

    # 3. Combine them into a new, larger training set
    combined_data = pd.concat([original_data, new_batch]).reset_index(drop=True)
    print(f"New combined training set has {len(combined_data)} records.")

    # 4. Define features (X) and target (y)
    TARGET = 'y'
    categorical_features = MODEL_CONFIG.get('categorical_features')
    numerical_features = MODEL_CONFIG.get('numerical_features')
    
    X = combined_data.drop(columns=[TARGET])
    y = combined_data[TARGET]

    # Split for validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=MODEL_CONFIG.get("random_state", 42), 
        stratify=y
    )

    # 5. Create a preprocessing pipeline
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 6. Create the full model pipeline
    model_v2_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=MODEL_CONFIG.get("random_state", 42)))
    ])

    # 7. Start the MLflow Run
    with mlflow.start_run(run_name="retrain_on_drift_v2") as run:
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id
        print(f"MLflow Run Started:")
        print(f"  Run ID: {run_id}")
        print(f"  Experiment ID: {experiment_id}")
        
        # 8. Train the new model
        print("Training model_v2 on combined data...")
        model_v2_pipeline.fit(X_train, y_train)
        print("Training complete.")

        # 9. Evaluate and log metrics
        y_pred = model_v2_pipeline.predict(X_test)
        f1 = f1_score(y_test, y_pred, pos_label='yes')
        acc = accuracy_score(y_test, y_pred)
        
        print(f"Evaluation Metrics:")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Accuracy: {acc:.4f}")
        
        mlflow.log_params({"n_estimators": 100, "train_data_shape": combined_data.shape})
        mlflow.log_metrics({"test_f1": f1, "test_accuracy": acc})
        
        # 10. Log the model to MLflow
        print("Logging model to MLflow...")
        mlflow.sklearn.log_model(
            sk_model=model_v2_pipeline,
            artifact_path="model", # This 'model' path is what the deploy script looks for
            registered_model_name=None 
        )
        
        # 11. Save run info for the next script (This is the critical part!)
        retrain_info = {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "experiment_name": EXPERIMENT_NAME,
            "model_name": "banking_campaign_predictor_v2", # New model name
            "f1_score": f1
        }
        
        outputs_dir = os.path.join(BASE_DIR, "outputs")
        os.makedirs(outputs_dir, exist_ok=True)
        output_path = os.path.join(outputs_dir, "retrain_run_info.json")
        with open(output_path, "w") as f:
            json.dump(retrain_info, f, indent=2)

    print(f"\nSaved retraining info to: {output_path}")
    print("--- Retraining Job Finished (Step 3) ---")

if __name__ == "__main__":
    main()
