"""
Module 3 - Combined: Retrain Model and Register ONNX Version
=============================================================

This script combines model retraining with ONNX conversion and MLflow registration.
It follows the Cloudera AI Inference pattern for deploying models to inference service.

Key differences from the original workflow:
1. Converts trained sklearn pipeline to ONNX format
2. Registers model directly via MLflow (not CML API)
3. Handles mixed numerical/categorical features properly for ONNX
4. Single script execution (no separate registration step)

Workflow:
1. Load original + new batch data
2. Combine into larger training set
3. Train RandomForest pipeline (with preprocessing)
4. Convert trained pipeline to ONNX
5. Infer model signature
6. Log and register ONNX model via MLflow

Output:
  - ONNX model registered in MLflow
  - Metadata saved for tracking
  
Next Step: Deploy to Cloudera AI Inference Service (via API)
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import mlflow
import mlflow.sklearn
import mlflow.onnx
import onnx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score
from mlflow.models import infer_signature

# ONNX conversion imports
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType

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
    # Define minimal config if imports fail - use only columns that exist in the data
    MODEL_CONFIG = {
        "random_state": 42,
        "categorical_features": ['job', 'marital', 'education', 'default', 'housing', 'loan',
                                 'contact', 'month', 'day_of_week', 'poutcome'],
        "numerical_features": ['age', 'duration', 'campaign', 'pdays', 'previous',
                              'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
                              'euribor3m', 'nr.employed']
    }
    def setup_mlflow(exp_name):
        mlflow.set_experiment(exp_name)


def convert_to_onnx(model_pipeline, sample_data, categorical_features, numerical_features):
    """
    Convert sklearn pipeline to ONNX format with proper type handling.
    
    Critical for Cloudera AI Inference: ONNX format enables deployment to the
    inference service with optimized runtime performance.
    
    Args:
        model_pipeline: Trained sklearn pipeline (preprocessor + classifier)
        sample_data: Sample input data (DataFrame) for type inference
        categorical_features: List of categorical feature names
        numerical_features: List of numerical feature names
    
    Returns:
        ONNX model object ready for MLflow logging
    """
    print("\n🔄 Converting sklearn pipeline to ONNX...")
    
    # Define initial types for ONNX conversion
    # Key: Mixed data types require different TensorTypes
    initial_types = []
    
    # Add numerical features as FloatTensorType
    for feature in numerical_features:
        if feature in sample_data.columns:
            initial_types.append((feature, FloatTensorType([None, 1])))
    
    # Add categorical features as StringTensorType
    # Note: OneHotEncoder in pipeline expects string inputs
    for feature in categorical_features:
        if feature in sample_data.columns:
            initial_types.append((feature, StringTensorType([None, 1])))
    
    print(f"   Input schema: {len(numerical_features)} numerical, {len(categorical_features)} categorical features")
    
    # Convert sklearn pipeline to ONNX
    onnx_model = convert_sklearn(
        model=model_pipeline,
        initial_types=initial_types,
        target_opset=12  # Use opset 12 for broad compatibility
    )
    
    # Log ONNX version as metadata
    mlflow.set_tag("onnx_version", onnx.__version__)
    mlflow.set_tag("conversion_method", "skl2onnx")
    mlflow.set_tag("model_type", "RandomForestClassifier_with_preprocessing")
    
    print(f"   ✅ ONNX conversion complete")
    print(f"   ONNX version: {onnx.__version__}")
    
    return onnx_model


def main():
    """
    Main combined retraining and registration pipeline.
    """
    print("=" * 80)
    print("Module 3 - RETRAIN AND REGISTER ONNX MODEL")
    print("=" * 80)
    print("This script trains a model and registers it in ONNX format for CAI Inference")
    print("-" * 80)

    # =================================================================
    # PHASE 1: SETUP AND DATA LOADING
    # =================================================================
    print("\n[PHASE 1] Setup MLflow and Load Data")
    print("-" * 80)
    
    # Setup MLflow experiment
    EXPERIMENT_NAME = "banking_onnx_retraining_pipeline"
    setup_mlflow(EXPERIMENT_NAME)
    print(f"✅ MLflow experiment: {EXPERIMENT_NAME}")
    
    # Load datasets
    try:
        original_data_path = os.path.join(BASE_DIR, "module1", "data", "bank-additional", "bank-additional-full.csv")
        new_batch_path = os.path.join(BASE_DIR, "outputs", "new_labeled_batch_01.csv")

        original_data = pd.read_csv(original_data_path, sep=";")
        new_batch = pd.read_csv(new_batch_path)
        print(f"✅ Loaded {len(original_data)} original records and {len(new_batch)} new records")
    except FileNotFoundError as e:
        print(f"❌ ERROR: Could not find required data files")
        print(f"   {e}")
        print("   Did you run '2_simulate_labeling_job.py' first?")
        sys.exit(1)

    # Combine datasets
    combined_data = pd.concat([original_data, new_batch]).reset_index(drop=True)
    print(f"✅ Combined training set: {len(combined_data)} records")

    # =================================================================
    # PHASE 2: DATA PREPARATION
    # =================================================================
    print("\n[PHASE 2] Prepare Features and Split Data")
    print("-" * 80)
    
    TARGET = 'y'
    categorical_features = MODEL_CONFIG.get('categorical_features')
    numerical_features = MODEL_CONFIG.get('numerical_features')
    
    print(f"   Feature configuration:")
    print(f"   - Numerical features: {len(numerical_features)}")
    print(f"   - Categorical features: {len(categorical_features)}")
    
    X = combined_data.drop(columns=[TARGET])
    y = combined_data[TARGET]

    # Split for validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=MODEL_CONFIG.get("random_state", 42), 
        stratify=y
    )
    
    print(f"✅ Train/test split: {len(X_train)} train, {len(X_test)} test samples")

    # =================================================================
    # PHASE 3: BUILD AND TRAIN PIPELINE
    # =================================================================
    print("\n[PHASE 3] Build and Train Model Pipeline")
    print("-" * 80)
    
    # Create preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create full model pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100, 
            random_state=MODEL_CONFIG.get("random_state", 42)
        ))
    ])

    print("   Training RandomForest pipeline...")
    model_pipeline.fit(X_train, y_train)
    print("   ✅ Training complete")

    # =================================================================
    # PHASE 4: EVALUATE MODEL
    # =================================================================
    print("\n[PHASE 4] Evaluate Model Performance")
    print("-" * 80)
    
    y_pred = model_pipeline.predict(X_test)
    f1 = f1_score(y_test, y_pred, pos_label='yes')
    acc = accuracy_score(y_test, y_pred)
    
    print(f"   Evaluation Metrics:")
    print(f"   - F1 Score: {f1:.4f}")
    print(f"   - Accuracy: {acc:.4f}")

    # =================================================================
    # PHASE 5: CONVERT TO ONNX AND REGISTER
    # =================================================================
    print("\n[PHASE 5] Convert to ONNX and Register via MLflow")
    print("-" * 80)
    
    # Start MLflow run
    with mlflow.start_run(run_name="retrain_onnx_v1") as run:
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id
        
        print(f"\n📊 MLflow Run Started:")
        print(f"   Run ID: {run_id}")
        print(f"   Experiment ID: {experiment_id}")
        
        # Log parameters
        mlflow.log_params({
            "n_estimators": 100,
            "random_state": MODEL_CONFIG.get("random_state", 42),
            "train_data_shape": combined_data.shape,
            "original_records": len(original_data),
            "new_batch_records": len(new_batch),
            "model_format": "onnx"
        })
        
        # Log metrics
        mlflow.log_metrics({
            "test_f1": f1,
            "test_accuracy": acc
        })
        
        # Infer model signature (critical for inference service)
        print("\n📝 Inferring model signature...")
        sample_input = X_train[:5]  # Use small sample for signature
        predictions = model_pipeline.predict(sample_input)
        model_signature = infer_signature(sample_input, predictions)
        print("   ✅ Model signature inferred")
        
        # Convert to ONNX
        onnx_model = convert_to_onnx(
            model_pipeline=model_pipeline,
            sample_data=X_train,
            categorical_features=categorical_features,
            numerical_features=numerical_features
        )
        
        # Register model name
        REGISTERED_MODEL_NAME = "BankingCampaignPredictor_ONNX"
        
        print(f"\n📦 Logging and registering ONNX model...")
        print(f"   Model name: {REGISTERED_MODEL_NAME}")
        
        # Log ONNX model to MLflow and register it
        # This single call does both logging and registration
        mlflow.onnx.log_model(
            onnx_model=onnx_model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            signature=model_signature
        )
        
        print("   ✅ ONNX model logged and registered in MLflow")
        
        # Add descriptive tags
        mlflow.set_tag("model_type", "onnx")
        mlflow.set_tag("framework", "sklearn_to_onnx")
        mlflow.set_tag("purpose", "retrained_model_for_cai_inference")
        mlflow.set_tag("data_version", "original_plus_batch_01")
        
        # Save run info for reference
        retrain_info = {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "experiment_name": EXPERIMENT_NAME,
            "registered_model_name": REGISTERED_MODEL_NAME,
            "model_format": "onnx",
            "f1_score": float(f1),
            "accuracy": float(acc),
            "training_samples": int(len(combined_data)),
            "onnx_version": onnx.__version__
        }
        
        outputs_dir = os.path.join(BASE_DIR, "outputs")
        os.makedirs(outputs_dir, exist_ok=True)
        output_path = os.path.join(outputs_dir, "retrain_onnx_info.json")
        
        with open(output_path, "w") as f:
            json.dump(retrain_info, f, indent=2)
        
        print(f"\n💾 Saved metadata to: {output_path}")

    # =================================================================
    # SUCCESS SUMMARY
    # =================================================================
    print("\n" + "=" * 80)
    print("✅ RETRAINING AND REGISTRATION COMPLETE!")
    print("=" * 80)
    print(f"\n📊 Model Performance:")
    print(f"   - F1 Score: {f1:.4f}")
    print(f"   - Accuracy: {acc:.4f}")
    print(f"\n📦 Model Registration:")
    print(f"   - Name: {REGISTERED_MODEL_NAME}")
    print(f"   - Format: ONNX")
    print(f"   - Run ID: {run_id}")
    print(f"\n🎯 Next Steps:")
    print(f"   1. Check MLflow UI: Registered Models > {REGISTERED_MODEL_NAME}")
    print(f"   2. Verify model version is registered")
    print(f"   3. Deploy to Cloudera AI Inference Service (via API)")
    print(f"   4. Use model_id and version from MLflow for deployment payload")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()