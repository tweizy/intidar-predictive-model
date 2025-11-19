import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
DATA_FILE = "synthetic_clinic_data.csv"
EXPERIMENT_NAME = "Queue_Wait_Time_Optimization"
TOLERANCE_MINUTES = 10
RANDOM_STATE = 42


# --- 1. DATA LOADER ---
def load_data():
    """
    Loads data and performs specific cleanup to prevent data leakage.
    """
    try:
        df = pd.read_csv(DATA_FILE)
        print(f"✅ Loaded {len(df)} records from {DATA_FILE}.")
    except FileNotFoundError:
        raise Exception(f"❌ Data file '{DATA_FILE}' not found. Please run generate_synthetic_data.py first.")

    drop_cols = ["actual_wait_time"]
    
    # Create X (Features) and y (Target)
    X = df.drop([c for c in drop_cols if c in df.columns], axis=1)
    y = df["actual_wait_time"]
    
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# --- 2. EVALUATION METRICS ---
def calculate_metrics(y_true, y_pred):
    """
    Calculates standard regression metrics plus custom business metrics.
    """
    errors = np.abs(y_true - y_pred)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        # Business Metric: % of predictions that are "Good Enough" (within TOLERANCE_MINUTES mins)
        "tolerance_acc": np.mean(errors <= TOLERANCE_MINUTES) * 100
    }

# --- 3. EXPERIMENT RUNNER ---
def run_experiment(model_name, model_instance, preprocessor, X_train, X_test, y_train, y_test):
    """
    Runs a single model experiment, logs everything to MLflow, and returns artifacts.
    """
    run_name = f"{model_name}_Run"
    
    # Start MLflow run
    with mlflow.start_run(run_name=run_name) as run:
        print(f"\n🧪 Running Experiment: {model_name}...")
        
        # Build Pipeline: Preprocessing -> Model
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model_instance)
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_test)
        
        # Evaluate
        metrics = calculate_metrics(y_test, y_pred)
        print(f"   -> MAE: {metrics['mae']:.2f} min | Tolerance Acc: {metrics['tolerance_acc']:.1f}%")
        
        # --- MLFLOW LOGGING ---
        # 1. Log Identifiers
        mlflow.log_param("model_type", model_name)
        
        # 2. Log Hyperparameters (if available)
        if hasattr(model_instance, "get_params"):
            try:
                mlflow.log_params(model_instance.get_params())
            except Exception:
                pass # Skip if params are not serializable
            
        # 3. Log Performance Metrics
        mlflow.log_metrics(metrics)
        
        # 4. Log Feature Importance (Explainability) if supported
        if hasattr(model_instance, "feature_importances_"):
            # Extract feature names from preprocessor
            cat_features = preprocessor.transformers_[1][1].get_feature_names_out()
            num_features = preprocessor.transformers_[0][2] # Numerical cols
            all_features = list(num_features) + list(cat_features)
            
            # Get importance values
            importances = model_instance.feature_importances_
            
            # Create dictionary for logging (top 5)
            imp_dict = dict(zip(all_features, importances))
            sorted_imp = dict(sorted(imp_dict.items(), key=lambda item: item[1], reverse=True)[:5])
            mlflow.log_dict(sorted_imp, "feature_importance_top5.json")
        
        # 5. Log Model Artifact & Signature
        # The signature ensures the input schema is strictly enforced in production
        signature = infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(
            pipeline, 
            "model", 
            signature=signature,
            input_example=X_train.iloc[:5]
        )
        
        # Return metrics, pipeline, AND the run_id for tracking the champion
        return metrics, pipeline, run.info.run_id

def generate_analysis_report(model_uri, X_test, y_test):
    import mlflow.pyfunc
    
    print("\n📋 GENERATING DEEP ANALYSIS REPORT")
    print("="*60)
    
    # 1. Load the Champion Model
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    predictions = loaded_model.predict(X_test)
    
    # 2. Feature Importance (The "Why")
    # We extract the sklearn pipeline from the mlflow flavor
    pipeline = loaded_model._model_impl.sklearn_model
    model_step = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    
    # Get feature names
    try:
        cat_names = preprocessor.transformers_[1][1].get_feature_names_out()
        num_names = preprocessor.transformers_[0][2]
        feature_names = list(num_names) + list(cat_names)
        
        importances = model_step.feature_importances_
        feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feat_df = feat_df.sort_values(by='Importance', ascending=False)
        
        print("🔍 TOP 10 DRIVERS OF WAIT TIME:")
        print(feat_df.head(10).to_string(index=False))
    except Exception as e:
        print(f"Could not extract detailed features: {e}")

    # 3. Error Analysis (The "Where it Fails")
    # Create a dataframe of failures
    results = X_test.copy()
    results["Actual"] = y_test
    results["Predicted"] = predictions
    results["Error"] = results["Predicted"] - results["Actual"]
    results["AbsError"] = results["Error"].abs()
    
    print("\n⚠️ WORST 5 PREDICTIONS (Outliers):")
    # Show the cases where the model failed hardest
    print(results.sort_values("AbsError", ascending=False).head(5)[
        ["people_ahead_count", "active_staff_count", "current_delay_minutes", 
         "appointment_type", "Actual", "Predicted", "Error"]
    ].to_string(index=False))

    print("\n✅ BEST 5 PREDICTIONS (Spot On):")
    print(results.sort_values("AbsError", ascending=True).head(5)[
        ["people_ahead_count", "Actual", "Predicted"]
    ].to_string(index=False))
    print("="*60)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Setup Experiment
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Load Data
    X_train, X_test, y_train, y_test = load_data()
    
    # Define Preprocessing (Shared across all models to ensure fair comparison)
    categorical_cols = ["clinic_scale", "doctor_id", "appointment_type"]
    numerical_cols = [
        "people_ahead_count", "current_delay_minutes", "active_staff_count", 
        "estimated_duration", "rolling_avg_service_duration", "no_show_rate_today",
        "day_of_week", "is_weekend", "hour_of_day"
    ]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols), # Standard scaling for linear models
            ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    # Define Candidate Models (The "Challengers")
    # We compare simple baselines vs complex trees to justify the cost/complexity
    models_to_test = [
        # 1. Baseline: Always predicts the mean. (Sanity check)
        ("Baseline_Dummy", DummyRegressor(strategy="mean")),
        
        # 2. Linear: Checks if the relationship is simple (Wait = A*x + B*y)
        ("Linear_Ridge", Ridge(alpha=1.0)),
        
        # 3. Random Forest: Good default, handles non-linearity well
        ("Random_Forest", RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_STATE)),
        
        # 4. XGBoost: Gradient Boosting, typically best for tabular data
        ("XGBoost_Opt", xgb.XGBRegressor(
            n_estimators=500, 
            learning_rate=0.05, 
            max_depth=6, 
            n_jobs=-1,
            random_state=RANDOM_STATE
        ))
    ]
    
    best_mae = float("inf")
    best_model_name = None
    best_run_id = None
    
    print(f"🚀 Starting MLOps Pipeline for Experiment: {EXPERIMENT_NAME}")
    print("="*70)

    # Iterate through models
    for name, model in models_to_test:
        # UNPACKING FIX: Now expects 3 values
        metrics, _, run_id = run_experiment(name, model, preprocessor, X_train, X_test, y_train, y_test)
        
        # Champion Selection Logic (Lower MAE is better)
        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_model_name = name
            best_run_id = run_id
            
    print("="*70)
    print(f"🏆 CHAMPION MODEL: {best_model_name}")
    print(f"   Best MAE: {best_mae:.2f} minutes")
    print(f"   Run ID:   {best_run_id}")
    
    # --- MODEL REGISTRY (Production Promotion) ---
    # In a real CI/CD pipeline, this would happen only if accuracy > threshold
    print(f"\n📦 Registering '{best_model_name}' to MLflow Model Registry...")
    
    try:
        model_uri = f"runs:/{best_run_id}/model"
        mv = mlflow.register_model(model_uri, "Production_Wait_Time_Model")
        print(f"✅ Model Version {mv.version} is registered as 'Production_Wait_Time_Model'.")
        print(f"👉 Run 'mlflow ui' to visualize the comparison.")
    except Exception as e:
        print(f"⚠️ Model registration skipped (likely due to local store): {e}")

    model_uri = f"runs:/{best_run_id}/model"
    generate_analysis_report(model_uri, X_test, y_test)