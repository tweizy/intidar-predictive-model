import pandas as pd
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import uvicorn
import os

# --- CONFIGURATION ---
MODEL_NAME = "Production_Wait_Time_Model"

# --- 1. MODEL LIFECYCLE MANAGER ---
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model_path = "./production_model"
        print(f"🔄 Loading model from: {model_path} ...")
        
        ml_models["wait_time_model"] = mlflow.pyfunc.load_model(model_path)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not load model. {e}")
        ml_models["wait_time_model"] = None
    yield
    ml_models.clear()

app = FastAPI(title="Queue Wait Time Prediction API", version="1.0", lifespan=lifespan)

# --- 2. DEFINE INPUT SCHEMA ---
class PredictionRequest(BaseModel):
    # Context Features
    clinic_scale: str
    doctor_id: str
    
    # Queue State
    people_ahead_count: int = Field(..., ge=0)
    active_staff_count: int = Field(..., ge=1)
    current_delay_minutes: float = Field(..., ge=0.0)
    
    # Appointment Details
    appointment_type: str
    estimated_duration: int = Field(..., gt=0)
    
    # Computed Features
    rolling_avg_service_duration: float = Field(..., gt=0)
    no_show_rate_today: float = Field(..., ge=0.0, le=1.0)
    
    # Time Context
    day_of_week: int = Field(..., ge=0, le=6) # 0=Monday
    hour_of_day: int = Field(..., ge=0, le=23)
    is_weekend: int = Field(..., ge=0, le=1) # 0=No, 1=Yes (The missing field!)

    # Example for Swagger UI
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "clinic_scale": "medium_center",
                    "doctor_id": "doc_4",
                    "people_ahead_count": 3,
                    "active_staff_count": 2,
                    "current_delay_minutes": 10.5,
                    "appointment_type": "consultation",
                    "estimated_duration": 15,
                    "rolling_avg_service_duration": 18.5,
                    "no_show_rate_today": 0.1,
                    "day_of_week": 0,
                    "hour_of_day": 10,
                    "is_weekend": 0
                }
            ]
        }
    }

# --- 3. PREDICTION ENDPOINT ---
@app.post("/predict")
def predict_wait_time(request: PredictionRequest):
    model = ml_models.get("wait_time_model")
    if not model:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        # Convert Pydantic object to Pandas DataFrame
        input_data = pd.DataFrame([request.model_dump()])
        
        # Predict
        prediction = model.predict(input_data)[0]
        
        # Apply Business Logic / Guardrails
        safe_prediction = max(0, round(prediction * 1.1)) # 10% buffer
        
        return {
            "predicted_wait_minutes": safe_prediction,
            "raw_model_output": round(float(prediction), 2),
            "status": "success"
        }
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        # In dev, return the error. In prod, return a generic message.
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)