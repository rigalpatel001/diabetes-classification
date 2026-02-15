from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import logging
import pandas as pd

from src.save_load import load_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Diabetes Risk Prediction API")

try:
    model = load_model("model.joblib")
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error("Model loading failed.")
    raise RuntimeError("Model could not be loaded.") from e


class PatientData(BaseModel):
    Pregnancies: int = Field(..., ge=0)
    Glucose: float = Field(..., gt=0)
    BloodPressure: float = Field(..., gt=0)
    SkinThickness: float = Field(..., ge=0)
    Insulin: float = Field(..., ge=0)
    BMI: float = Field(..., gt=0)
    DiabetesPedigreeFunction: float = Field(..., ge=0)
    Age: int = Field(..., ge=0)


@app.get("/health")
def health_check():
    return {"status": "API is running"}


@app.post("/predict")
def predict(data: PatientData):
    try:
        input_df = pd.DataFrame([data.dict()])

        probability = model.predict_proba(input_df)[0][1]
        prediction = int(probability >= 0.4)

        risk_level = (
            "High Risk" if probability >= 0.7
            else "Moderate Risk" if probability >= 0.4
            else "Low Risk"
        )

        logger.info(f"Prediction made. Probability: {probability:.4f}")

        return {
            "prediction": prediction,
            "probability": round(float(probability), 4),
            "risk_level": risk_level
        }

    except Exception as e:
        logger.error("Prediction failed.")
        raise HTTPException(status_code=500, detail="Prediction failed.")
