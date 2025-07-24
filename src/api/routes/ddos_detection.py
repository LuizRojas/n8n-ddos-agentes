# src/api/routes/ddos_detection.py

from fastapi import APIRouter, HTTPException
from ..schemas.prediction_schemas import FeaturesInput, PredictionOutput
from ..services import ml_service # Importa o módulo de serviço

router = APIRouter()

@router.post("/predict_ddos", response_model=PredictionOutput, summary="Predict DDoS attack from network flow features")
async def predict_ddos_endpoint(features_data: FeaturesInput):
    """
    Receives a set of network flow features and returns a prediction
    (whether it's a DDoS attack or normal traffic) along with confidence.
    """
    try:
        # Chama a função de serviço para fazer a predição
        prediction_results = ml_service.predict_ddos_attack(features_data.dict())
        return PredictionOutput(**prediction_results)
    except Exception as e:
        # Captura exceções e retorna um erro HTTP
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")