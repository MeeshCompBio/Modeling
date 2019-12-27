from fastapi import APIRouter
from model_algorithm.models import IrisFeatures, Response, predict

from app.config import MODEL

api = APIRouter()


@api.post("/predict", response_model=Response)
async def predict_endpoint(
    carrier_specific_features: IrisFeatures
) -> Response:
    return predict(MODEL, carrier_specific_features)