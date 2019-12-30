from fastapi import APIRouter, Body
from model_algorithm.models import IrisFeatures, Response, predict

from app.config import MODEL

api = APIRouter()


@api.post("/predict", response_model=Response)
async def predict_endpoint(carrier_specific_features: IrisFeatures = Body(...,
        example={
                "sepal_length": [
                    4.3
                ],
                "sepal_width": [
                    3.6
                ],
                "petal_length": [
                    1.4
                ],
                "petal_width": [
                    0.2
                ]
                }
                )
                          ) -> Response:
    return predict(MODEL, carrier_specific_features)