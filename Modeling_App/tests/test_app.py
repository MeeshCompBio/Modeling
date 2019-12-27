import pytest

from starlette.testclient import TestClient

from model_algorithm.data_models import TrainTestSet

from tests.fixtures.client import client
from tests.fixtures.train_test import train_test


def test_example_predict(client: TestClient, train_test: TrainTestSet):
    response = client.post(
        "/predict",
        json=train_test.X_train.dict()
    )

    assert response.status_code == 200