import os
import numpy as np
from pathlib import Path

from model_algorithm.data_models import TrainTestSet
from model_algorithm.models import train, serialize, deserialize, predict
from tests.fixtures.train_test import train_test


def test_train_serialize_deserialize(tmpdir, train_test: TrainTestSet):
    model = train(train_test)
    model_id = "test"
    model_path = Path(tmpdir) / Path(model_id)
    model_path = serialize(bst=model, model_path=model_path)
    assert model_path.exists()
    model_deserialized = deserialize(model_path=model_path)

    y_hat = predict(model, train_test.X_train)
    y_hat_deserialized = predict(model_deserialized, train_test.X_train)

    assert np.all(np.isclose()