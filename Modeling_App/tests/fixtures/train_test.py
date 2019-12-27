import pytest

import os

from model_algorithm.data import fetch_train_test_split
from model_algorithm.data_models import TrainTestSet


@pytest.fixture
def train_test() -> TrainTestSet:
    results = fetch_train_test_split(os.path.join(DATA_SOURCE_PATH, "features_10_000.pickle"))
    return results