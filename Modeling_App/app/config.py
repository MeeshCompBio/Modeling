import os

import pathlib

from model_algorithm.models import deserialize

MODEL_PATH = pathlib.Path(os.getenv("MODEL_PATH", "./models/test_model.pickle"))

# TODO: make this a metaclass check in with sa-geocoder
MODEL = deserialize(MODEL_PATH)