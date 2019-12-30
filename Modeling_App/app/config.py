import os
import pathlib

from model_algorithm.models import deserialize

MODEL_PATH = pathlib.Path(os.getenv("MODEL_PATH",
                                    "../models/test_model.pickle"
                                    )
                          )
MODEL = deserialize(MODEL_PATH)