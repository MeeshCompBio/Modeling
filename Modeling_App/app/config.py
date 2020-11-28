import os
import pathlib

from model_algorithm.models import deserialize

MODEL_PATH = pathlib.Path(os.getenv("MODEL_PATH",
                                    "../models/TEST.pickle"
                                    )
                          )
MODEL = deserialize(MODEL_PATH)