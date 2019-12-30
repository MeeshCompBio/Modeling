# define how the data should look like going into the model

from typing import List
from pydantic import BaseModel
from pydantic import validate_model

import numpy as np
import pandas as pd

class PandasModel(BaseModel):
    def get_pandas_df(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self.dict())

    @classmethod
    def from_pandas_df(cls, df: pd.DataFrame) -> 'PandasModel':
        obj = df.to_dict(orient="list")
        m = cls.__new__(cls)
        values, fields_set, validation_error = validate_model(cls, obj)
        if validation_error:
            raise validation_error
        object.__setattr__(m, '__dict__', values)
        object.__setattr__(m, '__fields_set__', fields_set)

        return m


class IrisFeatures(PandasModel):
    sepal_length: List[float]
    sepal_width: List[float]
    petal_length: List[float]
    petal_width: List[float]


class Response(PandasModel):
    target: List[float]


class TrainTestSet(BaseModel):
    X_train: IrisFeatures
    X_test: IrisFeatures
    Y_train: Response
    Y_test: Response

    @property
    def df_X_train(self) -> pd.DataFrame:
        return self.X_train.get_pandas_df()

    @property
    def df_X_test(self) -> pd.DataFrame:
        return self.X_test.get_pandas_df()

    @property
    def df_Y_train(self) -> pd.DataFrame:
        return self.Y_train.get_pandas_df()

    @property
    def df_Y_test(self) -> pd.DataFrame:
        return self.Y_test.get_pandas_df()
