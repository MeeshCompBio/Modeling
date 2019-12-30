# load data
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn import datasets
from itertools import chain

import os
import numpy as np
import pandas as pd


from model_algorithm.data_models import IrisFeatures, Response, TrainTestSet


def fetch_train_test_split(random_seed: int = 123456, n_splits: int = 10) -> TrainTestSet:
    iris = datasets.load_iris()
    data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                        columns= ['sepal_length',
                                  'sepal_width',
                                  'petal_length',
                                  'petal_width', 
                                  'target'
                                ] 
                        )

    # in case we wanted to subset other features, otherwise no chain
    general_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    features = list(chain(
        *[general_features, ['target']]))

    X, Y = data.loc[:, features[:-1]], data.loc[:, 'target']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=random_seed)
    X_train = IrisFeatures.from_pandas_df(X_train)
    X_test = IrisFeatures.from_pandas_df(X_test)
    Y_train = pd.DataFrame(Y_train, columns=["target"])
    Y_test = pd.DataFrame(Y_test, columns=["target"])
    Y_train = Response.from_pandas_df(Y_train)
    Y_test = Response.from_pandas_df(Y_test)
    return TrainTestSet(
        X_train=X_train,
        X_test=X_test,
        Y_train=Y_train,
        Y_test=Y_test
    )
