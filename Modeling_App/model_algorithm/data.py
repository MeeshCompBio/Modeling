# load data
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn import datasets
from itertools import chain

import os
import numpy as np
import pandas as pd


from model_algorithm.data_models import IrisFeatures, Response, TrainTestSet

def fetch_train_test_split(input_folder: os.PathLike, random_seed: int = 123456, n_splits: int = 10) -> TrainTestSet:
    iris = datasets.load_iris()
    data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                        columns= ['sepal_width',
                                'selpal_height',
                                'petal_width',
                                'petal_height', 
                                'target'
                                ] 
                        )

    # in case we wanted to subset other features, otherwise no chain
    general_features = ['sepal_length', 'sepal_width', 'petal_length']
    features = list(chain(
        *[general_features, ['target']]))

    X, Y = data.loc[:, features[:-1]], data.loc[:, 'target']

    return TrainTestSet(
        X_train=X,
        Y_train=Y
    )