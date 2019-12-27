import os
import pickle
import xgboost as xgb
import pandas as pd

from model_algorithm.data_models import TrainTestSet, IrisFeatures, Response


def train(train_test_split: TrainTestSet, num_round: int = 10, learning_rate: float = 0.01) -> xgb.XGBRegressor:
    bst = xgb.train(
        {"learning_rate": learning_rate, "objective": "reg:squarederror"},
        xgb.DMatrix(train_test_split.df_X_train, label=train_test_split.df_Y_train),
        num_round
    )
    return bst


def serialize(bst: xgb.XGBRegressor, output_folder: os.PathLike, model_id: str) -> os.PathLike:
    model_path = output_folder / f"{model_id}.pickle"
    with open(model_path, "wb") as outf:
        pickle.dump(bst, outf)
    return model_path


def deserialize(model_path: os.PathLike) -> xgb.XGBRegressor:
    with open(model_path, "rb") as inf:
        bst = pickle.load(inf)
    return bst


def predict(bst: xgb.XGBRegressor, input: IrisFeatures) -> Response:
    xgb_design = xgb.DMatrix(input.get_pandas_df())
    Y_hat = bst.predict(xgb_design)
    Y_hat = pd.DataFrame(Y_hat, columns=["target"])
    return Response.from_pandas_df(Y_hat)
