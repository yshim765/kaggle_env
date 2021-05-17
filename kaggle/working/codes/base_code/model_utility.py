import pickle

import pandas as pd
# 例なので自分が使いたいモデルに変える
from lightgbm import LGBMRegressor

from .data_utility import Data


# モデルを格納するクラス
class Model:
    def __init__(self, model_settings: dict, PATH_DICT: dict) -> None:
        self.model_settings = model_settings
        self.PATH_DICT = PATH_DICT

        # 例なので自分が使いたいモデルに変える
        self.model = LGBMRegressor(**model_settings["LIGHTGBM_PARAM"])

    # モデルの学習を行う
    def train(self, data: Data) -> None:
        self.model.fit(data.data_preprocessed_train["data"], data.data_preprocessed_train["target"])

        pred_train = pd.DataFrame(self.model.predict(data.data_preprocessed_train["data"]), columns=["pred"])
        pred_train.to_csv(self.PATH_DICT["SAVE_DIR"] / "pred_train.csv")

        with open(self.PATH_DICT["SAVE_DIR"] / "model.pkl", "wb") as f:
            pickle.dump(self.model, f)

    # 予測を行う
    def predict(self, data) -> Data:
        pred_test = pd.DataFrame(self.model.predict(data.data_preprocessed_test["data"]), columns=["pred"])
        pred_test.to_csv(self.PATH_DICT["SAVE_DIR"] / "pred_test.csv")
