from copy import deepcopy

# 例なので自分が使いたいモデルに変える
from lightgbm import LGBMRegressor

from .data_utility import Data


# モデルを格納するクラス
class Model:
    def __init__(self, model_settings: dict):
        self.model_settings = model_settings

        # 例なので自分が使いたいモデルに変える
        self.model = LGBMRegressor(**model_settings["LIGHTGBM_PARAM"])

    # モデルの学習を行う
    def train(self, data_train: Data):
        self.model.fit(data_train.data_preprocessed, data_train.label_preprocessed)

    # 予測を行う
    def predict(self, data_pred: Data) -> Data:
        data_result = deepcopy(data_pred)
        data_result.raw_label = self.model.predict(data_pred.data_preprocessed)

        return data_result
