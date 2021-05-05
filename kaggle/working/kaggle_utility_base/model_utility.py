import dataclasses
from copy import deepcopy

from .data_utility import Data

# 例なので自分が使いたいモデルに変える
from lightgbm import LGBMRegressor


# モデルを格納するクラス
@dataclasses.dataclass
class Model:
    def __init__(self, model_param: dict):
        # 例なので自分が使いたいモデルに変える
        self.model = LGBMRegressor(**model_param)

    # モデルの学習を行う
    def train(self, data_train: Data):
        self.model.fit(data_train.data, data_train.label)

    # 予測を行う
    def predict(self, data_pred: Data) -> Data:
        data_result = deepcopy(data_pred)
        data_result.label = self.model.predict(data_result.data)
        return data_result
