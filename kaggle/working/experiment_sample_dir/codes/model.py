import pickle

import pandas as pd
# 例なので自分が使いたいモデルに変える
from lightgbm import LGBMRegressor

from .data import Data
from .utility import KagglePath


# モデルを格納するクラス
class Model:
    def __init__(self, model_settings: dict, PATH: KagglePath) -> None:
        self.model_settings = model_settings
        self.PATH = PATH

    # モデルの作成
    def build(self, params):
        # 例なので自分が使いたいモデルに変える
        self.model = LGBMRegressor(**self.model_settings["LIGHTGBM_PARAM"])

    # モデルの保存
    def save(self, model_path=None):
        if model_path is None:
            model_path = self.PATH.OUTPUT_WORKDIR / self.model_settings["OUTPUT_MODEL_DIR"]

        if not model_path.exists():
            model_path.mkdir(parents=True, exists_ok=True)

        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
    
    # モデルの読み込み
    def load(self, model_path=None):
        if model_path is None:
            model_path = self.PATH.OUTPUT_WORKDIR / self.model_settings["OUTPUT_MODEL_DIR"]

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        

# モデルを学習させるクラス
class Trainer:
    def __init__(self, model: Model, PATH: KagglePath) -> None:
        self.model = model
        self.PATH = PATH

    # モデルの学習を行う
    def train(self, data_train: Data) -> None:
        self.model.model.fit(data_train.data, data_train.label)

        self.model.save()


# モデルで推論を行うクラス
class Predictor:
    def __init__(self, model: Model, PATH: KagglePath) -> None:
        self.model = model
        self.PATH = PATH

    # 予測を行う
    def predict(self, data_pred: Data) -> Data:
        self.model.load()
        pred_test = pd.DataFrame(self.model.model.predict(data_pred.data), columns=["pred"])
        pred_test.to_csv(self.PATH.OUTPUT_WORKDIR / "pred_result.csv")
