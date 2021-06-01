import pickle
import sys

import pandas as pd
# 例なので自分が使いたいモデルに変える
from lightgbm import LGBMRegressor

from .data import Data

sys.path.append("../../input/kaggle_utils")

from kaggle_utils import KagglePath


# モデルを格納するクラス
class Model:
    def __init__(self, model_settings: dict, PATH: KagglePath) -> None:
        self.model_settings = model_settings
        self.PATH = PATH

        # 例なので自分が使いたいモデルに変える
        self.model = LGBMRegressor(**model_settings["LIGHTGBM_PARAM"])


# モデルを学習させるクラス
class Trainer:
    def __init__(self, model: Model, PATH: KagglePath) -> None:
        self.model = model.model
        self.PATH = PATH

    # モデルの学習を行う
    def train(self, data_train: Data) -> None:
        self.model.fit(data_train.data, data_train.label)

        MODEL_SAVE_DIR = self.PATH.SAVE_DIR / "model"

        if not MODEL_SAVE_DIR.exists():
            MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

        with open(MODEL_SAVE_DIR / "model.pkl", "wb") as f:
            pickle.dump(self.model, f)


# モデルで推論を行うクラス
class Predictor:
    def __init__(self, model: Model, PATH: KagglePath) -> None:
        self.model = model.model
        self.PATH = PATH

    # 予測を行う
    def predict(self, data_pred: Data) -> Data:
        MODEL_SAVE_DIR = self.PATH.SAVE_DIR / "model"

        with open(MODEL_SAVE_DIR / "model.pkl", "wrb") as f:
            model = pickle.load(f)

        pred_test = pd.DataFrame(model.predict(data_pred.data), columns=["pred"])
        pred_test.to_csv(self.PATH.SAVE_DIR / "pred_result.csv")
