import json
import os
import random
import sys

import numpy as np
import torch

from .data import Data
from .model import Model, Trainer, Predictor
from .utility import KagglePath, Settings


class Runner():
    def __init__(self):
        pass

    def run(self) -> None:
        pass

    def run_cv(self, num_cv: int) -> None:
        pass



# 実験パラメータの読み込み
with open("settings.json", "r") as f:
    SETTINGS = Settings(json.load(f))

# 実験名、この試行の説明などを設定
EXPERIMENT_NAME = SETTINGS.global_settings["EXPERIMENT_NAME"]
RUN_NAME = SETTINGS.global_settings["RUN_NAME"]
RUN_DESCRIPTION = SETTINGS.global_settings["RUN_DESCRIPTION"]

# Path の設定
PATH = KagglePath(SETTINGS.global_settings["COMPETITION_NAME"])

# %%
# SEEDの設定
SEED = SETTINGS.global_settings["SEED"]

os.environ["SEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# 実験コード
# モデルの作成
model = Model(SETTINGS.model_settings, PATH)

# モデルを学習させる場合
if hasattr(SETTINGS, "train_data_settings"):
    # データの読み込み
    data_train = Data(SETTINGS.train_data_settings, PATH)
    data_train.read_data()

    # モデルの学習
    trainer = Trainer(model, PATH)
    trainer.train(data_train)

# テストデータを予測する場合
if hasattr(SETTINGS, "pred_data_settings"):
    # データの読み込み
    data_test = Data(SETTINGS.pred_data_settings, PATH)
    data_test.read_data()

    # モデルの学習
    predictor = Predictor(model, PATH)
    predictor.predict(data_test)
