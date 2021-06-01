# スクリプト実行時の引数の設定
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("experiment_params", help='experiment params json.')

args = parser.parse_args()

import json
import os
import random
import sys
from importlib import import_module

import numpy as np
import torch

sys.path.append("../input/kaggle_utils")

from kaggle_utils import KagglePath, Settings

# 実験パラメータの読み込み
with open(args.experiment_params, "r") as f:
    SETTINGS = Settings(json.load(f))

tmp = import_module(SETTINGS.global_settings["MODULE_DIR"])
Data = tmp.Data
Model = tmp.Model
Trainer = tmp.Trainer
Predictor = tmp.Predictor
evaluate = tmp.evaluate
DataProcessor = tmp.DataProcessor

# 実験名、この試行の説明などを設定
EXPERIMENT_NAME = SETTINGS.global_settings["EXPERIMENT_NAME"]
RUN_NAME = SETTINGS.global_settings["RUN_NAME"]
RUN_DESCRIPTION = SETTINGS.global_settings["RUN_DESCRIPTION"]

# Path の設定
PATH = KagglePath(SETTINGS.global_settings["COMPETITION_NAME"], EXPERIMENT_NAME, RUN_NAME)

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
