# %%
# スクリプト実行時の引数の設定
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("global_settings", help='global settings json.')
parser.add_argument("model_settings", help='model settings json.')
parser.add_argument("train_data_settings", help='train data conf json.')
parser.add_argument("--test_data_settings", default=None, help='test data conf json.')

args = parser.parse_args()

# %%
import json
import os
import random
from importlib import import_module

import numpy as np
import torch

# 実験パラメータの読み込み
with open(args.global_settings, "r") as f:
    global_settings = json.load(f)

tmp = import_module(global_settings["MODULE_DIR"])
Data = tmp.Data
Model = tmp.Model
set_path = tmp.set_path

# %%
# 実験名、この試行の説明などを設定
EXPERIMENT_NAME = global_settings["EXPERIMENT_NAME"]
RUN_NAME = global_settings["RUN_NAME"]
RUN_DESCRIPTION = global_settings["RUN_DESCRIPTION"]

# print(f"""
# Experiment name : {EXPERIMENT_NAME}
# Running file name : {__file__}
# Run dexcription : {RUN_DESCRIPTION}
# """)

# %%
# Path の設定
PATH_DICT = set_path(global_settings["COMPETITION_NAME"], EXPERIMENT_NAME, RUN_NAME)

ROOT = PATH_DICT["ROOT"]
INPUT_ROOT = PATH_DICT["INPUT_ROOT"]
RAW_DATA_DIR = PATH_DICT["RAW_DATA_DIR"]
WORK_DIR = PATH_DICT["WORK_DIR"]
OUTPUT_WORKDIR = PATH_DICT["OUTPUT_WORKDIR"]
PROC_DATA = PATH_DICT["PROC_DATA"]
SAVE_DIR = PATH_DICT["SAVE_DIR"]

# %%
# SEEDの設定
SEED = global_settings["SEED"]

os.environ["SEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# 実験コード
# データの読み込み
with open(args.train_data_settings, "r") as f:
    train_data_settings = json.load(f)

if args.test_data_settings:
    with open(args.train_data_settings, "r") as f:
        test_data_settings = json.load(f)
else:
    train_data_settings = None

data = Data(train_data_settings=train_data_settings, test_data_settings=test_data_settings, PATH_DICT=PATH_DICT)
data.read_data()

# データの前処理
data.preprocess_data()

# モデルの作成
with open(args.model_settings, "r") as f:
    model_settings = json.load(f)

model = Model(model_settings, PATH_DICT)

# モデルの学習
model.train(data)

# testデータの予測、後処理
if args.test_data_settings:
    model.predict(data)

# print(global_settings)
# print(model_settings)
# print(train_data_settings)
# if args.test_data_settings:
#     print(test_data_settings)
