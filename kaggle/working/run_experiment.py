# %%
# スクリプト実行時の引数の設定
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("global_settings", help='global settings json.')
parser.add_argument("model_settings", help='model settings json.')
parser.add_argument("train_data_settings", help='train data conf json.')
parser.add_argument("--val_data_settings", default=None, help='val data conf json.')
parser.add_argument("--test_data_settings", default=None, help='test data conf json.')
parser.add_argument("--do_val", action='store_true', default=False, help='do test data prediction')

args = parser.parse_args()

# %%
import json
import os
import pickle
import random
import shutil
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

# 実験パラメータの読み込み
with open(args.global_settings, "r") as f:
    global_settings = json.load(f)

tmp = import_module(global_settings["MODULE_DIR"])
Data = tmp.Data
Model = tmp.Model
evaluate = tmp.evaluate

# %%
# Path の設定
COMPETITION_NAME = global_settings["COMPETITION_NAME"]

ROOT = Path(".").resolve().parent
INPUT_ROOT = ROOT / "input"
RAW_DATA_DIR = INPUT_ROOT / COMPETITION_NAME
WORK_DIR = ROOT / "working"
OUTPUT_WORKDIR = WORK_DIR / "output"
PROC_DATA = ROOT / "processed_data"

if not OUTPUT_WORKDIR.exists():
    OUTPUT_WORKDIR.mkdir()
if not PROC_DATA.exists():
    PROC_DATA.mkdir()

# kaggle の code で回した時に output で表示する必要がない場合は
# 以下のoutputも使用する
# OUTPUT_ROOT = ROOT / "output"

# if not OUTPUT_ROOT.exists():
#     OUTPUT_ROOT.mkdir()

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

# 結果を保存するディレクトリの指定
SAVE_DIR = OUTPUT_WORKDIR / EXPERIMENT_NAME / RUN_NAME
if SAVE_DIR.exists():
    shutil.rmtree(SAVE_DIR)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

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

data_train = Data(train_data_settings)
data_train.read_data()

# データの前処理
data_train.preprocess_data()

# モデルの作成
with open(args.model_settings, "r") as f:
    model_settings = json.load(f)

model = Model(model_settings)

# モデルの学習
model.train(data_train)

with open(SAVE_DIR / "model.pkl", "wb") as f:
    pickle.dump(model, f)

# モデルでの予測、予測の保存
result_train = model.predict(data_train)

result_train.postprocess_data()

with open(SAVE_DIR / "result_train.pkl", "wb") as f:
    pickle.dump(result_train, f)

# 結果の評価
score = {}

score["score_train"] = evaluate(data_train, result_train)

# 検証データで検証する場合
if args.do_val:
    with open(args.val_data_settings, "r") as f:
        val_data_settings = json.load(f)

    data_val = Data(val_data_settings)
    data_val.read_data()
    data_val.preprocess_data()

    # モデルでの予測、予測の保存
    result_val = model.predict(data_val)

    result_val.postprocess_data()

    with open(SAVE_DIR / "result_val.pkl", "wb") as f:
        pickle.dump(result_val, f)

    score["score_val"] = evaluate(data_val, result_val)

with open(OUTPUT_WORKDIR / EXPERIMENT_NAME / RUN_NAME / "score.json", "w") as f:
    json.dump(score, f)

# print(global_settings)
# print(model_settings)
# print(train_data_settings)
# if args.do_val:
#     print(val_data_settings)
