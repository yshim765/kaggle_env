# %%
# スクリプト実行時の引数の設定
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-d", '--debug', action='store_true', default=False,
                    help='dubug flag. If you run it with this as an argument, it will run with debug settings.')

args = parser.parse_args()

# %%
import os
import pickle
import random
import subprocess
import shutil
from pathlib import Path

try:
    import mlflow
except ModuleNotFoundError:
    subprocess.run(["pip", "install", "mlflow"])
    import mlflow

import numpy as np
import torch

from .kaggle_utility import Data
from .kaggle_utility import read_data
from .kaggle_utility import preprocess_data
from .kaggle_utility import Model

# %%
# Path の設定
COMPETITION_NAME = "compe_dir"

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
# 適宜変えること
EXPERIMENT_NAME = 'experiment001'
RUN_NAME = "test run"
RUN_DESCRIPTION = "test for mlflow runnning. this is written in notes"

mlflow.set_experiment(EXPERIMENT_NAME)

print(f"""
Experiment name : {EXPERIMENT_NAME}
Running file name : {__file__}
Run dexcription : {RUN_DESCRIPTION}
""")

if not args.debug:
    mlflow.start_run(run_name=RUN_NAME)

if mlflow.active_run():
    mlflow.set_tag("mlflow.note.content", RUN_DESCRIPTION)

# %%
# SEEDの設定
SEED = 777

os.environ["SEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

if mlflow.active_run():
    mlflow.log_param("seed", SEED)

# 実験コード

# CVの実施
# 全データの読み込み
raw_data = read_data()

# CV用データの作成
make_cv_data(raw_data, PROC_DATA, args.number_of_cv, SEED)

cv_evaluation = []

for cv_phase_number in range(args.number_of_cv):
    # cv用のデータの読み込み
    with open(PROC_DATA / 'cv_data' / str(cv_phase_number) / "cv_data_index.pkl", "rb") as f:
        cv_data_index = pickle.load(f)

    raw_data_cv_train = raw_data[cv_data_index["train_index"]]
    raw_data_cv_val = raw_data[cv_data_index["val_index"]]

    data_cv_train = preprocess_data(raw_data_cv_train)
    data_cv_val = preprocess_data(raw_data_cv_val)

    # モデルの読み込み
    model = Model({})

    # モデルの学習
    model.train(data_cv_train)

    # モデルでの予測、予測の保存
    result = model.predict(data_cv_val)

    CV_PRED_DIR = PROC_DATA / 'cv_pred' / str(cv_phase_number)

    if CV_PRED_DIR.exists():
        shutil.rmtree(CV_PRED_DIR)
        CV_PRED_DIR.mkdir(parents=True)
    else:
        CV_PRED_DIR.mkdir(parents=True)

    with open(CV_PRED_DIR / "cv_pred.pkl", "wb") as f:
        pickle.dump(result, f)


# 評価関数
def evaluate(data: Data):
    return 0

    # 結果の評価
    cv_evaluation.append(evaluate(result))

    print(cv_data_index)

print("cv :", cv_evaluation)
print("mean cv :", np.mean(cv_evaluation))

if mlflow.active_run():
    mlflow.log_metrics("cv", cv_evaluation)

if mlflow.active_run():
    mlflow.log_metrics("mean cv", np.mean(cv_evaluation))

# %%
if mlflow.active_run():
    mlflow.end_run()

# %%
