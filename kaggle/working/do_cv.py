# %%
# スクリプト実行時の引数の設定
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-d", '--debug', action='store_true', default=False,
                    help='dubug flag. If you run it with this as an argument, it will run with debug settings.')
parser.add_argument("--number_of_cv", default=5,
                    help="number of cv")

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
import pandas as pd
from sklearn.model_selection import KFold
# from tqdm import tqdm

# %%
# Path の設定
COMPETITION_NAME = "indoor-location-navigation"

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

# with block を使う場合 end_run は要らない
if not args.debug:
    mlflow.start_run(run_name=RUN_NAME)

if mlflow.active_run():
    mlflow.set_tag("mlflow.note.content", RUN_DESCRIPTION)

# %%
# 実験コード
# 設定値など
seed = 777

os.environ["SEED"] = str(seed)

random.seed(seed)
np.random.seed(seed)

if mlflow.active_run():
    mlflow.log_param("random seed", seed)
    mlflow.log_param("numpy seed", seed)


# cv用のデータの作成
def make_cv_data(data, PROC_DATA, number_of_cv, seed):
    K_fold = KFold(n_splits=number_of_cv, shuffle=True, random_state=seed)

    for cv_phase_number, cv_data_index in enumerate(K_fold.split(data)):
        CV_DATA_DIR = PROC_DATA / 'cv_data' / str(cv_phase_number)
        
        if CV_DATA_DIR.exists():
            shutil.rmtree(CV_DATA_DIR)
            CV_DATA_DIR.mkdir(parents=True)
        else:
            CV_DATA_DIR.mkdir(parents=True)

        cv_data_index = {"train_index": cv_data_index[0], "val_index": cv_data_index[1]}
        with open(CV_DATA_DIR / "cv_data_index.pkl", "wb") as f:
            pickle.dump(cv_data_index, f)

    with open(PROC_DATA / 'cv_data' / "cv_data.pkl", "wb") as f:
        pickle.dump(data, f)


data = pd.DataFrame({
    "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "label": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
})

make_cv_data(data, PROC_DATA, args.number_of_cv, seed)


# 評価関数
def evaluate(pred, true_label):
    return 0


def train_model(data, train_data_index):
    return 0


def predict(model, data, val_data_index):
    tmp_data = data.copy()
    tmp_data = tmp_data.iloc[val_data_index]
    tmp_data["pred"] = tmp_data["label"]
    return tmp_data


# モデルの学習
# 予測結果の保存
# 評価
cv_evaluation = []

for cv_phase_number in range(args.number_of_cv):
    # cv用のデータの読み込み
    with open(PROC_DATA / 'cv_data' / str(cv_phase_number) / "cv_data_index.pkl", "rb") as f:
        cv_data_index = pickle.load(f)

    # モデルの学習
    model = train_model(data, cv_data_index["train_index"])

    # モデルでの予測、予測の保存
    result = predict(model, data, cv_data_index["val_index"])

    CV_PRED_DIR = PROC_DATA / 'cv_pred' / str(cv_phase_number)

    if CV_PRED_DIR.exists():
        shutil.rmtree(CV_PRED_DIR)
        CV_PRED_DIR.mkdir(parents=True)
    else:
        CV_PRED_DIR.mkdir(parents=True)

    with open(CV_PRED_DIR / "cv_pred.pkl", "wb") as f:
        pickle.dump(result, f)

    # 結果の評価
    cv_evaluation.append(evaluate(result["pred"], result["label"]))

    print(cv_data_index)

print("cv :", cv_evaluation)
print("mean cv :", np.mean(cv_evaluation))

# %%
if mlflow.active_run():
    mlflow.end_run()

# %%
