# スクリプト実行時の引数の設定
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("experiment_params", help='experiment params json.')

args = parser.parse_args()

# %%
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
from importlib import import_module
from copy import deepcopy

try:
    import mlflow
except ModuleNotFoundError:
    subprocess.run(["pip", "install", "mlflow"])
    import mlflow

import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import KFold

sys.path.append("../input/kaggle_utils")

from kaggle_utils import KagglePath, Settings

# 実験パラメータの読み込み
with open(args.experiment_params, "r") as f:
    SETTINGS = Settings(json.load(f))

tmp = import_module(SETTINGS.global_settings["MODULE_DIR"])
Data = tmp.Data

# 実験名、この試行の説明などを設定
EXPERIMENT_NAME = SETTINGS.global_settings["EXPERIMENT_NAME"]
RUN_NAME = SETTINGS.global_settings["RUN_NAME"]
RUN_DESCRIPTION = SETTINGS.global_settings["RUN_DESCRIPTION"]

mlflow.set_experiment(EXPERIMENT_NAME)

print(f"""
Experiment name : {EXPERIMENT_NAME}
Running file name : {__file__}
Run dexcription : {RUN_DESCRIPTION}
""")

if SETTINGS.global_settings["IS_SAVE_MLFLOW"]:
    mlflow.start_run(run_name=RUN_NAME)

if mlflow.active_run():
    mlflow.set_tag("mlflow.note.content", RUN_DESCRIPTION)

# Path の設定
PATH = KagglePath(SETTINGS.global_settings["COMPETITION_NAME"], EXPERIMENT_NAME, RUN_NAME)

# SEEDの設定
SEED = SETTINGS.global_settings["SEED"]

os.environ["SEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# 全データの読み込み
data = Data(SETTINGS.train_data_settings, PATH)
data.read_data()

# cv用のデータの作成
K_fold = KFold(n_splits=SETTINGS.global_settings["NUMBER_OF_CV"], shuffle=True, random_state=SEED)

for cv_phase_number, cv_index in tqdm(enumerate(K_fold.split(data.data)), total=SETTINGS.global_settings["NUMBER_OF_CV"], desc="make data"):
    CV_DATA_DIR = PATH.PROC_DATA / f'cv_{cv_phase_number}'

    if CV_DATA_DIR.exists():
        shutil.rmtree(CV_DATA_DIR)
    CV_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    cv_index = {"train_index": cv_index[0], "val_index": cv_index[1]}
    with open(CV_DATA_DIR / "cv_index.pkl", "wb") as f:
        pickle.dump(cv_index, f)

    with open(CV_DATA_DIR / "cv_data_train.pkl", "wb") as f:
        pickle.dump(data.data[cv_index["train_index"]], f)

    with open(CV_DATA_DIR / "cv_label_train.pkl", "wb") as f:
        pickle.dump(data.label[cv_index["train_index"]], f)

    with open(CV_DATA_DIR / "cv_data_val.pkl", "wb") as f:
        pickle.dump(data.data[cv_index["val_index"]], f)

    with open(CV_DATA_DIR / "cv_label_val.pkl", "wb") as f:
        pickle.dump(data.label[cv_index["val_index"]], f)

with open(PATH.PROC_DATA / "cv_data.pkl", "wb") as f:
    pickle.dump(data.data, f)

with open(PATH.PROC_DATA / "cv_label.pkl", "wb") as f:
    pickle.dump(data.label, f)

# cvの実行
cv_evaluation = []

for cv_phase_number in tqdm(range(SETTINGS.global_settings["NUMBER_OF_CV"]), total=SETTINGS.global_settings["NUMBER_OF_CV"], desc="do CV"):
    CV_DATA_DIR = PATH.PROC_DATA / f'cv_{cv_phase_number}'

    tmp_global_settings = deepcopy(SETTINGS.global_settings)
    tmp_global_settings["RUN_NAME"] = tmp_global_settings["RUN_NAME"] + f"/cv_{cv_phase_number}"

    tmp_train_data_settings = {}
    tmp_train_data_settings["DATA_PATH"] = str(CV_DATA_DIR / "cv_data_train.pkl")
    tmp_train_data_settings["LABEL_PATH"] = str(CV_DATA_DIR / "cv_label_train.pkl")

    tmp_pred_data_settings = {}
    tmp_pred_data_settings["DATA_PATH"] = str(CV_DATA_DIR / "cv_data_val.pkl")
    tmp_pred_data_settings["LABEL_PATH"] = str(CV_DATA_DIR / "cv_label_val.pkl")

    tmp_settings = {
        "global_settings": tmp_global_settings,
        "model_settings": SETTINGS.model_settings,
        "train_data_settings": tmp_train_data_settings,
        "pred_data_settings": tmp_pred_data_settings
    }

    with open(CV_DATA_DIR / "settings.json", "w") as f:
        json.dump(tmp_settings, f, ensure_ascii=False, indent=4, separators=(',', ': '))

    subprocess.run([
        "python",
        "run_experiment.py",
        str(CV_DATA_DIR / "settings.json")
    ])

    # with open(PATH.OUTPUT_WORKDIR / tmp_global_settings["EXPERIMENT_NAME"] / tmp_global_settings["RUN_NAME"] / "score.json", "r") as f:
    #     tmp_score = json.load(f)
    
    # cv_evaluation.append(float(tmp_score["score_val"]))

# print("cv :", cv_evaluation)
# print("mean cv :", np.mean(cv_evaluation))

if mlflow.active_run():
    mlflow.log_dict({"cv": cv_evaluation}, "cv_evaluation.json")

if mlflow.active_run():
    mlflow.log_metric("mean cv", np.mean(cv_evaluation))

# %%
if mlflow.active_run():
    mlflow.log_artifact(args.global_settings)
    mlflow.log_artifact(args.model_settings)
    mlflow.log_artifact(args.train_data_settings)
    mlflow.log_artifact(__file__)

if mlflow.active_run():
    mlflow.end_run()
