# %%
# スクリプト実行時の引数の設定
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-d", '--debug', action='store_true', default=False,
                    help='dubug flag. If you run it with this as an argument, it will run with debug settings.')
parser.add_argument("global_settings", help='global settings json.')
parser.add_argument("model_settings", help='model settings json.')
parser.add_argument("train_data_settings", help='train data conf json.')

args = parser.parse_args()

# %%
import json
import os
import pickle
import random
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

try:
    import mlflow
except ModuleNotFoundError:
    subprocess.run(["pip", "install", "mlflow"])
    import mlflow

import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import KFold

# 実験パラメータの読み込み
with open(args.global_settings, "r") as f:
    global_settings = json.load(f)

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

# 全データの読み込み
with open(args.train_data_settings, "r") as f:
    train_data_settings = json.load(f)

with open(train_data_settings["DATA_PATH"], "rb") as f:
    data = pickle.load(f)

with open(train_data_settings["LABEL_PATH"], "rb") as f:
    label = pickle.load(f)

# cv用のデータの作成
K_fold = KFold(n_splits=global_settings["NUMBER_OF_CV"], shuffle=True, random_state=SEED)

for cv_phase_number, cv_index in tqdm(enumerate(K_fold.split(data)), total=global_settings["NUMBER_OF_CV"], desc="make data"):
    CV_DATA_DIR = PROC_DATA / 'cv_data' / str(cv_phase_number)

    if CV_DATA_DIR.exists():
        shutil.rmtree(CV_DATA_DIR)
    CV_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    cv_index = {"train_index": cv_index[0], "val_index": cv_index[1]}
    with open(CV_DATA_DIR / "cv_index.pkl", "wb") as f:
        pickle.dump(cv_index, f)

    with open(CV_DATA_DIR / "cv_data_train.pkl", "wb") as f:
        pickle.dump(data.loc[cv_index["train_index"], :], f)

    with open(CV_DATA_DIR / "cv_label_train.pkl", "wb") as f:
        pickle.dump(label.loc[cv_index["train_index"], :], f)

    with open(CV_DATA_DIR / "cv_data_val.pkl", "wb") as f:
        pickle.dump(data.loc[cv_index["val_index"], :], f)

    with open(CV_DATA_DIR / "cv_label_val.pkl", "wb") as f:
        pickle.dump(label.loc[cv_index["val_index"], :], f)

with open(PROC_DATA / 'cv_data' / "cv_data.pkl", "wb") as f:
    pickle.dump(data, f)

with open(PROC_DATA / 'cv_data' / "cv_label.pkl", "wb") as f:
    pickle.dump(label, f)

# cvの実行
cv_evaluation = []

for cv_phase_number in tqdm(range(global_settings["NUMBER_OF_CV"]), total=global_settings["NUMBER_OF_CV"], desc="do CV"):
    CV_DATA_DIR = PROC_DATA / 'cv_data' / str(cv_phase_number)

    tmp_global_settings = deepcopy(global_settings)
    tmp_global_settings["RUN_NAME"] = tmp_global_settings["RUN_NAME"] + f"_cv_{cv_phase_number}"

    with open(CV_DATA_DIR / "global_settings.json", "w") as f:
        json.dump(tmp_global_settings, f, ensure_ascii=False, indent=4, separators=(',', ': '))

    tmp_train_data_settings = {}
    tmp_train_data_settings["DATA_PATH"] = str(CV_DATA_DIR / "cv_data_train.pkl")
    tmp_train_data_settings["LABEL_PATH"] = str(CV_DATA_DIR / "cv_label_train.pkl")

    with open(CV_DATA_DIR / "train_data_settings.json", "w") as f:
        json.dump(tmp_train_data_settings, f, ensure_ascii=False, indent=4, separators=(',', ': '))

    tmp_val_data_settings = {}
    tmp_val_data_settings["DATA_PATH"] = str(CV_DATA_DIR / "cv_data_val.pkl")
    tmp_val_data_settings["LABEL_PATH"] = str(CV_DATA_DIR / "cv_label_val.pkl")

    with open(CV_DATA_DIR / "val_data_settings.json", "w") as f:
        json.dump(tmp_val_data_settings, f, ensure_ascii=False, indent=4, separators=(',', ': '))

    subprocess.run([
        "python",
        "run_experiment.py",
        str(CV_DATA_DIR / "global_settings.json"),
        args.model_settings,
        str(CV_DATA_DIR / "train_data_settings.json"),
        "--val_data_settings", str(CV_DATA_DIR / "val_data_settings.json"),
        "--do_val"
    ])

    with open(OUTPUT_WORKDIR / tmp_global_settings["EXPERIMENT_NAME"] / tmp_global_settings["RUN_NAME"] / "score.json", "r") as f:
        tmp_score = json.load(f)
    
    cv_evaluation.append(float(tmp_score["score_val"]))

print("cv :", cv_evaluation)
print("mean cv :", np.mean(cv_evaluation))

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
