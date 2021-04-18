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


# データの作成
def make_data(data, PROC_DATA, seed):
    DATA_DIR = PROC_DATA / 'all_data'
    
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
        DATA_DIR.mkdir(parents=True)
    else:
        DATA_DIR.mkdir(parents=True)

    with open(DATA_DIR / "data.pkl", "wb") as f:
        pickle.dump(data, f)

    return data


data = pd.DataFrame({
    "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "label": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
})

processed_data = make_data(data, PROC_DATA, seed)


def train_model(data):
    return 0


# モデルの学習
model = train_model(processed_data)

OUTPUT_WORKDIR_MODEL = OUTPUT_WORKDIR / "model"

if OUTPUT_WORKDIR_MODEL.exists():
    shutil.rmtree(OUTPUT_WORKDIR_MODEL)
    OUTPUT_WORKDIR_MODEL.mkdir(parents=True)
else:
    OUTPUT_WORKDIR_MODEL.mkdir(parents=True)

with open(OUTPUT_WORKDIR_MODEL / "model.pkl", "wb") as f:
    pickle.dump(model, f)

print("train finished")

# %%
if mlflow.active_run():
    mlflow.end_run()

# %%
