# %%
# スクリプト実行時の引数の設定
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-d", '--debug', action='store_true', default=False,
                    help='dubug flag. If you run it with this as an argument, it will run with debug settings.')

args = parser.parse_args()

# %%
import os
import random
import subprocess
from pathlib import Path

try:
    import mlflow
except ModuleNotFoundError:
    subprocess.run(["pip", "install", "mlflow"])
    import mlflow

import numpy as np
import pandas as pd
from tqdm import tqdm

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

# デバッグ設定でない時はmlflowで実験を保存
if not args.debug:
    mlflow.start_run(run_name=RUN_NAME)

if mlflow.active_run():
    mlflow.set_tag("mlflow.note.content", RUN_DESCRIPTION)

# %%
# 実験コード
seed = 777

os.environ["SEED"] = str(seed)

random.seed(seed)
np.random.seed(seed)

if mlflow.active_run():
    mlflow.log_param("random seed", seed)
    mlflow.log_param("numpy seed", seed)

# %%
# start_runした時は最後にend_runする
if mlflow.active_run():
    mlflow.end_run()
