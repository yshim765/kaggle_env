# %%
import mlflow

import os
from random import random, randint
from pathlib import Path

# %%
if __name__ == "__main__":
    # PATHの設定
    COMPETITION_NAME = "COMPETITION_NAME"

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

    # 実験の設定

    # 実験名、この試行の説明などを設定
    # 適宜変えること
    EXPERIMENT_NAME = 'experiment001'
    RUN_NAME = "test run"
    RUN_DESCRIPTION = "test for mlflow runnning. this is written in notes"

    mlflow.set_experiment(EXPERIMENT_NAME)

    print("Experiment name : {}".format(EXPERIMENT_NAME))
    print("Running file name : {}".format(__file__))
    print("Run dexcription : {}".format(RUN_DESCRIPTION))

    # 実験コード

    # with block を使う場合 end_run は要らない
    with mlflow.start_run(run_name=RUN_NAME) as run:
        mlflow.set_tag("mlflow.note.content", RUN_DESCRIPTION)
    
        mlflow.log_param("param1", randint(0, 100))

        mlflow.log_metric("foo", random())
        mlflow.log_metric("foo", random() + 1)
        mlflow.log_metric("foo", random() + 2)

        with open(OUTPUT_WORKDIR / "test.txt", "w") as f:
            f.write("hello world!")

        mlflow.log_artifacts(OUTPUT_WORKDIR)
