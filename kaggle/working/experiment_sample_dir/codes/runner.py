import json
import os
import random
from logging import StreamHandler, FileHandler, basicConfig, DEBUG, getLogger, Formatter

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data.dataset import Subset
from tqdm import tqdm

from .data import Data, evaluate
from .model import Model, Predictor, Trainer
from .utility import KagglePath, Settings


class Runner():
    def __init__(self, logging=False):
        # 実験パラメータの読み込み
        with open("settings.json", "r") as f:
            self.SETTINGS = Settings(json.load(f))

        # Path の設定
        self.PATH = KagglePath(self.SETTINGS.global_settings["COMPETITION_NAME"])

        # SEEDの設定
        self.SEED = self.SETTINGS.global_settings["SEED"]

        os.environ["SEED"] = str(self.SEED)

        random.seed(self.SEED)
        np.random.seed(self.SEED)
        os.environ['PYTHONHASHSEED'] = str(self.SEED)
        torch.manual_seed(self.SEED)
        torch.cuda.manual_seed(self.SEED)
        torch.backends.cudnn.deterministic = True

        self.logging = logging

        if self.logging:
            EXPERIMENT_NAME = self.SETTINGS.global_settings["EXPERIMENT_NAME"]
            RUN_NAME = self.SETTINGS.global_settings["RUN_NAME"]
            RUN_DESCRIPTION = self.SETTINGS.global_settings["RUN_DESCRIPTION"]

            mlflow.set_tracking_uri(str(self.PATH.WORK_DIR / ".." / "mlruns"))
            mlflow.set_experiment(EXPERIMENT_NAME)

            mlflow.start_run(run_name=RUN_NAME)

            mlflow.set_tag("mlflow.note.content", RUN_DESCRIPTION)

            LOGFILE_PATH = self.PATH.OUTPUT_WORKDIR / "log.txt"

            if LOGFILE_PATH.exists():
                LOGFILE_PATH.unlink()

            format_str = '%(asctime)s@%(name)s %(levelname)s # %(message)s'
            basicConfig(level=DEBUG, format=format_str)
            stream_handler = StreamHandler()
            stream_handler.setFormatter(Formatter(format_str))
            file_handler = FileHandler(filename=str(LOGFILE_PATH))
            file_handler.setFormatter(Formatter(format_str))
            getLogger().addHandler(stream_handler)
            getLogger().addHandler(file_handler)

            self.logger = getLogger(__name__)

    def run(self) -> None:
        # 実験コード
        # モデルの作成
        model = Model(self.SETTINGS.model_settings, self.PATH)
        model.build()

        # モデルを学習させる場合
        if hasattr(self.SETTINGS, "train_data_settings"):
            # データの読み込み
            data_train = Data(self.SETTINGS.train_data_settings, self.PATH)
            data_train.read_data()

            # モデルの学習
            trainer = Trainer(model, self.PATH)
            trainer.train(data_train)

        # テストデータを予測する場合
        if hasattr(self.SETTINGS, "pred_data_settings"):
            # データの読み込み
            data_test = Data(self.SETTINGS.pred_data_settings, self.PATH)
            data_test.read_data()

            # モデルの学習
            predictor = Predictor(model, self.PATH)
            pred_result = predictor.predict(data_test)

            pred_result_df = pd.DataFrame(pred_result, columns=["pred"])

            pred_result_df.to_csv(self.PATH.OUTPUT_WORKDIR / "pred_result.csv", index=False)
        
        if self.logging:
            mlflow.end_run()

    def run_cv(self) -> None:
        data_train = Data(self.SETTINGS.train_data_settings, self.PATH)
        data_train.read_data()

        n_splits = self.SETTINGS.global_settings["NUMBER_OF_CV"]

        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=self.SEED)

        for cv_num, (train_index, val_index) in tqdm(enumerate(kfold.split(data_train)), total=n_splits):
            model = Model(self.SETTINGS.model_settings, self.PATH)
            model.build()

            # モデルを学習させる
            # データの読み込み
            data_train_cv = Subset(data_train, train_index)

            # モデルの学習
            trainer = Trainer(model, self.PATH)
            trainer.train(data_train_cv)

            # テストデータを予測する
            # データの読み込み
            data_val_cv = Subset(data_train, val_index)

            # モデルの学習
            predictor = Predictor(model, self.PATH)
            pred_result = predictor.predict(data_val_cv)

            if self.logging:
                self.logger.info(f"score cv{cv_num} : {evaluate(data_val_cv[:][1], pred_result)}")

        if self.logging:
            mlflow.end_run()
