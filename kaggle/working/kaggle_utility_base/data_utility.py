import dataclasses
import pickle

import pandas as pd


# データを格納する汎用クラス
@dataclasses.dataclass
class Data:
    data_settings: dict
    raw_data: pd.DataFrame = dataclasses.field(default=None, init=False)
    data_preprocessed: pd.DataFrame = dataclasses.field(default=None, init=False)
    data_postprocessed: pd.DataFrame = dataclasses.field(default=None, init=False)
    raw_label: pd.DataFrame = dataclasses.field(default=None, init=False)
    label_preprocessed: pd.DataFrame = dataclasses.field(default=None, init=False)
    label_postprocessed: pd.DataFrame = dataclasses.field(default=None, init=False)

    # データの読み込みを行う
    def read_data(self) -> None:
        with open(self.data_settings["DATA_PATH"], "rb") as f:
            raw_data = pickle.load(f)

        if self.data_settings.get("LABEL_PATH"):
            with open(self.data_settings["LABEL_PATH"], "rb") as f:
                raw_label = pickle.load(f)
        else:
            raw_label = None

        self.raw_data = raw_data
        self.raw_label = raw_label

    # データの前処理を行う
    def preprocess_data(self) -> None:
        data_preprocessed = self.raw_data.copy()

        if self.raw_label is not None:
            label_preprocessed = self.raw_label.copy()
        else:
            label_preprocessed = None

        self.data_preprocessed = data_preprocessed
        self.label_preprocessed = label_preprocessed

    # データの後処理を行う
    def postprocess_data(self) -> None:
        data_postprocessed = self.raw_data.copy()

        if self.raw_label is not None:
            label_postprocessed = self.raw_label.copy()
        else:
            label_postprocessed = None

        self.data_postprocessed = data_postprocessed
        self.label_postprocessed = label_postprocessed


# 評価関数
from sklearn.metrics import mean_squared_error


def evaluate(data_grand_truth: Data, data_predicted: Data) -> float:
    return mean_squared_error(data_grand_truth.raw_label, data_predicted.label_postprocessed)
