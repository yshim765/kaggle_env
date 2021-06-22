import dataclasses
import pickle
from typing import List

from .utility import KagglePath


# ToDo: putorch Dataset に対応する
# データを格納する汎用クラス
@dataclasses.dataclass
class Data:
    data_settings: dict
    PATH: KagglePath

    # データの読み込みを行う
    def read_data(self) -> None:
        with open(self.PATH.RAW_DATA_DIR / self.data_settings["DATA_PATH"], "rb") as f:
            data = pickle.load(f)

        self.data = data

        if self.data_settings.get("LABEL_PATH"):
            with open(self.PATH.RAW_DATA_DIR / self.data_settings["LABEL_PATH"], "rb") as f:
                label = pickle.load(f)

            self.label = label

    def make_cv_data_generatore(self, num_cv: int) -> None:
        pass


class MetaData:
    def __init__(self) -> None:
        pass

    def make_meta_data(self, data_list: List[Data]) -> None:
        pass


# 前処理をするクラス
class DataProcessor():
    def __init__(self) -> None:
        pass

    def fit(self, data: Data, meta_data: MetaData) -> None:
        pass

    def transform(selaf, data: Data) -> Data:
        return Data


# 評価関数
from sklearn.metrics import mean_squared_error


def evaluate(data_grand_truth: Data, data_predicted: Data) -> float:
    return mean_squared_error(data_grand_truth.label, data_predicted.label)
