import dataclasses
import pickle


# データを格納する汎用クラス
@dataclasses.dataclass
class Data:
    train_data_settings: dict
    test_data_settings: dict

    PATH_DICT: dict

    # データの読み込みを行う
    def read_data(self) -> None:
        with open(self.PATH_DICT["RAW_DATA_DIR"] / self.train_data_settings["DATA_PATH"], "rb") as f:
            data_train = pickle.load(f)

        with open(self.PATH_DICT["RAW_DATA_DIR"] / self.test_data_settings["DATA_PATH"], "rb") as f:
            data_test = pickle.load(f)

        self.data_train = data_train
        self.data_test = data_test


# 前処理をするクラス
class DataProcessor():
    def __init__(self) -> None:
        pass

    def fit(data: Data) -> None:
        pass

    def transform(data: Data) -> Data:
        return Data


# 評価関数
from sklearn.metrics import mean_squared_error


def evaluate(data_grand_truth: Data, data_predicted: Data) -> float:
    return mean_squared_error(data_grand_truth.raw_label, data_predicted.label_postprocessed)
