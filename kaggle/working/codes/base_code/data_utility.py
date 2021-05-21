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
            data_raw_train = pickle.load(f)

        with open(self.PATH_DICT["RAW_DATA_DIR"] / self.test_data_settings["DATA_PATH"], "rb") as f:
            data_raw_test = pickle.load(f)

        self.data_raw_train = data_raw_train
        self.data_raw_test = data_raw_test


# 前処理をするクラス
class DataProcessor():
    def __init__(self) -> None:
        pass


# 評価関数
from sklearn.metrics import mean_squared_error


def evaluate(data_grand_truth: Data, data_predicted: Data) -> float:
    return mean_squared_error(data_grand_truth.raw_label, data_predicted.label_postprocessed)
