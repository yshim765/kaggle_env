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

    # データの前処理を行う
    def preprocess_data(self) -> None:
        data_preprocessed_train = self.data_raw_train.copy()
        data_preprocessed_test = self.data_raw_test.copy()

        self.data_preprocessed_train = data_preprocessed_train
        self.data_preprocessed_test = data_preprocessed_test

    # データの後処理を行う
    def postprocess_data(self) -> None:
        data_postprocess_train = self.data_raw_train.copy()
        data_postprocess_test = self.data_raw_test.copy()

        self.data_postprocess_train = data_postprocess_train
        self.data_postprocess_test = data_postprocess_test


# 評価関数
from sklearn.metrics import mean_squared_error


def evaluate(data_grand_truth: Data, data_predicted: Data) -> float:
    return mean_squared_error(data_grand_truth.raw_label, data_predicted.label_postprocessed)
