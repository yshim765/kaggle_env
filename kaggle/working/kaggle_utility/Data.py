import dataclasses
from pathlib import Path

import pandas as pd


# データを格納する汎用クラス
@dataclasses.dataclass
class Data:
    data: pd.DataFrame
    label: pd.DataFrame = None

    # pandasのDataFrameに
    def __getitem__(self, index):
        return Data(data=self.data.loc[index, :], label=self.label.loc[index, :] if self.label is not None else None)


# データの前処理を行う
def preprocess_data(data: Data) -> Data:
    data_preprocessed = data
    return data_preprocessed


# データの読み込みを行う
def read_data(dirs: Path = None) -> Data:
    data = pd.read_csv(dirs / "train.csv")
    label = pd.read_csv(dirs / "train_label.csv")
    return Data(data=data, label=label)
