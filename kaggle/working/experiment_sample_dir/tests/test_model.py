import shutil
import pytest

from lightgbm import LGBMRegressor

from codes.model import Model, Trainer, Predictor
from codes.data import Data
from codes.utility import KagglePath


@pytest.fixture
def model():
    PATH = KagglePath(COMPETITION_NAME="compe_dir")

    model_settings = {
        "OUTPUT_MODEL_DIR": "model",
        "MODEL_PARAM": {}
    }

    return Model(model_settings=model_settings, PATH=PATH)


@pytest.fixture
def PATH():
    return KagglePath(COMPETITION_NAME="compe_dir")


@pytest.fixture
def model_settings():
    model_settings = {
        "OUTPUT_MODEL_DIR": "model",
        "MODEL_PARAM": {}
    }

    return model_settings


@pytest.fixture
def data_train():
    PATH = KagglePath(COMPETITION_NAME="compe_dir")
    data_settings = {
        "DATA_PATH": "train_data.pkl",
        "LABEL_PATH": "train_label.pkl"
    }

    data_train = Data(data_settings=data_settings, PATH=PATH)

    data_train.read_data()
    
    return data_train


@pytest.fixture
def data_test():
    PATH = KagglePath(COMPETITION_NAME="compe_dir")
    data_settings = {
        "DATA_PATH": "train_data.pkl"
    }

    data_test = Data(data_settings=data_settings, PATH=PATH)

    data_test.read_data()

    return data_test


def test_model_build(model):
    model.build()

    assert type(model.model) == LGBMRegressor


def test_model_save(model, PATH, model_settings):
    model_dir_path = PATH.OUTPUT_WORKDIR / model_settings["OUTPUT_MODEL_DIR"]

    if model_dir_path.exists():
        shutil.rmtree(model_dir_path)

    model.build()

    model.save()

    model_path = model_dir_path / "model.pkl"

    assert model_path.exists()


def test_model_save_another_dir(model, PATH):
    model_dir_path = PATH.OUTPUT_WORKDIR / "another_model_dir"

    if model_dir_path.exists():
        shutil.rmtree(model_dir_path)

    model.build()

    model.save(model_dir_path)

    model_path = model_dir_path / "model.pkl"

    assert model_path.exists()


def test_model_load(model, PATH, model_settings):
    model_dir_path = PATH.OUTPUT_WORKDIR / model_settings["OUTPUT_MODEL_DIR"]

    if model_dir_path.exists():
        shutil.rmtree(model_dir_path)

    model.build()

    model.save()

    model.load()

    assert type(model.model) == LGBMRegressor


def test_model_load_another_dir(model, PATH):
    model_dir_path = PATH.OUTPUT_WORKDIR / "another_model_dir"

    if model_dir_path.exists():
        shutil.rmtree(model_dir_path)

    model.build()

    model.save(model_dir_path)

    model.load(model_dir_path)

    assert type(model.model) == LGBMRegressor


def test_trainer(model, PATH, data_train):
    model.build()

    trainer = Trainer(model, PATH)

    trainer.train(data_train)

    assert type(trainer.model.model) == LGBMRegressor


def test_predictor(model, PATH, data_test):
    model.build()

    predictor = Predictor(model, PATH)

    pred_result = predictor.predict(data_test)

    print(pred_result)

    assert len(pred_result) == len(data_test.data)
