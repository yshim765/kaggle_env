import shutil

from lightgbm import LGBMRegressor

from codes.model import Model
from codes.utility import KagglePath


def test_model_build():
    PATH = KagglePath(COMPETITION_NAME="compe_dir")
    model = Model(
        model_settings={
            "OUTPUT_MODEL_DIR": "model",
            "MODEL_PARAM": {}
        },
        PATH=PATH
    )

    model.build()

    assert type(model.model) == LGBMRegressor


def test_model_save():
    PATH = KagglePath(COMPETITION_NAME="compe_dir")

    model_settings = {
        "OUTPUT_MODEL_DIR": "model",
        "MODEL_PARAM": {}
    }

    model = Model(model_settings=model_settings, PATH=PATH)

    model_dir_path = PATH.OUTPUT_WORKDIR / model_settings["OUTPUT_MODEL_DIR"]

    if model_dir_path.exists():
        shutil.rmtree(model_dir_path)

    model.build()

    model.save()

    model_path = model_dir_path / "model.pkl"

    assert model_path.exists()
