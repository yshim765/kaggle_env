# %%
from pathlib import Path


class KagglePath:
    def __init__(self, COMPETITION_NAME: str, is_use_output_root: bool = False) -> None:
        tmp_current_dir = Path(".").resolve().parts
        tmp_current_dir = tmp_current_dir[:tmp_current_dir.index("kaggle") + 1]

        self.ROOT = Path().joinpath(*tmp_current_dir)
        self.INPUT_ROOT = self.ROOT / "input"
        self.RAW_DATA_DIR = self.INPUT_ROOT / COMPETITION_NAME
        self.WORK_DIR = Path(".")
        self.OUTPUT_WORKDIR = self.WORK_DIR / "output"
        self.PROC_DATA = self.ROOT / "processed_data"

        self.OUTPUT_WORKDIR.mkdir(parents=True, exist_ok=True)
        self.PROC_DATA.mkdir(parents=True, exist_ok=True)

        # kaggle の code で回した時に output で表示する必要がない場合は
        # 以下のoutputも使用する
        if is_use_output_root:
            self.OUTPUT_ROOT = self.ROOT / "output"

            self.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


class Settings:
    def __init__(self, settings: dict):
        self.global_settings = settings["global_settings"]
        self.model_settings = settings["model_settings"]

        if settings.get("train_data_settings"):
            self.train_data_settings = settings["train_data_settings"]

        if settings.get("val_data_settings"):
            self.val_data_settings = settings["val_data_settings"]

        if settings.get("pred_data_settings"):
            self.pred_data_settings = settings["pred_data_settings"]
