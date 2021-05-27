# %%
import shutil
from pathlib import Path


class KagglePath:
    def __init__(self, COMPETITION_NAME: str, EXPERIMENT_NAME: str, RUN_NAME: str, is_use_output_root: bool = False) -> None:
        tmp_current_dir = Path(".").resolve().parts
        tmp_current_dir = tmp_current_dir[:tmp_current_dir.index("kaggle") + 1]

        self.ROOT = Path().joinpath(*tmp_current_dir)
        self.INPUT_ROOT = self.ROOT / "input"
        self.RAW_DATA_DIR = self.INPUT_ROOT / COMPETITION_NAME
        self.WORK_DIR = self.ROOT / "working"
        self.OUTPUT_WORKDIR = self.WORK_DIR / "output"
        self.PROC_DATA = self.ROOT / "processed_data"

        self.OUTPUT_WORKDIR.mkdir(parents=True, exist_ok=True)
        self.PROC_DATA.mkdir(parents=True, exist_ok=True)

        # kaggle の code で回した時に output で表示する必要がない場合は
        # 以下のoutputも使用する
        if is_use_output_root:
            self.OUTPUT_ROOT = self.ROOT / "output"

            self.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

        self.SAVE_DIR = self.OUTPUT_WORKDIR / EXPERIMENT_NAME / RUN_NAME

        if self.SAVE_DIR.exists():
            shutil.rmtree(self.SAVE_DIR)
        self.SAVE_DIR.mkdir(parents=True, exist_ok=True)
