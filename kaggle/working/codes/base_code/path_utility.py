import shutil
from pathlib import Path


def set_path(COMPETITION_NAME: str, EXPERIMENT_NAME: str, RUN_NAME: str, is_use_output_root: bool = False) -> dict:
    PATH_DICT = {}
    
    PATH_DICT["ROOT"] = Path(".").resolve().parent
    PATH_DICT["INPUT_ROOT"] = PATH_DICT["ROOT"] / "input"
    PATH_DICT["RAW_DATA_DIR"] = PATH_DICT["INPUT_ROOT"] / COMPETITION_NAME
    PATH_DICT["WORK_DIR"] = PATH_DICT["ROOT"] / "working"
    PATH_DICT["OUTPUT_WORKDIR"] = PATH_DICT["WORK_DIR"] / "output"
    PATH_DICT["PROC_DATA"] = PATH_DICT["ROOT"] / "processed_data"

    PATH_DICT["OUTPUT_WORKDIR"].mkdir(parents=True, exist_ok=True)
    PATH_DICT["PROC_DATA"].mkdir(parents=True, exist_ok=True)

    # kaggle の code で回した時に output で表示する必要がない場合は
    # 以下のoutputも使用する
    if is_use_output_root:
        PATH_DICT["OUTPUT_ROOT"] = PATH_DICT["ROOT"] / "output"

        PATH_DICT["OUTPUT_ROOT"].mkdir(parents=True, exist_ok=True)

    PATH_DICT["SAVE_DIR"] = PATH_DICT["OUTPUT_WORKDIR"] / EXPERIMENT_NAME / RUN_NAME

    if PATH_DICT["SAVE_DIR"].exists():
        shutil.rmtree(PATH_DICT["SAVE_DIR"])
    PATH_DICT["SAVE_DIR"].mkdir(parents=True, exist_ok=True)

    return PATH_DICT
