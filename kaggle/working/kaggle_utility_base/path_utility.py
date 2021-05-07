from pathlib import Path


def set_path(COMPETITION_NAME: str, is_use_output_root: bool = False) -> dict:
    PATH_DICT = {}
    
    PATH_DICT["ROOT"] = Path(".").resolve().parent
    PATH_DICT["INPUT_ROOT"] = PATH_DICT["ROOT"] / "input"
    PATH_DICT["RAW_DATA_DIR"] = PATH_DICT["INPUT_ROOT"] / COMPETITION_NAME
    PATH_DICT["WORK_DIR"] = PATH_DICT["ROOT"] / "working"
    PATH_DICT["OUTPUT_WORKDIR"] = PATH_DICT["WORK_DIR"] / "output"
    PATH_DICT["PROC_DATA"] = PATH_DICT["ROOT"] / "processed_data"

    if not PATH_DICT["OUTPUT_WORKDIR"].exists():
        PATH_DICT["OUTPUT_WORKDIR"].mkdir()
    if not PATH_DICT["PROC_DATA"].exists():
        PATH_DICT["PROC_DATA"].mkdir()

    # kaggle の code で回した時に output で表示する必要がない場合は
    # 以下のoutputも使用する
    if is_use_output_root:
        PATH_DICT["OUTPUT_ROOT"] = PATH_DICT["ROOT"] / "output"

        if not PATH_DICT["OUTPUT_ROOT"].exists():
            PATH_DICT["OUTPUT_ROOT"].mkdir()

    return PATH_DICT
