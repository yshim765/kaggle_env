# %%
from pathlib import Path

COMPETITION_NAME = "COMPETITION_NAME"

ROOT = Path(".").resolve().parent
INPUT_ROOT = ROOT / "input"
RAW_DATA_DIR = INPUT_ROOT / COMPETITION_NAME
WORK_DIR = ROOT / "working"
OUTPUT_WORKDIR = WORK_DIR / "output"
PROC_DATA = ROOT / "processed_data"

if not OUTPUT_WORKDIR.exists():
    OUTPUT_WORKDIR.mkdir()
if not PROC_DATA.exists():
    PROC_DATA.mkdir()

# kaggle の code で回した時に output で表示する必要がない場合は
# 以下のoutputも使用する
# OUTPUT_ROOT = ROOT / "output"

# if not OUTPUT_ROOT.exists():
#     OUTPUT_ROOT.mkdir()

# %%
