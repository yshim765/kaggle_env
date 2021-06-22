import subprocess
from pathlib import Path

tmp_dir = Path(".").resolve().parts
tmp_dir = tmp_dir[:tmp_dir.index("kaggle") + 1]

ROOT = Path().joinpath(*tmp_dir)

subprocess.run(["pip", "install", "datasets", "--no-index", f"--find-links=file://{str(ROOT.absolute())}/input/kaggle_nlputils/packages/datasets"])
subprocess.run(["pip", "install", "seqeval", "--no-index", f"--find-links=file://{str(ROOT.absolute())}/input/kaggle_nlputils/packages/seqeval-1.2.2-py3-none-any.whl"])
subprocess.run(["pip", "install", "pysbd", "--no-index", f"--find-links=file://{str(ROOT.absolute())}//input/kaggle_nlputils/packages/pysbd-0.3.4-py3-none-any.whl"])
subprocess.run(["pip", "install", "fsspec", "--no-index", f"--find-links=file://{str(ROOT.absolute())}//input/kaggle_nlputils/packages/fsspec-2021.6.0-py3-none-any.whl"])
subprocess.run(["pip", "install", "pysbd", "--no-index", f"--find-links=file://{str(ROOT.absolute())}//input/kaggle_nlputils/packages/blingfire-0.1.7-py3-none-any.whl"])
