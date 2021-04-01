# kaggle_env

kaggle のnotebookが動いている場所のディレクトリ構造↓

/kaggle
├── input
│   └── compe_dir
│       ├── test.csv
│       └── train.csv
├── lib
│   └── kaggle
│       └── gcp.py
├── src
│   └── script.ipynb <= notebook をsave するとできる
└── working
    └── __notebook_source__.ipynb   <= 動かしている notebook

他のディレクトリを作成することも可能。
（`/kaggle/preprocessed_data/` `/kaggle/working/output/` など）

commitした後に表示されるファイルは `kaggle/working` 下にあるやつだけらしい。