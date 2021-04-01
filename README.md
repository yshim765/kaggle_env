# kaggle_env

`sh kaggle_docker.sh` でdockerを起動して入る。
`docker attach` で入る場合、`exit`で出るとコンテナが停止する。
`attach` で入った時にログイン状態を抜けるには"control"を押した状態で"P、Q"を順番に押す。

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