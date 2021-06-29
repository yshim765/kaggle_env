# kaggle_env

vscode で README.md があるディレクトリを開き、左下の `><` から `Reopen in Container`をする。

メインで使うディレクトリ
/kaggle  
├── input  
│   └── compe_dir  
└── working  

他のディレクトリを作成することも可能。  
（`/kaggle/preprocessed_data/` `/kaggle/working/output/` など）  

kaggleにsubmitした後に表示されるファイルは `kaggle/working` 下にあるやつだけらしい。  

* コンテナ内でjupyterを開いてブラウザで見たい場合↓
    * コンテナ内で `jupyter notebook --port 8000 --ip=0.0.0.0 --allow-root` を実行  