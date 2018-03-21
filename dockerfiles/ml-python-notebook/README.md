![docker pulls](https://img.shields.io/docker/pulls/nozma/ml-python-notebook.svg) ![docker stars](https://img.shields.io/docker/stars/nozma/ml-python-notebook.svg) ![Docker Build Status](https://img.shields.io/docker/build/nozma/ml-python-notebook.svg)

# Jupyter notebook for "Introduction to Machine Learning with Python"

「Pythonではじめる機械学習」のサンプルコードが実行できるように調整したJupyter notebookです。

## usage

`docker run -p 8888:8888 nozma/ml-python-notebook`

## 変更点

- 日本語関連
    - 環境変数の変更
    - IPAexフォントのインストール
- パッケージの追加
    - graphviz
    - mglearn

