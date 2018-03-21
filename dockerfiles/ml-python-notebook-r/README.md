![docker pulls](https://img.shields.io/docker/pulls/nozma/ml-python-notebook-r.svg) ![docker stars](https://img.shields.io/docker/stars/nozma/ml-python-notebook-r.svg) ![Docker Build Status](https://img.shields.io/docker/build/nozma/ml-python-notebook-r.svg)

# RStudio for "Introduction to Machine Learning with Python"

「Pythonではじめる機械学習」のサンプルコードが実行できるように調整したRStudioです。

## usage

`docker run -p 8787:8787 nozma/ml-python-notebook-r`

## 変更点

- 日本語関連
    - 環境変数の変更
    - IPAexフォントのインストール
- Rパッケージの追加
    - reticulate
- Pythonモジュールの追加
    - pip3
    - numpy
    - scipy
    - matplotlib
    - scikit-learn
    - pandas
    - pillow
    - ipython
    - mglearn
    - graphviz
- graphvizのインストール
