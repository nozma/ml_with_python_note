---
title: "Pythonで始める機械学習の学習"
author: "R. Ito"
date: "2018-04-11"
site: bookdown::bookdown_site
documentclass: book
---

# まえおき {-}



やるぞい(๑•̀ㅂ•́)و✧

## 方針とか {-}

- [Pythonではじめる機械学習 ―scikit-learnで学ぶ特徴量エンジニアリングと機械学習の基礎](https://www.amazon.co.jp/dp/4873117984/)の備忘録です。
- <del>テキストのコードは見た目重視で冗長なので、どんどん省略していきます</del>。
    - そんな風に思っていた時期もありました。
    - `mglearn`によって過度にコードが省略されている部分や、それに起因してR Markdown上では動かない部分もあるため、適宜[amueller/introduction_to_ml_with_python: Notebooks and code for the book "Introduction to Machine Learning with Python"](https://github.com/amueller/introduction_to_ml_with_python)等を参照して不足部分を補っています。
- がんばりすぎない。

## 実行環境とか {-}

### サンプルコード実行用Jupyter notebook {-}

書籍のサンプルコードが動く&日本語が使えるように調整したJupyter notebookのDockerイメージを`nozma/ml-python-notebook`で公開しています([nozma/ml-python-notebook - Docker Hub](https://hub.docker.com/r/nozma/ml-python-notebook/))。

```bash
docker run -p 8888:8888 nozma/ml-python-notebook
```

などとやればJupyter notebookが起動すると思います。


### この文章を執筆しているRStudio {-}

この文章自体はR Studioとbookdownパッケージを用いて執筆しており、こちらの環境は`nozma/ml-python-notebook-r`で公開しています([nozma/ml-python-notebook-r - Docker Hub](https://hub.docker.com/r/nozma/ml-python-notebook-r/))。

Dockerがセットアップされている環境で

```bash
docker run -p 8787:8787 nozma/ml-python-notebook-r
```

とし、 http://localhost:8787 にアクセスするとR Studio Serverの起動画面が表示されます。ユーザー名、パスワードはいずれも`rstudio`です。

R Markdown中でPythonを使用するためには、Pythonのengine.pathを明示的にpython3と指定してやる必要があります。このテキストでは、.Rmdファイルに、以下のコードチャンクを設置してこの設定を行っています。

````markdown
```{r setup, echo=FALSE}
knitr::opts_chunk$set(
  engine.path = list(python = "/usr/bin/python3"),
  collapse = TRUE,
  comment = " ##"
  )
```
````

また、日本語フォントとしてIPAexGothicをインストールしてあります。matplotlibで使用する場合は、以下のコードチャンクをファイル冒頭などに記述してください。

````markdown
```{python}
matplotlib.rc('font', family='IPAexGothic') # 日本語プロット設定
```
````

### R sessionInfo {-}



```r
utils::sessionInfo()
 ## R version 3.4.3 (2017-11-30)
 ## Platform: x86_64-pc-linux-gnu (64-bit)
 ## Running under: Debian GNU/Linux 9 (stretch)
 ## 
 ## Matrix products: default
 ## BLAS: /usr/lib/openblas-base/libblas.so.3
 ## LAPACK: /usr/lib/libopenblasp-r0.2.19.so
 ## 
 ## locale:
 ##  [1] LC_CTYPE=ja_JP.UTF-8       LC_NUMERIC=C              
 ##  [3] LC_TIME=ja_JP.UTF-8        LC_COLLATE=ja_JP.UTF-8    
 ##  [5] LC_MONETARY=ja_JP.UTF-8    LC_MESSAGES=C             
 ##  [7] LC_PAPER=ja_JP.UTF-8       LC_NAME=C                 
 ##  [9] LC_ADDRESS=C               LC_TELEPHONE=C            
 ## [11] LC_MEASUREMENT=ja_JP.UTF-8 LC_IDENTIFICATION=C       
 ## 
 ## attached base packages:
 ## [1] stats     graphics  grDevices utils     datasets  base     
 ## 
 ## loaded via a namespace (and not attached):
 ##  [1] compiler_3.4.3  backports_1.1.2 bookdown_0.7    magrittr_1.5   
 ##  [5] rprojroot_1.3-2 tools_3.4.3     htmltools_0.3.6 yaml_2.1.18    
 ##  [9] Rcpp_0.12.16    stringi_1.1.7   rmarkdown_1.9   knitr_1.20     
 ## [13] methods_3.4.3   xfun_0.1        stringr_1.3.0   digest_0.6.15  
 ## [17] evaluate_0.10.1
```


### Python環境 {-}


```python
import sys
print(sys.version)
 ## 3.5.3 (default, Jan 19 2017, 14:11:04) 
 ## [GCC 6.3.0 20170118]
```


```python
from pip.utils import get_installed_distributions
[print(d) for d in get_installed_distributions()]
 ## wcwidth 0.1.7
 ## traitlets 4.3.2
 ## simplegeneric 0.8.1
 ## scipy 1.0.0
 ## scikit-learn 0.19.1
 ## pytz 2018.3
 ## python-dateutil 2.7.0
 ## pyparsing 2.2.0
 ## Pygments 2.2.0
 ## ptyprocess 0.5.2
 ## prompt-toolkit 1.0.15
 ## Pillow 5.0.0
 ## pickleshare 0.7.4
 ## pexpect 4.4.0
 ## parso 0.1.1
 ## pandas 0.22.0
 ## numpy 1.14.2
 ## mglearn 0.1.6
 ## matplotlib 2.2.2
 ## kiwisolver 1.0.1
 ## jedi 0.11.1
 ## ipython 6.2.1
 ## ipython-genutils 0.2.0
 ## graphviz 0.8.2
 ## decorator 4.2.1
 ## cycler 0.10.0
```
