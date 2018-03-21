FROM jupyter/scipy-notebook

# Change environment to Japanese
USER root
RUN apt-get -y update && apt-get -y upgrade \
 && apt-get -y install language-pack-ja
ENV LANG ja_JP.UTF-8
ENV LC_ALL ja_JP.UTF-8
RUN sed -i '$d' /etc/locale.gen \
  && echo "ja_JP.UTF-8 UTF-8" >> /etc/locale.gen \
  && locale-gen ja_JP.UTF-8 \
  && /usr/sbin/update-locale LANG=ja_JP.UTF-8 LANGUAGE="ja_JP:ja"
RUN /bin/bash -c "source /etc/default/locale"
RUN ln -sf  /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
RUN apt-get -y install fonts-ipaexfont

# Install graphviz
RUN apt-get -y install graphviz
USER $NB_UID
RUN pip install --upgrade pip
RUN pip install graphviz

# Install other packages
RUN pip install mglearn

# clear fontcache
RUN rm -rf /home/jovyan/.cache/matplotlib/*
