FROM rocker/verse

RUN mkdir -p /home/rstudio/.rstudio/monitored/user-settings/
RUN chown -R rstudio:rstudio /home/rstudio/.rstudio

# Change environment to Japanese(Character and DateTime)
USER root
RUN apt-get -y update && apt-get -y upgrade \
 && apt-get -y install task-japanese
ENV LANG ja_JP.UTF-8
ENV LC_ALL ja_JP.UTF-8
RUN sed -i '$d' /etc/locale.gen \
  && echo "ja_JP.UTF-8 UTF-8" >> /etc/locale.gen \
  && locale-gen ja_JP.UTF-8 \
  && /usr/sbin/update-locale LANG=ja_JP.UTF-8 LANGUAGE="ja_JP:ja"
RUN /bin/bash -c "source /etc/default/locale"
RUN ln -sf  /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

# Install pip and modules
RUN apt-get -y install \
    python3-pip \
    python3-tk \
    fonts-ipaexfont
RUN pip3 install numpy scipy matplotlib scikit-learn pandas pillow ipython mglearn

# Install R packages
RUN install2.r --error --deps TRUE reticulate

# Install graphviz
RUN apt-get -y install graphviz
RUN pip3 install graphviz

CMD ["/init"]
