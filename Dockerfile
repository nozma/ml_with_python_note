FROM rocker/verse

RUN mkdir -p /home/rstudio/.rstudio/monitored/user-settings/
RUN chown -R rstudio:rstudio /home/rstudio/.rstudio
