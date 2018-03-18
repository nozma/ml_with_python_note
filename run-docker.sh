#/usr/bin/bash

docker run \
  -v ~/.rstudio-desktop/monitored/user-settings:/home/rstudio/.rstudio/monitored/user-settings \
  -v ~/.R/rstudio/keybindings:/home/rstudio/.R/rstudio/keybindings \
  -v $(pwd):/home/rstudio \
  -d -p 8787:8787 my-rstudio
