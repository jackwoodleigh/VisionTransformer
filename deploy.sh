#!/bin/bash

HOST=$1
PORT=$2
FULL=${3:-"F"}
# scp -P 23865 root@ssh5.vast.ai:/workspace/proj/Logs/ModelSaves/model_save_.pth ~/Downloads/


if [ "$FULL" != "F" ]; then

  ssh -p ${PORT} root@${HOST} "mkdir -p /workspace/proj && mkdir -p /workspace/proj/data"
  #ssh -p ${PORT} root@${HOST} " pip install -r /workspace/proj/requirements.txt && mkdir -p /workspace/proj && mkdir -p /workspace/proj/data"

  ssh -p ${PORT} root@${HOST} "\
  pip install gdown && \
  gdown -O /workspace/proj/data.tar.gz "${FULL}" && \
  tar -xzvf /workspace/proj/data.tar.gz -C /workspace/proj/ && \
  rm /workspace/proj/data.tar.gz"

  scp -r -P ${PORT} test.tar.gz root@${HOST}:/workspace/proj/data
  ssh -p ${PORT} root@${HOST} "\
  tar -xzvf /workspace/proj/data/test.tar.gz -C /workspace/proj/data && \
  rm /workspace/proj/data/test.tar.gz"

fi

scp -r -P ${PORT} toolkit root@${HOST}:/workspace/proj
scp -P ${PORT} *.py root@${HOST}:/workspace/proj
scp -P ${PORT} requirements.txt root@${HOST}:/workspace/proj
scp -P ${PORT} config.yaml root@${HOST}:/workspace/proj
