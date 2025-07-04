#!/bin/bash -e
if [ "$1" == "-h" ]; then
  echo "Usage: `basename $0` version gpu-id(s) data-path train-path trainer-script [trainer-args] [--host=EXTERNAL_IP]"
  echo "Example: `basename $0` 0.1 0 /data/huggingface/ ~/unsloth unsloth_trainer.py --host=10.2.23.35"
  echo "This script runs the unsloth-docker container with the specified version and GPU IDs."
  echo "The data path and train path are mounted into the container."
  echo "Trainer arguments can be passed after the paths."
  exit 0
fi

echo "unsloth-docker: starting verion $1 on GPU#$2"
export USER_HOME="/home/unsloth"
export HF_DATA="$USER_HOME/.cache/huggingface/"
export TRAIN_PATH="$USER_HOME/train"

# -v $PWD:$USER_HOME
docker run -it --rm \
    --user $(id -u):$(id -g) \
    --volume $3:$HF_DATA --volume $4:$TRAIN_PATH --volume ./src:$USER_HOME/src \
    --gpus device=$2 \
    --network=host \
    unsloth:$1 \
    python3 $USER_HOME/src/$5 \
    $6 $7 $8 $9
