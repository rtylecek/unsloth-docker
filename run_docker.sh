#!/bin/bash -e
if [ "$1" == "-h" ]; then
  echo "Usage: `basename $0` [version] [gpu id] [data path] [train path]"
  exit 0
fi

echo "unsloth-docker: starting v$1 on GPU#$2"
export USER_HOME="/home/unsloth"
export HF_DATA="$USER_HOME/.cache/huggingface/"
export TRAIN_PATH="$USER_HOME/train"

# -v $PWD:$USER_HOME
docker run -it --rm \
    --user $(id -u):$(id -g) \
    --volume $3:$HF_DATA --volume $4:$TRAIN_PATH \
    --gpus device=$2 \
    --network=host \
    unsloth:$1 \
    conda run -n unsloth_env \
    python3 unsloth_trainer.py \
    $5 $6 $7 $8 $9
