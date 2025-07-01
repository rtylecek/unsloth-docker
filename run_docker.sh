#!/bin/bash -e
if [ "$1" == "-h" ]; then
  echo "Usage: `basename $0` [version] [gpu id] [output path]"
  exit 0
fi

echo "unsloth-docker: starting v$1 on GPU#$2"
export HOST_DATA=$3
export USER_HOME="/home/unsloth"
export HF_DATA="$USER_HOME/.cache/huggingface/"

# -v $PWD:$USER_HOME
docker run -it --rm \
    --user $(id -u):$(id -g) \
    --volume $HOST_DATA:$HF_DATA \
    --gpus device=$2 \
    --network=host \
    unsloth:$1 \
    python3 unsloth_trainer.py \
    $3 $4 $5 $6 $7 $8 $9
