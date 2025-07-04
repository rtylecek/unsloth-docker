# Unsloth Docker

A Dockerfile for Unsloth.ai with Blackwell GPU support

## Intro

Training LLMs is oftern limited by the available VRAM of your GPU or other resources like time. [Unsloth](https://github.com/unslothai/unsloth) is a great library that helps you train LLMs faster and with less memory. Based on their [benchmarks](https://github.com/unslothai/unsloth?tab=readme-ov-file#-performance-benchmarking) up to 2x faster and with up to 80% less memory.

The following examples shows a minimum **training code example** and a **Dockerfile** you can use in your environment to get started training your models faster.

## Prerequisites

Host must have

- Docker 
- CUDA 12.9

## How to Build the Docker Image

Recommended: Use `docker login` to pull the base image with `docker pull nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04` before building docker.

To build the Docker image for Unsloth with Blackwell GPU support, use the provided `build_docker.sh` script. This script will build the Docker image according to the specifications in the `Dockerfile`.

```sh
sh build_docker.sh
```

This will create a Docker image that you can use to run your training jobs. Make sure you have Docker and CUDA 12.9 installed as described in the prerequisites.

## Example: Minimum Trainer Code for Unsloth

The following example shows how to fine-tune your model with Unsloth. The code is put together based on the examples provided on [Unsloth Github](https://github.com/unslothai/unsloth).

[unsloth_trainer.py example](https://github.com/eightBEC/unsloth-docker/blob/main/unsloth_trainer.py)

## How to Run the Docker Container

After building the Docker image, you can start a container using the provided `run_docker.sh` script. This script will launch the container with the appropriate settings for Unsloth and Blackwell GPU support.

```sh
sh run_docker.sh <version> <gpu-id(s)> <data-path> <train-path> <trainer-script> [trainer-args] [--host=EXTERNAL_IP]
```

### Example usage:

```sh
sh run_docker.sh 0.1 0 /data/huggingface/ ~/unsloth unsloth_trainer.py --host=10.2.23.35
```

- `<version>`: Docker image version tag (e.g., 0.1)
- `<gpu-id(s)>`: GPU device ID(s) to use (e.g., 0 or 0,1)
- `<data-path>`: Path to your HuggingFace data directory on the host
- `<train-path>`: Path to your training directory on the host
- `<trainer-script>`: Python script from src folder to run (e.g., unsloth_trainer.py)
- `[trainer-args]`: (Optional) Additional arguments for your trainer script 
- `[--host=EXTERNAL_IP]`: (Optional) Host IP for distributed training or networking

This will start the container and allow you to begin training your models inside the Docker environment.

## Further information

The sample python and Dockerfiles can also be found on my [Github](https://github.com/eightBEC/unsloth-docker/tree/main).
For those of you interested in diving deeper, please refer to the [Unsloth Github](https://github.com/unslothai/unsloth) for the latest updates, models, etc. - This library is developing balzingly fast.
Blackwell support based on [official compatibility notes](https://github.com/unslothai/unsloth/tree/main/blackwell).
