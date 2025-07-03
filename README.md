# Unsloth Docker

A Dockerfile for Unsloth.ai with Blackwell GPU support

## Intro

Training LLMs is oftern limited by the available VRAM of your GPU or other resources like time. [Unsloth](https://github.com/unslothai/unsloth) is a great library that helps you train LLMs faster and with less memory. Based on their [benchmarks](https://github.com/unslothai/unsloth?tab=readme-ov-file#-performance-benchmarking) up to 2x faster and with up to 80% less memory.

The following examples shows a minimum **training code example** and a **Dockerfile** you can use in your environment to get started training your models faster.

## Prerequisites

- Docker / Podman

### Example: Minimum Trainer Code for Unsloth

The following example shows how to fine-tune your model with Unsloth. The code is put together based on the examples provided on [Unsloth Github](https://github.com/unslothai/unsloth).

[unsloth_trainer.py example](https://github.com/eightBEC/unsloth-docker/blob/main/unsloth_trainer.py)

## Further information

The sample python and Dockerfiles can also be found on my [Github](https://github.com/eightBEC/unsloth-docker/tree/main).
For those of you interested in diving deeper, please refer to the [Unsloth Github](https://github.com/unslothai/unsloth) for the latest updates, models, etc. - This library is developing balzingly fast.
Blackwell support based in [official compatibility notes](https://github.com/unslothai/unsloth/tree/main/blackwell).
