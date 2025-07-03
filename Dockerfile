# Start from the NVIDIA CUDA base image
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

# Set a fixed model cache directory.
ARG USR="unsloth"
ARG UID=1000
ARG GID=1000
ARG PIP_CACHE="/home/${USR}/.cache/pip"
ARG UV_CACHE="/home/${USR}/.cache/uv"
ENV TORCH_HOME="/home/${USR}/.cache/torch"
ENV CONDA_DIR="/home/${USR}/conda"

SHELL ["/bin/bash", "-c"]

# Install Python and necessary packages
RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && apt-get install -y --no-install-recommends \
    wget build-essential python3-pip python3-dev \
    git ca-certificates openssh-server \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create group and user only if they do not exist
RUN getent group ${USR} || groupadd -r ${USR} -g ${GID}
RUN id -u ${USR} &>/dev/null || useradd -l -u ${UID} -r -m -g ${USR} ${USR}
USER ${USR}
WORKDIR /home/${USR}

# install miniconda
RUN --mount=type=cache,mode=0755,uid=${UID},gid=${GID},target=${PIP_CACHE} \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py312_25.5.1-0-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p ${CONDA_DIR}

RUN wget -qO- https://astral.sh/uv/install.sh | sh
ENV UV_HTTP_TIMEOUT=600
ENV PATH=${CONDA_DIR}/bin:/home/${USR}/.local/bin:$PATH

# as described in the Unsloth.ai Github
RUN --mount=type=cache,mode=0755,uid=${UID},gid=${GID},target=${PIP_CACHE} \
    --mount=type=cache,mode=0755,uid=${UID},gid=${GID},target=${UV_CACHE} \
    uv pip install --system torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

RUN --mount=type=cache,mode=0755,uid=${UID},gid=${GID},target=${PIP_CACHE} \
    --mount=type=cache,mode=0755,uid=${UID},gid=${GID},target=${UV_CACHE} \
    uv pip install --system -U vllm --torch-backend=cu128 --extra-index-url https://wheels.vllm.ai/nightly

RUN --mount=type=cache,mode=0755,uid=${UID},gid=${GID},target=${PIP_CACHE} \
    --mount=type=cache,mode=0755,uid=${UID},gid=${GID},target=${UV_CACHE} \
    uv pip install --system unsloth unsloth_zoo bitsandbytes

# build xformers from source
RUN pip uninstall xformers -y
ADD --chown=${USR}:${USR} xformers_blackwell.sh /home/${USR}/xformers_blackwell.sh
RUN chmod +x /home/${USR}/xformers_blackwell.sh
RUN --mount=type=cache,mode=0755,uid=${UID},gid=${GID},target=${PIP_CACHE} \
    bash /home/${USR}/xformers_blackwell.sh && \
    rm -rf /home/${USR}/xformers_blackwell.sh

RUN --mount=type=cache,mode=0755,uid=${UID},gid=${GID},target=${PIP_CACHE} \
    --mount=type=cache,mode=0755,uid=${UID},gid=${GID},target=${UV_CACHE} \
    uv pip install --system -U triton>=3.3.1

RUN --mount=type=cache,mode=0755,uid=${UID},gid=${GID},target=${PIP_CACHE} \
    --mount=type=cache,mode=0755,uid=${UID},gid=${GID},target=${UV_CACHE} \
    uv pip install --system -U transformers==4.52.4

# extras
RUN --mount=type=cache,mode=0755,uid=${UID},gid=${GID},target=${PIP_CACHE} \
    --mount=type=cache,mode=0755,uid=${UID},gid=${GID},target=${UV_CACHE} \
    uv pip install --system matplotlib  && \
    uv pip install --system --no-deps trl peft accelerate bitsandbytes && \
    uv pip install --system autoawq tensorboard

# copy the fine-tuning script into the container
COPY ./unsloth_trainer.py /home/${USR}/unsloth_trainer.py

WORKDIR /home/${USR}
# reduce GPU memory usage 
ENV PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"
ENV CUDA_MODULE_LOADING="LAZY"

# endless running task to avoid container to be stopped
CMD [ "/bin/bash" , "-c", "tail -f /dev/null" ]
