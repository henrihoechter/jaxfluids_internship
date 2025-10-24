# syntax=docker/dockerfile:1.6

############################################
# Base OS
############################################
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Basic OS deps + git/ssh + some handy CLI tools
# (ripgrep/fd/bat/jq/htop/entr are optional but super useful in workflows)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates tzdata locales \
    git openssh-client \
    ripgrep fd-find bat jq htop entr \
    # For python 3.11 install (Ubuntu 22 ships 3.10 by default)
    software-properties-common gnupg && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-distutils && \
    rm -rf /var/lib/apt/lists/*

# Make a Python 3.11 venv and install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && \
    python3.11 /tmp/get-pip.py && rm /tmp/get-pip.py && \
    python3.11 -m venv /opt/venv && chown -R ${USER_UID}:${USER_GID} /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

############################################
# Non-root user
############################################
ARG USERNAME=devuser
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} && \
    # Allow git to treat the mounted workspace as safe
    git config --system --add safe.directory '*' && \
    chsh -s /bin/bash ${USERNAME}

############################################
# Python deps (minimal + your asks)
############################################
# - jax (CPU) -> default version from PyPI
# - plotly for go.Figure
# - tensorboard for jax.profiler traces
# - pytest for tests
# - pre-commit + black (formatter) + ruff (fast linter)
# - ipykernel to use the container as a Jupyter kernel in VS Code
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      jax \
      jaxtyping \
      plotly \
      tensorboard \
      pytest \
      pre-commit black ruff \
      ipykernel \
      pydantic pydantic-settings

############################################
# Pre-commit: install hooks if repo mounts / has config
############################################
# (Safe no-op if no .pre-commit-config.yaml yet.)
USER ${USERNAME}
WORKDIR /workspace

# Helpful: pre-create a cache dir for pre-commit to avoid permission hiccups
ENV PRE_COMMIT_HOME="/home/${USERNAME}/.cache/pre-commit"

# Default command (shell). Devcontainer.json will override the workspace mount.
CMD ["/bin/bash"]
