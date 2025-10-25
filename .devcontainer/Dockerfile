# syntax=docker/dockerfile:1.6
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
# ---- OS deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates tzdata locales \
    git openssh-client \
    ripgrep fd-find bat jq htop entr \
    software-properties-common gnupg \
  && add-apt-repository ppa:deadsnakes/ppa -y \
  && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-distutils \
  && rm -rf /var/lib/apt/lists/*

# Install pip for system Python
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
  && python3.11 /tmp/get-pip.py \
  && rm /tmp/get-pip.py
# Do NOT create /opt/venv; we’ll use a project-local .venv

# ---- Non-root user ----
ARG USERNAME=devuser
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid ${USER_GID} ${USERNAME} \
  && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} \
  && git config --system --add safe.directory '*' \
  && chsh -s /bin/bash ${USERNAME}

USER ${USERNAME}
WORKDIR /workspace

# Put project venv first on PATH (when it exists)
ENV PATH="/workspace/.venv/bin:${PATH}"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PRE_COMMIT_HOME="/home/${USERNAME}/.cache/pre-commit"

# Bootstrap script that:
#  - creates .venv if missing
#  - upgrades pip
#  - installs requirements.txt if present
COPY --chown=${USERNAME}:${USERNAME} docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Ensure that any interactive shell automatically prefers the project venv
RUN echo '\n# Activate project venv automatically\n' \
    'if [ -d /workspace/.venv/bin ]; then\n' \
    '  export PATH="/workspace/.venv/bin:$PATH"\n' \
    'fi\n' >> /home/devuser/.bashrc
    
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]

# Helper to run a local ssh-agent and load keys if provided
COPY --chown=${USERNAME}:${USERNAME} docker/ensure-ssh-agent.sh /usr/local/bin/ensure-ssh-agent.sh
RUN chmod +x /usr/local/bin/ensure-ssh-agent.sh
