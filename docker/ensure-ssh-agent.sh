#!/usr/bin/env bash
set -euo pipefail

# Persist agent env so all shells can reuse it
ENV_FILE="/home/devuser/.ssh-agent-env"

start_agent() {
  eval "$(ssh-agent -s)" >/dev/null
  echo "export SSH_AUTH_SOCK=$SSH_AUTH_SOCK" > "$ENV_FILE"
  echo "export SSH_AGENT_PID=$SSH_AGENT_PID" >> "$ENV_FILE"
  chmod 600 "$ENV_FILE"
}

# If we have a stored agent and it's alive, reuse it
if [ -f "$ENV_FILE" ]; then
  # shellcheck source=/dev/null
  . "$ENV_FILE"
  if ! kill -0 "$SSH_AGENT_PID" >/dev/null 2>&1; then
    start_agent
  fi
else
  start_agent
fi

# Load keys passed as args (filenames under ~/.ssh)
for key in "$@"; do
  if [ -f "$HOME/.ssh/$key" ]; then
    # Read-only mount is fine; ssh-add only needs read access
    ssh-add "$HOME/.ssh/$key" || true
  fi
done

# Ensure all future interactive shells see the agent
grep -qF "$ENV_FILE" "$HOME/.bashrc" || echo '. "$HOME/.ssh-agent-env" 2>/dev/null || true' >> "$HOME/.bashrc"
