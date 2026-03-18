#!/usr/bin/env bash

load_project_env() {
  local project_root="${1:?project root is required}"
  local env_file="${project_root}/dev.env"

  if [ -f "$env_file" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$env_file"
    set +a
  fi
}
