#!/usr/bin/env bash
# Load the repository .env for shell entrypoints without overriding values
# already exported by the caller.

_llm_env_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_llm_env_file="${_llm_env_repo_root}/.env"

if [[ -f "${_llm_env_file}" ]]; then
    while IFS='=' read -r _llm_env_key _llm_env_value; do
        [[ "${_llm_env_key}" =~ ^[[:space:]]*# ]] && continue
        _llm_env_key="${_llm_env_key//[[:space:]]/}"
        [[ -z "${_llm_env_key}" ]] && continue
        if [[ -z "${!_llm_env_key+x}" ]]; then
            _llm_env_value="${_llm_env_value%$'\r'}"
            _llm_env_value="${_llm_env_value#\"}"
            _llm_env_value="${_llm_env_value%\"}"
            _llm_env_value="${_llm_env_value#\'}"
            _llm_env_value="${_llm_env_value%\'}"
            export "${_llm_env_key}=${_llm_env_value}"
        fi
    done < "${_llm_env_file}"
fi

unset _llm_env_repo_root _llm_env_file _llm_env_key _llm_env_value

