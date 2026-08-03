#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MODEL_PATH="${MODEL_PATH:-checkpoints/dft_mag_density}"
MODEL_CKPT="${MODEL_PATH}/checkpoints/last.ckpt"
REFERENCE_PATH="${REFERENCE_PATH:-data-release/alex-mp/reference_TRI2024correction.gz}"
MODEL_SIZE=511777278
MODEL_SHA256="01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e"
REFERENCE_SIZE=901725196
REFERENCE_SHA256="3631b54625f2a5410fb83aab16fda78073a2a713e8457e3beec523d0682315f5"

verify_asset() {
  local path="$1"
  local expected_size="$2"
  local expected_sha256="$3"
  local actual_size

  if [[ ! -s "${path}" ]]; then
    echo "Required local asset is missing or empty: ${path}" >&2
    return 1
  fi
  actual_size="$(stat -c '%s' "${path}")"
  if [[ "${actual_size}" != "${expected_size}" ]]; then
    echo "Incorrect file size for ${path}: ${actual_size}; expected ${expected_size}" >&2
    return 1
  fi
  echo "${expected_sha256}  ${path}" | sha256sum --check --status || {
    echo "SHA-256 verification failed for ${path}" >&2
    return 1
  }
}

if ! git lfs version >/dev/null 2>&1; then
  echo "Git LFS is required to download the local model and reference dataset." >&2
  exit 1
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "This download must run inside a full git clone of microsoft/mattergen." >&2
  echo "A GitHub source ZIP cannot resolve Git LFS pointer files." >&2
  exit 1
fi

echo "Downloading the dft_mag_density checkpoint to ${MODEL_PATH}"
git lfs pull -I "${MODEL_PATH}/**" --exclude=""

echo "Downloading the TRI2024 reference dataset to ${REFERENCE_PATH}"
git lfs pull -I "${REFERENCE_PATH}" --exclude=""

[[ -s "${MODEL_PATH}/config.yaml" ]] || {
  echo "Missing model config: ${MODEL_PATH}/config.yaml" >&2
  exit 1
}
verify_asset "${MODEL_CKPT}" "${MODEL_SIZE}" "${MODEL_SHA256}"
verify_asset "${REFERENCE_PATH}" "${REFERENCE_SIZE}" "${REFERENCE_SHA256}"

echo "Local assets are ready:"
ls -lh "${MODEL_CKPT}" "${MODEL_PATH}/config.yaml" "${REFERENCE_PATH}"
