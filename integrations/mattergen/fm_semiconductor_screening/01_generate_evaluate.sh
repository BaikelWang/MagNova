#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RESULTS_PATH="${RESULTS_PATH:-results/fm_semiconductor_test32}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_BATCHES="${NUM_BATCHES:-2}"
MAG_DENSITY="${MAG_DENSITY:-0.20}"
GUIDANCE_FACTOR="${GUIDANCE_FACTOR:-2.0}"
STABILITY_THRESHOLD="${STABILITY_THRESHOLD:-0.10}"
DEVICE="${DEVICE:-cuda}"
SKIP_GENERATION="${SKIP_GENERATION:-0}"
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
    echo "Local asset is missing or empty: ${path}" >&2
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

if ! command -v mattergen-generate >/dev/null 2>&1; then
  echo "mattergen-generate was not found. Activate the MatterGen environment." >&2
  exit 1
fi

[[ -s "${MODEL_PATH}/config.yaml" ]] || {
  echo "Missing model config: ${MODEL_PATH}/config.yaml" >&2
  exit 1
}
verify_asset "${MODEL_CKPT}" "${MODEL_SIZE}" "${MODEL_SHA256}" || {
  echo "Run ${SCRIPT_DIR}/00_download_assets.sh to repair the local checkpoint." >&2
  exit 1
}
verify_asset "${REFERENCE_PATH}" "${REFERENCE_SIZE}" "${REFERENCE_SHA256}" || {
  echo "Run ${SCRIPT_DIR}/00_download_assets.sh to repair the TRI2024 reference." >&2
  exit 1
}

mkdir -p "${RESULTS_PATH}"

if [[ "${SKIP_GENERATION}" == "1" ]]; then
  if [[ ! -s "${RESULTS_PATH}/generated_crystals.extxyz" ]]; then
    echo "Cannot skip generation; generated_crystals.extxyz is missing." >&2
    exit 1
  fi
  echo "Reusing existing structures in ${RESULTS_PATH}/generated_crystals.extxyz"
else
  echo "Generating $((BATCH_SIZE * NUM_BATCHES)) structures from local model ${MODEL_PATH}"
  mattergen-generate "${RESULTS_PATH}" \
    --model_path="${MODEL_PATH}" \
    --checkpoint_epoch=last \
    --batch_size="${BATCH_SIZE}" \
    --num_batches="${NUM_BATCHES}" \
    --properties_to_condition_on="{'dft_mag_density': ${MAG_DENSITY}}" \
    --diffusion_guidance_factor="${GUIDANCE_FACTOR}"
fi

echo "Relaxing and evaluating generated structures with MatterSim"
mattergen-evaluate \
  --structures_path="${RESULTS_PATH}/generated_crystals.extxyz" \
  --relax=True \
  --structure_matcher=disordered \
  --reference_dataset_path="${REFERENCE_PATH}" \
  --energy_correction_scheme=TRI2024 \
  --device="${DEVICE}" \
  --save_as="${RESULTS_PATH}/metrics.json" \
  --save_detailed_as="${RESULTS_PATH}/detailed_metrics.json" \
  --structures_output_path="${RESULTS_PATH}/relaxed.extxyz"

python "${SCRIPT_DIR}/prepare_candidates.py" \
  --results-dir "${RESULTS_PATH}" \
  --stability-threshold "${STABILITY_THRESHOLD}"

echo "Workflow complete. MatterSim-screened candidates are available at:"
echo "  ${RESULTS_PATH}/stable_candidates.csv"
echo "  ${RESULTS_PATH}/stable_cifs/"
