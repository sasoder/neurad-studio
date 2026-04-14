#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Build the post-render stereo Difix -> LUT assets for detector training.

This script:
1. Runs Difix over cached stereo RGB renders into:
   - stereo_front_left_difix
   - stereo_front_right_difix
2. Applies the existing LUT sidecar to those Difix caches into:
   - stereo_front_left_difix_photometric_lut_cropped
   - stereo_front_right_difix_photometric_lut_cropped
3. Regenerates matching train pickles in degree-project-work/data:
   - av2_infos_train_stereo_front_left_rendered_difix_photometric_lut_cropped.pkl
   - av2_infos_train_stereo_front_right_rendered_difix_photometric_lut_cropped.pkl

Example:
  bash scripts/build_av2_stereo_difix_lut_assets.sh \
    --difix-root /home/samuelsoderberg/Difix3D \
    --renders-root /home/samuelsoderberg/neurad-studio/renders \
    --av2-root /home/samuelsoderberg/neurad-studio/data/av2/sensor \
    --project-root /home/samuelsoderberg/GitHub/degree-project-work
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEURAD_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NEURAD_ROOT="${NEURAD_ROOT_DEFAULT}"
PROJECT_ROOT="/home/samuelsoderberg/GitHub/degree-project-work"
RENDERS_ROOT="${NEURAD_ROOT}/renders"
AV2_ROOT="${NEURAD_ROOT}/data/av2/sensor"
INFO_DIR="${PROJECT_ROOT}/data"
DIFIX_ROOT=""
LUT_SIDECAR=""
DIFIX_OUTPUT_SUFFIX="difix"
LUT_OUTPUT_SUFFIX="difix_photometric_lut_cropped"
MODEL_ID="nvidia/difix"
PROMPT="remove degradation"
HEIGHT=""
WIDTH=""
TIMESTEP="199"
MAX_SCENES=""
OVERWRITE=false
SCENE_IDS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --neurad-root)
      NEURAD_ROOT="$2"
      shift 2
      ;;
    --project-root)
      PROJECT_ROOT="$2"
      shift 2
      ;;
    --renders-root)
      RENDERS_ROOT="$2"
      shift 2
      ;;
    --av2-root)
      AV2_ROOT="$2"
      shift 2
      ;;
    --info-dir)
      INFO_DIR="$2"
      shift 2
      ;;
    --difix-root)
      DIFIX_ROOT="$2"
      shift 2
      ;;
    --lut-sidecar)
      LUT_SIDECAR="$2"
      shift 2
      ;;
    --difix-output-suffix)
      DIFIX_OUTPUT_SUFFIX="$2"
      shift 2
      ;;
    --lut-output-suffix)
      LUT_OUTPUT_SUFFIX="$2"
      shift 2
      ;;
    --model-id)
      MODEL_ID="$2"
      shift 2
      ;;
    --prompt)
      PROMPT="$2"
      shift 2
      ;;
    --height)
      HEIGHT="$2"
      shift 2
      ;;
    --width)
      WIDTH="$2"
      shift 2
      ;;
    --timestep)
      TIMESTEP="$2"
      shift 2
      ;;
    --scene-id)
      SCENE_IDS+=("$2")
      shift 2
      ;;
    --max-scenes)
      MAX_SCENES="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${DIFIX_ROOT}" ]]; then
  echo "ERROR: --difix-root is required" >&2
  exit 1
fi

if [[ -z "${LUT_SIDECAR}" ]]; then
  LUT_SIDECAR="${RENDERS_ROOT}/stereo_photometric_lut_cropped.json"
fi

APPLY_DIFIX_SCRIPT="${NEURAD_ROOT}/scripts/apply_difix_to_renders.py"
PHOTOMETRIC_SCRIPT="${NEURAD_ROOT}/scripts/av2_stereo_photometric.py"
CONVERTER="${PROJECT_ROOT}/tools/converters/av2_converter.py"

for path in "${APPLY_DIFIX_SCRIPT}" "${PHOTOMETRIC_SCRIPT}" "${CONVERTER}" "${LUT_SIDECAR}"; do
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: required file not found: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${INFO_DIR}"

common_scene_args=()
if [[ -n "${MAX_SCENES}" ]]; then
  common_scene_args+=(--max-scenes "${MAX_SCENES}")
fi
if [[ ${#SCENE_IDS[@]} -gt 0 ]]; then
  common_scene_args+=(--scene-ids "${SCENE_IDS[@]}")
fi

overwrite_args=()
if [[ "${OVERWRITE}" == "true" ]]; then
  overwrite_args+=(--overwrite)
fi

LEFT_PICKLE_SUFFIX="stereo_front_left_rendered_${LUT_OUTPUT_SUFFIX}"
RIGHT_PICKLE_SUFFIX="stereo_front_right_rendered_${LUT_OUTPUT_SUFFIX}"

echo
echo "============================================================"
echo "Building AV2 stereo Difix + LUT assets"
echo "Neurad root:        ${NEURAD_ROOT}"
echo "Project root:       ${PROJECT_ROOT}"
echo "Difix root:         ${DIFIX_ROOT}"
echo "Renders root:       ${RENDERS_ROOT}"
echo "AV2 root:           ${AV2_ROOT}"
echo "Info dir:           ${INFO_DIR}"
echo "LUT sidecar:        ${LUT_SIDECAR}"
echo "Difix suffix:       ${DIFIX_OUTPUT_SUFFIX}"
echo "LUT output suffix:  ${LUT_OUTPUT_SUFFIX}"
echo "Model id:           ${MODEL_ID}"
echo "Prompt:             ${PROMPT}"
echo "Resolution override:${WIDTH:+ ${WIDTH}x${HEIGHT}}${WIDTH:- <auto from each input image>}"
echo "Timestep:           ${TIMESTEP}"
echo "Overwrite:          ${OVERWRITE}"
echo "============================================================"

difix_args=(
  --renders-root "${RENDERS_ROOT}"
  --difix-root "${DIFIX_ROOT}"
  --output-camera-dir-suffix "${DIFIX_OUTPUT_SUFFIX}"
  --model-id "${MODEL_ID}"
  --prompt "${PROMPT}"
  --timestep "${TIMESTEP}"
  "${common_scene_args[@]}"
  "${overwrite_args[@]}"
)

if [[ -n "${HEIGHT}" || -n "${WIDTH}" ]]; then
  if [[ -z "${HEIGHT}" || -z "${WIDTH}" ]]; then
    echo "ERROR: provide both --height and --width together" >&2
    exit 1
  fi
  difix_args+=(--height "${HEIGHT}" --width "${WIDTH}")
fi

echo
echo "Applying Difix to stereo render caches"
python "${APPLY_DIFIX_SCRIPT}" "${difix_args[@]}"

echo
echo "Applying LUT sidecar to Difix stereo caches"
python "${PHOTOMETRIC_SCRIPT}" apply-lut \
  --renders-root "${RENDERS_ROOT}" \
  --sidecar-path "${LUT_SIDECAR}" \
  --input-camera-dir-suffix "${DIFIX_OUTPUT_SUFFIX}" \
  --output-suffix "${LUT_OUTPUT_SUFFIX}" \
  "${common_scene_args[@]}" \
  "${overwrite_args[@]}"

echo
echo "Generating stereo train pickles for Difix + LUT caches"
python "${CONVERTER}" \
  --av2-root "${AV2_ROOT}" \
  --out-dir "${INFO_DIR}" \
  --split train \
  --cameras stereo_front_left \
  --suffix "${LEFT_PICKLE_SUFFIX}" \
  --rendered-images-root "${RENDERS_ROOT}" \
  --rendered-camera-dir-suffix "${LUT_OUTPUT_SUFFIX}"

python "${CONVERTER}" \
  --av2-root "${AV2_ROOT}" \
  --out-dir "${INFO_DIR}" \
  --split train \
  --cameras stereo_front_right \
  --suffix "${RIGHT_PICKLE_SUFFIX}" \
  --rendered-images-root "${RENDERS_ROOT}" \
  --rendered-camera-dir-suffix "${LUT_OUTPUT_SUFFIX}"

echo
echo "Finished."
echo "Created / ensured:"
echo "  - ${RENDERS_ROOT}/<scene>/stereo_front_left_${DIFIX_OUTPUT_SUFFIX}/"
echo "  - ${RENDERS_ROOT}/<scene>/stereo_front_right_${DIFIX_OUTPUT_SUFFIX}/"
echo "  - ${RENDERS_ROOT}/<scene>/stereo_front_left_${LUT_OUTPUT_SUFFIX}/"
echo "  - ${RENDERS_ROOT}/<scene>/stereo_front_right_${LUT_OUTPUT_SUFFIX}/"
echo "  - ${INFO_DIR}/av2_infos_train_${LEFT_PICKLE_SUFFIX}.pkl"
echo "  - ${INFO_DIR}/av2_infos_train_${RIGHT_PICKLE_SUFFIX}.pkl"
