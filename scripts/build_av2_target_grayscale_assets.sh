#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Build the non-luminance grayscale assets needed for stereo-transfer runs.

This script:
1. Reuses an existing global photometric sidecar if present.
2. Fits a left/right LUT sidecar from cached stereo renders.
3. Applies the global + LUT mappings to:
   - cached stereo renders
   - RFC source-camera train images
4. Generates converter-compatible train pickles for the new caches.

Outputs created:
- stereo renders:
  - stereo_front_left_photometric_lut
  - stereo_front_right_photometric_lut
- RFC source caches:
  - ring_front_center_photometric_global_left
  - ring_front_center_photometric_global_right
  - ring_front_center_photometric_lut_left
  - ring_front_center_photometric_lut_right
- train pickles:
  - av2_infos_train_stereo_front_left_rendered_photometric_lut.pkl
  - av2_infos_train_stereo_front_right_rendered_photometric_lut.pkl
  - av2_infos_train_ring_front_center_photometric_global_left.pkl
  - av2_infos_train_ring_front_center_photometric_global_right.pkl
  - av2_infos_train_ring_front_center_photometric_lut_left.pkl
  - av2_infos_train_ring_front_center_photometric_lut_right.pkl

Example:
  bash scripts/build_av2_target_grayscale_assets.sh \
    --renders-root /home/samuelsoderberg/neurad-studio/renders \
    --av2-root /home/samuelsoderberg/neurad-studio/data/av2/sensor \
    --neurad-root /home/samuelsoderberg/neurad-studio
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEURAD_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCHOOL_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT_DEFAULT="${SCHOOL_ROOT_DEFAULT}/degree-project-work"

NEURAD_ROOT="${NEURAD_ROOT_DEFAULT}"
PROJECT_ROOT="${PROJECT_ROOT_DEFAULT}"
RENDERS_ROOT="${NEURAD_ROOT}/renders"
AV2_ROOT="${NEURAD_ROOT}/data/av2/sensor"
INFO_DIR="${PROJECT_ROOT}/data"
GLOBAL_SIDECAR=""
LUT_SIDECAR=""
FIT_LUT=false
OVERWRITE=false
MAX_SCENES=""

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
    --global-sidecar)
      GLOBAL_SIDECAR="$2"
      shift 2
      ;;
    --lut-sidecar)
      LUT_SIDECAR="$2"
      shift 2
      ;;
    --fit-lut)
      FIT_LUT=true
      shift
      ;;
    --overwrite)
      OVERWRITE=true
      shift
      ;;
    --max-scenes)
      MAX_SCENES="$2"
      shift 2
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

if [[ "${RENDERS_ROOT}" == "${NEURAD_ROOT_DEFAULT}/renders" ]]; then
  RENDERS_ROOT="${NEURAD_ROOT}/renders"
fi
if [[ "${AV2_ROOT}" == "${NEURAD_ROOT_DEFAULT}/data/av2/sensor" ]]; then
  AV2_ROOT="${NEURAD_ROOT}/data/av2/sensor"
fi
if [[ "${INFO_DIR}" == "${PROJECT_ROOT_DEFAULT}/data" ]]; then
  INFO_DIR="${PROJECT_ROOT}/data"
fi

if [[ -z "${GLOBAL_SIDECAR}" ]]; then
  GLOBAL_SIDECAR="${RENDERS_ROOT}/stereo_photometric_global.json"
fi

if [[ -z "${LUT_SIDECAR}" ]]; then
  LUT_SIDECAR="${RENDERS_ROOT}/stereo_photometric_lut.json"
fi

PHOTOMETRIC_SCRIPT="${NEURAD_ROOT}/scripts/av2_stereo_photometric.py"
RFC_APPLY_SCRIPT="${NEURAD_ROOT}/scripts/apply_av2_sidecar_to_source_images.py"
CONVERTER="${PROJECT_ROOT}/tools/converters/av2_converter.py"

for path in "${PHOTOMETRIC_SCRIPT}" "${RFC_APPLY_SCRIPT}" "${CONVERTER}"; do
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: required script not found: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${INFO_DIR}"

fit_args=()
apply_args=()
rfc_apply_args=()

if [[ -n "${MAX_SCENES}" ]]; then
  fit_args+=(--max-scenes "${MAX_SCENES}")
  apply_args+=(--max-scenes "${MAX_SCENES}")
  rfc_apply_args+=(--max-scenes "${MAX_SCENES}")
fi

if [[ "${OVERWRITE}" == "true" ]]; then
  apply_args+=(--overwrite)
  rfc_apply_args+=(--overwrite)
fi

echo
echo "============================================================"
echo "Building AV2 target grayscale assets"
echo "Neurad root:    ${NEURAD_ROOT}"
echo "Project root:   ${PROJECT_ROOT}"
echo "Renders root:   ${RENDERS_ROOT}"
echo "AV2 root:       ${AV2_ROOT}"
echo "Info dir:       ${INFO_DIR}"
echo "Global sidecar: ${GLOBAL_SIDECAR}"
echo "LUT sidecar:    ${LUT_SIDECAR}"
echo "============================================================"

if [[ ! -f "${GLOBAL_SIDECAR}" ]]; then
  echo "ERROR: global photometric sidecar not found: ${GLOBAL_SIDECAR}" >&2
  echo "Expected to reuse an existing fitted sidecar for RFC photometric outputs." >&2
  exit 1
fi

if [[ "${FIT_LUT}" == "true" || ! -f "${LUT_SIDECAR}" ]]; then
  echo
  echo "Fitting LUT sidecar"
  python "${PHOTOMETRIC_SCRIPT}" fit-lut \
    --renders-root "${RENDERS_ROOT}" \
    --output-path "${LUT_SIDECAR}" \
    "${fit_args[@]}"
else
  echo
  echo "Reusing existing LUT sidecar: ${LUT_SIDECAR}"
fi

echo
echo "Applying LUT mapping to stereo render caches"
python "${PHOTOMETRIC_SCRIPT}" apply-lut \
  --renders-root "${RENDERS_ROOT}" \
  --sidecar-path "${LUT_SIDECAR}" \
  "${apply_args[@]}"

echo
echo "Applying global photometric mapping to RFC source images"
python "${RFC_APPLY_SCRIPT}" \
  --av2-root "${AV2_ROOT}" \
  --split train \
  --source-camera ring_front_center \
  --output-root "${RENDERS_ROOT}" \
  --sidecar-path "${GLOBAL_SIDECAR}" \
  --entry-camera stereo_front_left \
  --output-suffix photometric_global_left \
  "${rfc_apply_args[@]}"

python "${RFC_APPLY_SCRIPT}" \
  --av2-root "${AV2_ROOT}" \
  --split train \
  --source-camera ring_front_center \
  --output-root "${RENDERS_ROOT}" \
  --sidecar-path "${GLOBAL_SIDECAR}" \
  --entry-camera stereo_front_right \
  --output-suffix photometric_global_right \
  "${rfc_apply_args[@]}"

echo
echo "Applying LUT mapping to RFC source images"
python "${RFC_APPLY_SCRIPT}" \
  --av2-root "${AV2_ROOT}" \
  --split train \
  --source-camera ring_front_center \
  --output-root "${RENDERS_ROOT}" \
  --sidecar-path "${LUT_SIDECAR}" \
  --entry-camera stereo_front_left \
  --output-suffix photometric_lut_left \
  "${rfc_apply_args[@]}"

python "${RFC_APPLY_SCRIPT}" \
  --av2-root "${AV2_ROOT}" \
  --split train \
  --source-camera ring_front_center \
  --output-root "${RENDERS_ROOT}" \
  --sidecar-path "${LUT_SIDECAR}" \
  --entry-camera stereo_front_right \
  --output-suffix photometric_lut_right \
  "${rfc_apply_args[@]}"

echo
echo "Generating train pickles"
python "${CONVERTER}" \
  --av2-root "${AV2_ROOT}" \
  --out-dir "${INFO_DIR}" \
  --split train \
  --cameras stereo_front_left \
  --suffix stereo_front_left_rendered_photometric_lut \
  --rendered-images-root "${RENDERS_ROOT}" \
  --rendered-camera-dir-suffix photometric_lut

python "${CONVERTER}" \
  --av2-root "${AV2_ROOT}" \
  --out-dir "${INFO_DIR}" \
  --split train \
  --cameras stereo_front_right \
  --suffix stereo_front_right_rendered_photometric_lut \
  --rendered-images-root "${RENDERS_ROOT}" \
  --rendered-camera-dir-suffix photometric_lut

python "${CONVERTER}" \
  --av2-root "${AV2_ROOT}" \
  --out-dir "${INFO_DIR}" \
  --split train \
  --cameras ring_front_center \
  --suffix ring_front_center_photometric_global_left \
  --rendered-images-root "${RENDERS_ROOT}" \
  --rendered-camera-dir-suffix photometric_global_left

python "${CONVERTER}" \
  --av2-root "${AV2_ROOT}" \
  --out-dir "${INFO_DIR}" \
  --split train \
  --cameras ring_front_center \
  --suffix ring_front_center_photometric_global_right \
  --rendered-images-root "${RENDERS_ROOT}" \
  --rendered-camera-dir-suffix photometric_global_right

python "${CONVERTER}" \
  --av2-root "${AV2_ROOT}" \
  --out-dir "${INFO_DIR}" \
  --split train \
  --cameras ring_front_center \
  --suffix ring_front_center_photometric_lut_left \
  --rendered-images-root "${RENDERS_ROOT}" \
  --rendered-camera-dir-suffix photometric_lut_left

python "${CONVERTER}" \
  --av2-root "${AV2_ROOT}" \
  --out-dir "${INFO_DIR}" \
  --split train \
  --cameras ring_front_center \
  --suffix ring_front_center_photometric_lut_right \
  --rendered-images-root "${RENDERS_ROOT}" \
  --rendered-camera-dir-suffix photometric_lut_right

echo
echo "Finished."
echo "Created / ensured:"
echo "  - ${LUT_SIDECAR}"
echo "  - stereo render LUT caches under ${RENDERS_ROOT}/<log>/stereo_front_{left,right}_photometric_lut/"
echo "  - RFC photometric caches under ${RENDERS_ROOT}/<log>/ring_front_center_photometric_global_{left,right}/"
echo "  - RFC LUT caches under ${RENDERS_ROOT}/<log>/ring_front_center_photometric_lut_{left,right}/"
echo "  - train pickles in ${INFO_DIR}"
