#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Build the non-luminance grayscale assets needed for stereo-transfer runs.

This script:
1. Optionally fits left/right global and LUT sidecars from cached stereo renders.
3. Applies the global + LUT mappings to:
   - cached stereo renders
   - RFC source-camera train images
4. Generates converter-compatible train pickles for the new caches.

Outputs created:
- stereo renders:
  - stereo_front_left_<suffix>
  - stereo_front_right_<suffix>
- RFC source caches:
  - ring_front_center_<suffix>
- train pickles:
  - av2_infos_train_*_<suffix>.pkl

Example:
  bash scripts/build_av2_target_grayscale_assets.sh \
    --renders-root /home/samuelsoderberg/neurad-studio/renders \
    --av2-root /home/samuelsoderberg/neurad-studio/data/av2/sensor \
    --neurad-root /home/samuelsoderberg/neurad-studio \
    --fit-global \
    --fit-lut \
    --fit-crop-bottom-px 250 \
    --asset-suffix cropped
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEURAD_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCHOOL_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT_DEFAULT="${SCHOOL_ROOT_DEFAULT}/GitHub/degree-project-work"

NEURAD_ROOT="${NEURAD_ROOT_DEFAULT}"
PROJECT_ROOT="${PROJECT_ROOT_DEFAULT}"
RENDERS_ROOT="${NEURAD_ROOT}/renders"
AV2_ROOT="${NEURAD_ROOT}/data/av2/sensor"
INFO_DIR="${PROJECT_ROOT}/data"
GLOBAL_SIDECAR=""
LUT_SIDECAR=""
FIT_GLOBAL=false
FIT_LUT=false
FIT_CROP_BOTTOM_PX=0
ASSET_SUFFIX=""
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
    --fit-global)
      FIT_GLOBAL=true
      shift
      ;;
    --fit-lut)
      FIT_LUT=true
      shift
      ;;
    --fit-crop-bottom-px)
      FIT_CROP_BOTTOM_PX="$2"
      shift 2
      ;;
    --asset-suffix)
      ASSET_SUFFIX="$2"
      shift 2
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

append_suffix() {
  local base="$1"
  if [[ -n "${ASSET_SUFFIX}" ]]; then
    printf '%s_%s' "${base}" "${ASSET_SUFFIX}"
  else
    printf '%s' "${base}"
  fi
}

if [[ -z "${GLOBAL_SIDECAR}" ]]; then
  GLOBAL_SIDECAR="${RENDERS_ROOT}/$(append_suffix "stereo_photometric_global").json"
fi

if [[ -z "${LUT_SIDECAR}" ]]; then
  LUT_SIDECAR="${RENDERS_ROOT}/$(append_suffix "stereo_photometric_lut").json"
fi

STEREO_GLOBAL_SUFFIX="$(append_suffix "photometric_global")"
STEREO_LUT_SUFFIX="$(append_suffix "photometric_lut")"
RFC_GLOBAL_LEFT_SUFFIX="$(append_suffix "photometric_global_left")"
RFC_GLOBAL_RIGHT_SUFFIX="$(append_suffix "photometric_global_right")"
RFC_LUT_LEFT_SUFFIX="$(append_suffix "photometric_lut_left")"
RFC_LUT_RIGHT_SUFFIX="$(append_suffix "photometric_lut_right")"

STEREO_LEFT_GLOBAL_PICKLE_SUFFIX="$(append_suffix "stereo_front_left_rendered_photometric_global")"
STEREO_RIGHT_GLOBAL_PICKLE_SUFFIX="$(append_suffix "stereo_front_right_rendered_photometric_global")"
STEREO_LEFT_LUT_PICKLE_SUFFIX="$(append_suffix "stereo_front_left_rendered_photometric_lut")"
STEREO_RIGHT_LUT_PICKLE_SUFFIX="$(append_suffix "stereo_front_right_rendered_photometric_lut")"
RFC_GLOBAL_LEFT_PICKLE_SUFFIX="$(append_suffix "ring_front_center_photometric_global_left")"
RFC_GLOBAL_RIGHT_PICKLE_SUFFIX="$(append_suffix "ring_front_center_photometric_global_right")"
RFC_LUT_LEFT_PICKLE_SUFFIX="$(append_suffix "ring_front_center_photometric_lut_left")"
RFC_LUT_RIGHT_PICKLE_SUFFIX="$(append_suffix "ring_front_center_photometric_lut_right")"

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

fit_args+=(--fit-crop-bottom-px "${FIT_CROP_BOTTOM_PX}")

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
echo "Asset suffix:   ${ASSET_SUFFIX:-<none>}"
echo "Fit crop px:    ${FIT_CROP_BOTTOM_PX}"
echo "============================================================"

if [[ "${FIT_GLOBAL}" == "true" || ! -f "${GLOBAL_SIDECAR}" ]]; then
  echo
  echo "Fitting global sidecar"
  python "${PHOTOMETRIC_SCRIPT}" fit-global \
    --renders-root "${RENDERS_ROOT}" \
    --output-path "${GLOBAL_SIDECAR}" \
    --sample-pixels-per-frame 1000 \
    --max-total-samples-per-camera 200000 \
    --max-steps 1000 \
    --learning-rate 0.03 \
    "${fit_args[@]}"
else
  echo
  echo "Reusing existing global sidecar: ${GLOBAL_SIDECAR}"
fi

if [[ "${FIT_LUT}" == "true" || ! -f "${LUT_SIDECAR}" ]]; then
  echo
  echo "Fitting LUT sidecar"
python "${PHOTOMETRIC_SCRIPT}" fit-lut \
  --renders-root "${RENDERS_ROOT}" \
  --output-path "${LUT_SIDECAR}" \
  --sample-pixels-per-frame 1000 \
  --max-total-samples-per-camera 200000 \
  --max-steps 1000 \
  --learning-rate 0.03 \
  "${fit_args[@]}"

else
  echo
  echo "Reusing existing LUT sidecar: ${LUT_SIDECAR}"
fi

echo
echo "Applying global mapping to stereo render caches"
python "${PHOTOMETRIC_SCRIPT}" apply-global \
  --renders-root "${RENDERS_ROOT}" \
  --sidecar-path "${GLOBAL_SIDECAR}" \
  --output-suffix "${STEREO_GLOBAL_SUFFIX}" \
  "${apply_args[@]}"

echo
echo "Applying LUT mapping to stereo render caches"
python "${PHOTOMETRIC_SCRIPT}" apply-lut \
  --renders-root "${RENDERS_ROOT}" \
  --sidecar-path "${LUT_SIDECAR}" \
  --output-suffix "${STEREO_LUT_SUFFIX}" \
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
  --output-suffix "${RFC_GLOBAL_LEFT_SUFFIX}" \
  "${rfc_apply_args[@]}"

python "${RFC_APPLY_SCRIPT}" \
  --av2-root "${AV2_ROOT}" \
  --split train \
  --source-camera ring_front_center \
  --output-root "${RENDERS_ROOT}" \
  --sidecar-path "${GLOBAL_SIDECAR}" \
  --entry-camera stereo_front_right \
  --output-suffix "${RFC_GLOBAL_RIGHT_SUFFIX}" \
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
  --output-suffix "${RFC_LUT_LEFT_SUFFIX}" \
  "${rfc_apply_args[@]}"

python "${RFC_APPLY_SCRIPT}" \
  --av2-root "${AV2_ROOT}" \
  --split train \
  --source-camera ring_front_center \
  --output-root "${RENDERS_ROOT}" \
  --sidecar-path "${LUT_SIDECAR}" \
  --entry-camera stereo_front_right \
  --output-suffix "${RFC_LUT_RIGHT_SUFFIX}" \
  "${rfc_apply_args[@]}"

echo
echo "Generating train pickles"
python "${CONVERTER}" \
  --av2-root "${AV2_ROOT}" \
  --out-dir "${INFO_DIR}" \
  --split train \
  --cameras stereo_front_left \
  --suffix "${STEREO_LEFT_GLOBAL_PICKLE_SUFFIX}" \
  --rendered-images-root "${RENDERS_ROOT}" \
  --rendered-camera-dir-suffix "${STEREO_GLOBAL_SUFFIX}"

python "${CONVERTER}" \
  --av2-root "${AV2_ROOT}" \
  --out-dir "${INFO_DIR}" \
  --split train \
  --cameras stereo_front_right \
  --suffix "${STEREO_RIGHT_GLOBAL_PICKLE_SUFFIX}" \
  --rendered-images-root "${RENDERS_ROOT}" \
  --rendered-camera-dir-suffix "${STEREO_GLOBAL_SUFFIX}"

python "${CONVERTER}" \
  --av2-root "${AV2_ROOT}" \
  --out-dir "${INFO_DIR}" \
  --split train \
  --cameras stereo_front_left \
  --suffix "${STEREO_LEFT_LUT_PICKLE_SUFFIX}" \
  --rendered-images-root "${RENDERS_ROOT}" \
  --rendered-camera-dir-suffix "${STEREO_LUT_SUFFIX}"

python "${CONVERTER}" \
  --av2-root "${AV2_ROOT}" \
  --out-dir "${INFO_DIR}" \
  --split train \
  --cameras stereo_front_right \
  --suffix "${STEREO_RIGHT_LUT_PICKLE_SUFFIX}" \
  --rendered-images-root "${RENDERS_ROOT}" \
  --rendered-camera-dir-suffix "${STEREO_LUT_SUFFIX}"

python "${CONVERTER}" \
  --av2-root "${AV2_ROOT}" \
  --out-dir "${INFO_DIR}" \
  --split train \
  --cameras ring_front_center \
  --suffix "${RFC_GLOBAL_LEFT_PICKLE_SUFFIX}" \
  --rendered-images-root "${RENDERS_ROOT}" \
  --rendered-camera-dir-suffix "${RFC_GLOBAL_LEFT_SUFFIX}"

python "${CONVERTER}" \
  --av2-root "${AV2_ROOT}" \
  --out-dir "${INFO_DIR}" \
  --split train \
  --cameras ring_front_center \
  --suffix "${RFC_GLOBAL_RIGHT_PICKLE_SUFFIX}" \
  --rendered-images-root "${RENDERS_ROOT}" \
  --rendered-camera-dir-suffix "${RFC_GLOBAL_RIGHT_SUFFIX}"

python "${CONVERTER}" \
  --av2-root "${AV2_ROOT}" \
  --out-dir "${INFO_DIR}" \
  --split train \
  --cameras ring_front_center \
  --suffix "${RFC_LUT_LEFT_PICKLE_SUFFIX}" \
  --rendered-images-root "${RENDERS_ROOT}" \
  --rendered-camera-dir-suffix "${RFC_LUT_LEFT_SUFFIX}"

python "${CONVERTER}" \
  --av2-root "${AV2_ROOT}" \
  --out-dir "${INFO_DIR}" \
  --split train \
  --cameras ring_front_center \
  --suffix "${RFC_LUT_RIGHT_PICKLE_SUFFIX}" \
  --rendered-images-root "${RENDERS_ROOT}" \
  --rendered-camera-dir-suffix "${RFC_LUT_RIGHT_SUFFIX}"

echo
echo "Finished."
echo "Created / ensured:"
echo "  - ${LUT_SIDECAR}"
echo "  - ${GLOBAL_SIDECAR}"
echo "  - stereo render global caches under ${RENDERS_ROOT}/<log>/stereo_front_{left,right}_${STEREO_GLOBAL_SUFFIX}/"
echo "  - stereo render LUT caches under ${RENDERS_ROOT}/<log>/stereo_front_{left,right}_${STEREO_LUT_SUFFIX}/"
echo "  - RFC photometric caches under ${RENDERS_ROOT}/<log>/ring_front_center_${RFC_GLOBAL_LEFT_SUFFIX}/ and ..._${RFC_GLOBAL_RIGHT_SUFFIX}/"
echo "  - RFC LUT caches under ${RENDERS_ROOT}/<log>/ring_front_center_${RFC_LUT_LEFT_SUFFIX}/ and ..._${RFC_LUT_RIGHT_SUFFIX}/"
echo "  - train pickles in ${INFO_DIR}"
