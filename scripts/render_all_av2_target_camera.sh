#!/usr/bin/env bash
set -euo pipefail

# Batch render the latest SplatAD run for each scene into:
#   renders/<scene_id>/<target_camera>
#
# Example:
#   bash scripts/render_all_av2_target_camera.sh
#
# Optional overrides:
#   OUTPUTS_ROOT=outputs/unnamed/splatad
#   RENDERS_ROOT=renders
#   TARGET_CAMERA=stereo_front_left
#   APPEARANCE_SENSOR=ring_front_center
#   IMAGE_FORMAT=jpeg
#   MAX_FRAMES=8
#   SCENE_IDS="scene_a scene_b"
#   FORCE=1
#   PYTHON_BIN=python

OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs/unnamed/splatad}"
RENDERS_ROOT="${RENDERS_ROOT:-renders}"
TARGET_CAMERA="${TARGET_CAMERA:-stereo_front_left}"
APPEARANCE_SENSOR="${APPEARANCE_SENSOR:-ring_front_center}"
IMAGE_FORMAT="${IMAGE_FORMAT:-jpeg}"
MAX_FRAMES="${MAX_FRAMES:-}"
SCENE_IDS="${SCENE_IDS:-}"
FORCE="${FORCE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"

if [[ ! -d "$OUTPUTS_ROOT" ]]; then
  echo "Outputs root does not exist: $OUTPUTS_ROOT" >&2
  exit 1
fi

mkdir -p "$RENDERS_ROOT"
LOG_DIR="$RENDERS_ROOT/_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +"%Y-%m-%d_%H%M%S")"
FAIL_LOG="$LOG_DIR/render_${TARGET_CAMERA}_failures_${TIMESTAMP}.log"
RUN_LOG="$LOG_DIR/render_${TARGET_CAMERA}_${TIMESTAMP}.log"

scene_selected() {
  local scene_id="$1"
  if [[ -z "$SCENE_IDS" ]]; then
    return 0
  fi
  for selected in $SCENE_IDS; do
    if [[ "$selected" == "$scene_id" ]]; then
      return 0
    fi
  done
  return 1
}

latest_config_for_scene() {
  local scene_dir="$1"
  local latest=""
  shopt -s nullglob
  local candidates=("$scene_dir"/*/config.yml)
  shopt -u nullglob
  if [[ ${#candidates[@]} -eq 0 ]]; then
    return 1
  fi
  IFS=$'\n' latest="$(printf '%s\n' "${candidates[@]}" | sort | tail -n 1)"
  printf '%s\n' "$latest"
}

run_count=0
skip_count=0
fail_count=0

echo "Logging run output to $RUN_LOG"
echo "Logging failures to $FAIL_LOG"

while IFS= read -r -d '' scene_dir; do
  scene_id="$(basename "$scene_dir")"

  if ! scene_selected "$scene_id"; then
    continue
  fi

  if ! config_path="$(latest_config_for_scene "$scene_dir")"; then
    echo "Skipping $scene_id: no config.yml found" | tee -a "$RUN_LOG"
    continue
  fi

  output_dir="$RENDERS_ROOT/$scene_id/$TARGET_CAMERA"
  manifest_path="$output_dir/render_manifest.json"

  if [[ "$FORCE" != "1" && -f "$manifest_path" ]]; then
    echo "Skipping $scene_id: existing manifest at $manifest_path" | tee -a "$RUN_LOG"
    skip_count=$((skip_count + 1))
    continue
  fi

  cmd=(
    "$PYTHON_BIN"
    "nerfstudio/scripts/render.py"
    "av2-target-camera"
    "--load-config" "$config_path"
    "--sequence" "$scene_id"
    "--output-path" "$output_dir"
    "--target-camera" "$TARGET_CAMERA"
    "--appearance-sensor" "$APPEARANCE_SENSOR"
    "--image-format" "$IMAGE_FORMAT"
  )

  if [[ -n "$MAX_FRAMES" ]]; then
    cmd+=("--max-frames" "$MAX_FRAMES")
  fi

  echo "" | tee -a "$RUN_LOG"
  echo "=== Rendering $scene_id ===" | tee -a "$RUN_LOG"
  echo "Config: $config_path" | tee -a "$RUN_LOG"
  echo "Output: $output_dir" | tee -a "$RUN_LOG"
  echo "Command: TORCHDYNAMO_DISABLE=$TORCHDYNAMO_DISABLE ${cmd[*]}" | tee -a "$RUN_LOG"

  if TORCHDYNAMO_DISABLE="$TORCHDYNAMO_DISABLE" "${cmd[@]}" 2>&1 | tee -a "$RUN_LOG"; then
    run_count=$((run_count + 1))
  else
    fail_count=$((fail_count + 1))
    {
      echo "$scene_id"
      echo "  config: $config_path"
      echo "  output: $output_dir"
    } >> "$FAIL_LOG"
    echo "Failed scene: $scene_id" | tee -a "$RUN_LOG"
  fi
done < <(find "$OUTPUTS_ROOT" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

echo "" | tee -a "$RUN_LOG"
echo "Completed renders: $run_count" | tee -a "$RUN_LOG"
echo "Skipped renders: $skip_count" | tee -a "$RUN_LOG"
echo "Failed renders: $fail_count" | tee -a "$RUN_LOG"
echo "Failure log: $FAIL_LOG" | tee -a "$RUN_LOG"
