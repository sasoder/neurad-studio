#!/bin/bash
# Download KITTI-360 data for a single sequence.
# Usage: bash scripts/download_kitti360.sh [SEQUENCE_NUMBER] [OUTPUT_DIR] [CAMERA_MODE]
# Example: bash scripts/download_kitti360.sh 0009 data/kitti360 fisheye

set -euo pipefail

SEQ=${1:-0009}
OUTPUT_DIR=${2:-data/kitti360}
CAMERA_MODE=${3:-fisheye}  # fisheye | perspective
SEQUENCE="2013_05_28_drive_${SEQ}_sync"
BASE_URL="https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360"

echo "=== Downloading KITTI-360 data for sequence ${SEQ} ==="
echo "Output directory: ${OUTPUT_DIR}"
echo "Camera mode: ${CAMERA_MODE}"
mkdir -p "${OUTPUT_DIR}"
cd "${OUTPUT_DIR}"

if [ "${CAMERA_MODE}" != "fisheye" ] && [ "${CAMERA_MODE}" != "perspective" ]; then
    echo "Invalid CAMERA_MODE: ${CAMERA_MODE}. Use 'fisheye' or 'perspective'."
    exit 1
fi

download_and_extract() {
    local url="$1"
    local extract_dir="${2:-.}"
    local filename
    filename=$(basename "${url}")

    if [ -f "${filename}" ]; then
        echo "  [skip] ${filename} already exists"
    else
        echo "  [download] ${filename}"
        wget -q --show-progress "${url}" -O "${filename}"
    fi
    echo "  [extract] ${filename} -> ${extract_dir}"
    mkdir -p "${extract_dir}"
    unzip -qo "${filename}" -d "${extract_dir}"
    rm -f "${filename}"
}

# 1. Calibrations (3KB) - shared across all sequences
echo ""
echo "--- Step 1/6: Calibrations ---"
if [ -d "calibration" ]; then
    echo "  [skip] calibration/ already exists"
else
    download_and_extract "${BASE_URL}/384509ed5413ccc81328cf8c55cc6af078b8c444/calibration.zip"
fi

# 2. Vehicle poses (8.9MB) - all sequences included, small
# The zip extracts sequence dirs directly, so we extract into data_poses/
echo ""
echo "--- Step 2/6: Vehicle Poses ---"
if [ -d "data_poses/${SEQUENCE}" ]; then
    echo "  [skip] data_poses/${SEQUENCE}/ already exists"
else
    download_and_extract "${BASE_URL}/89a6bae3c8a6f789e12de4807fc1e8fdcf182cf4/data_poses.zip" "data_poses"
fi

# 3. 3D bounding boxes (30MB) - all sequences included, small
echo ""
echo "--- Step 3/6: 3D Bounding Boxes ---"
if [ -d "data_3d_bboxes" ]; then
    echo "  [skip] data_3d_bboxes/ already exists"
else
    download_and_extract "${BASE_URL}/ffa164387078f48a20f0188aa31b0384bb19ce60/data_3d_bboxes.zip"
fi

# 4. Camera images for the selected sequence
echo ""
if [ "${CAMERA_MODE}" = "fisheye" ]; then
    echo "--- Step 4/6: Fisheye Images (sequence ${SEQ}) ---"
    CAMS=(02 03)
else
    echo "--- Step 4/6: Perspective Images (sequence ${SEQ}) ---"
    CAMS=(00 01)
fi

for CAM in "${CAMS[@]}"; do
    CAM_DIR="data_rect"
    if [ "${CAMERA_MODE}" = "fisheye" ]; then
        CAM_DIR="data_rgb"
    fi
    if [ -d "data_2d_raw/${SEQUENCE}/image_${CAM}/${CAM_DIR}" ]; then
        echo "  [skip] image_${CAM} already exists"
    else
        download_and_extract \
            "${BASE_URL}/data_2d_raw/${SEQUENCE}_image_${CAM}.zip" \
            "data_2d_raw"
    fi
done

# 5. Camera + lidar timestamps
echo ""
echo "--- Step 5/6: Timestamps ---"
TS_CAMERA="${CAMS[0]}"
if [ -f "data_2d_raw/${SEQUENCE}/image_${TS_CAMERA}/timestamps.txt" ]; then
    echo "  [skip] camera timestamps already exist"
else
    if [ "${CAMERA_MODE}" = "fisheye" ]; then
        download_and_extract "${BASE_URL}/data_2d_raw/data_timestamps_fisheye.zip" "data_2d_raw"
    else
        download_and_extract "${BASE_URL}/data_2d_raw/data_timestamps_perspective.zip" "data_2d_raw"
    fi
fi

# Velodyne timestamps
if [ -f "data_3d_raw/${SEQUENCE}/velodyne_points/timestamps.txt" ]; then
    echo "  [skip] velodyne timestamps already exist"
else
    download_and_extract "${BASE_URL}/data_3d_raw/data_timestamps_velodyne.zip" "data_3d_raw"
fi

# 6. Raw Velodyne scans for the selected sequence (~11GB)
echo ""
echo "--- Step 6/6: Raw Velodyne Scans (sequence ${SEQ}) ---"
if [ -d "data_3d_raw/${SEQUENCE}/velodyne_points/data" ]; then
    echo "  [skip] velodyne data already exists"
else
    download_and_extract \
        "${BASE_URL}/data_3d_raw/${SEQUENCE}_velodyne.zip" \
        "data_3d_raw"
fi

echo ""
echo "=== Download complete! ==="
echo ""
echo "Data layout:"
echo "  ${OUTPUT_DIR}/"
echo "  ├── calibration/"
echo "  ├── data_2d_raw/${SEQUENCE}/"
if [ "${CAMERA_MODE}" = "fisheye" ]; then
echo "  │   ├── image_02/data_rgb/"
echo "  │   └── image_03/data_rgb/"
echo "  │   (run scripts/prepare_kitti360_fisheye.py after download)"
else
echo "  │   ├── image_00/data_rect/"
echo "  │   └── image_01/data_rect/"
fi
echo "  ├── data_3d_raw/${SEQUENCE}/"
echo "  │   └── velodyne_points/data/"
echo "  ├── data_3d_bboxes/train/"
echo "  └── data_poses/${SEQUENCE}/"
echo ""
echo "To train with SplatAD:"
echo "  ns-train neurad kitti360-data --data ${OUTPUT_DIR} --sequence ${SEQ}"
