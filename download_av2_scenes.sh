#!/bin/bash
# Script to download specific Argoverse 2 scenes
# Usage: ./download_av2_scenes.sh <split> <scene_id1> [scene_id2] [scene_id3] ...
# Example: ./download_av2_scenes.sh train 00a6ffc1-6ce9-3bc3-a060-6006e9893a1a

set -e

# Check if s5cmd is available
if ! command -v s5cmd &> /dev/null; then
    echo "Error: s5cmd not found. Install it with:"
    echo "  conda install s5cmd -c conda-forge"
    exit 1
fi

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <split> <scene_id1> [scene_id2] [scene_id3] ..."
    echo ""
    echo "Arguments:"
    echo "  split     - Dataset split: train, val, or test"
    echo "  scene_ids - One or more scene IDs to download"
    echo ""
    echo "Example:"
    echo "  $0 train 00a6ffc1-6ce9-3bc3-a060-6006e9893a1a"
    echo "  $0 val 02678d04-cc9f-3148-9f95-1ba66347dff9 0749e9e0-ca52-3546-b324-d704138b11b5"
    exit 1
fi

SPLIT=$1
shift  # Remove first argument, rest are scene IDs

# Validate split
if [[ ! "$SPLIT" =~ ^(train|val|test)$ ]]; then
    echo "Error: Invalid split '$SPLIT'. Must be train, val, or test."
    exit 1
fi

# Get script directory and set target directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TARGET_DIR="${SCRIPT_DIR}/data/av2/sensor/${SPLIT}"

echo "=========================================="
echo "Argoverse 2 Scene Downloader"
echo "=========================================="
echo "Split: $SPLIT"
echo "Target directory: $TARGET_DIR"
echo "Number of scenes: $#"
echo "=========================================="
echo ""

# Create target directory
mkdir -p "$TARGET_DIR"

# Download each scene
SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_SCENES=()

for SCENE_ID in "$@"; do
    echo "Downloading: $SCENE_ID"
    S3_URI="s3://argoverse/datasets/av2/sensor/${SPLIT}/${SCENE_ID}/"
    LOCAL_PATH="${TARGET_DIR}/${SCENE_ID}/"
    
    # Check if already exists
    if [ -d "$LOCAL_PATH" ]; then
        echo "  ⚠ Already exists, skipping: $LOCAL_PATH"
        continue
    fi
    
    # Download with progress
    if s5cmd --no-sign-request cp "${S3_URI}*" "${LOCAL_PATH}"; then
        echo "  ✓ Downloaded: $SCENE_ID (~1 GB)"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "  ✗ Failed: $SCENE_ID"
        FAILED_SCENES+=("$SCENE_ID")
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo ""
done

# Summary
echo "=========================================="
echo "DOWNLOAD SUMMARY"
echo "=========================================="
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAIL_COUNT"

if [ ${#FAILED_SCENES[@]} -gt 0 ]; then
    echo ""
    echo "Failed scenes:"
    for scene in "${FAILED_SCENES[@]}"; do
        echo "  - $scene"
    done
fi

echo ""
echo "Data location: $TARGET_DIR"
echo "=========================================="
