#!/bin/bash
# Downloads 50 AV2 train scenes not yet present locally.
# Usage: ./download_50_new_scenes.sh [split]   (default: train)

set -e

SPLIT="${1:-train}"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TARGET_DIR="${SCRIPT_DIR}/data/av2/sensor/${SPLIT}"
N=50

echo "=========================================="
echo "Fetching full scene list from S3..."
echo "=========================================="

# Get all available scenes from S3
ALL_SCENES=$(s5cmd --no-sign-request ls "s3://argoverse/datasets/av2/sensor/${SPLIT}/" \
    | awk '{print $NF}' \
    | tr -d '/')

# Get already-downloaded scenes
mkdir -p "$TARGET_DIR"
EXISTING=$(ls "$TARGET_DIR" 2>/dev/null || true)

echo "Total scenes on S3:   $(echo "$ALL_SCENES" | wc -l)"
echo "Already downloaded:   $(echo "$EXISTING" | grep -c . 2>/dev/null || echo 0)"

# Find scenes not yet downloaded, take first N
NEW_SCENES=$(comm -23 \
    <(echo "$ALL_SCENES" | sort) \
    <(echo "$EXISTING" | sort) \
    | head -n "$N")

NEW_COUNT=$(echo "$NEW_SCENES" | grep -c . 2>/dev/null || echo 0)

if [ "$NEW_COUNT" -eq 0 ]; then
    echo "No new scenes to download!"
    exit 0
fi

echo "Scenes to download:   $NEW_COUNT"
echo "=========================================="
echo ""

# Pass to the existing downloader
# shellcheck disable=SC2086
"${SCRIPT_DIR}/download_av2_scenes.sh" "$SPLIT" $NEW_SCENES
