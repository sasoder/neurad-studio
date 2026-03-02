#!/bin/bash
# Script to download AV2 scenes from a file list
# Usage: ./download_av2_from_file.sh <split> <scene_list_file>

set -e

if [ $# -ne 2 ]; then
    echo "Usage: $0 <split> <scene_list_file>"
    echo "Example: $0 train train_scenes.txt"
    exit 1
fi

SPLIT=$1
SCENE_FILE=$2

if [ ! -f "$SCENE_FILE" ]; then
    echo "Error: Scene file not found: $SCENE_FILE"
    exit 1
fi

# Read scenes into array
mapfile -t SCENES < "$SCENE_FILE"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "=========================================="
echo "Downloading ${#SCENES[@]} scenes from $SCENE_FILE"
echo "=========================================="

# Call the download script with all scenes
"${SCRIPT_DIR}/download_av2_scenes.sh" "$SPLIT" "${SCENES[@]}"
