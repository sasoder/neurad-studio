#!/usr/bin/env python3
"""Batch training script for SplatAD on Argoverse 2 scenes.

Scans the outputs directory for completed runs (final checkpoint exists),
then trains remaining scenes from data/av2/sensor/train/ sequentially.
"""

import os
import subprocess
import sys
from pathlib import Path

NEURAD_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = NEURAD_DIR / "outputs"
DATA_DIR = NEURAD_DIR / "data" / "av2" / "sensor" / "train"
FINAL_STEP = 30000


def get_completed_scenes():
    """Find scenes that already have a final checkpoint (step 30000)."""
    completed = set()
    if not OUTPUTS_DIR.exists():
        return completed

    for config_path in OUTPUTS_DIR.rglob("config.yml"):
        run_dir = config_path.parent
        ckpt_dir = run_dir / "nerfstudio_models"

        final_ckpt = ckpt_dir / f"step-{FINAL_STEP:09d}.ckpt"
        if not final_ckpt.exists():
            continue

        try:
            with open(config_path) as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("sequence:"):
                        seq_id = stripped.split(":", 1)[1].strip()
                        completed.add(seq_id)
                        break
        except Exception as e:
            print(f"Warning: Could not parse {config_path}: {e}")

    return completed


def get_available_scenes():
    """Get all scene IDs from the data directory."""
    if not DATA_DIR.exists():
        print(f"Error: Data directory not found: {DATA_DIR}")
        sys.exit(1)
    return sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])


def main():
    completed = get_completed_scenes()
    available = get_available_scenes()
    remaining = [s for s in available if s not in completed]

    print("=" * 60)
    print("SplatAD Batch Training")
    print("=" * 60)
    print(f"Total scenes in data dir:  {len(available)}")
    print(f"Already completed:         {len(completed)}")
    print(f"Remaining to train:        {len(remaining)}")
    print("=" * 60)

    if completed:
        print("\nCompleted scenes:")
        for s in sorted(completed):
            print(f"  {s}")

    if not remaining:
        print("\nAll scenes have been trained!")
        return

    print(f"\nScenes to train:")
    for i, s in enumerate(remaining, 1):
        print(f"  {i}. {s}")
    print()

    for i, scene_id in enumerate(remaining, 1):
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(remaining)}] Training scene: {scene_id}")
        print(f"{'=' * 60}\n")

        cmd = [
            "python", "nerfstudio/scripts/train.py", "splatad",
            "argoverse2-data",
            "--data", "data/av2",
            "--split", "train",
            "--sequence", scene_id,
        ]

        env = os.environ.copy()
        env["TORCHDYNAMO_DISABLE"] = "1"

        result = subprocess.run(cmd, cwd=str(NEURAD_DIR), env=env)

        if result.returncode != 0:
            print(f"\nWarning: Training failed for {scene_id} (exit code {result.returncode})")
            print("Continuing with next scene...")
        else:
            print(f"\nCompleted scene {scene_id}")

    print(f"\n{'=' * 60}")
    print("Batch training finished!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
