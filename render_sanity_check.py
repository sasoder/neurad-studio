#!/usr/bin/env python3
"""Render first frame of every completed SplatAD scene.

For each completed run (final checkpoint present), renders just the first
timestamp (all cameras) using the train split and saves to:
  renders/<scene_id>/full/noshift/
"""

import os
import subprocess
import sys
from pathlib import Path

NEURAD_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = NEURAD_DIR / "outputs"
FINAL_STEP = 30000


def get_completed_runs():
    """Return list of (scene_id, config_path) for all fully trained scenes."""
    runs = []
    if not OUTPUTS_DIR.exists():
        return runs

    for config_path in OUTPUTS_DIR.rglob("config.yml"):
        run_dir = config_path.parent
        final_ckpt = run_dir / "nerfstudio_models" / f"step-{FINAL_STEP:09d}.ckpt"
        if not final_ckpt.exists():
            continue

        try:
            with open(config_path) as f:
                content = f.read()
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith("sequence:"):
                    scene_id = stripped.split(":", 1)[1].strip()
                    runs.append((scene_id, config_path))
                    break
        except Exception as e:
            print(f"Warning: Could not parse {config_path}: {e}")

    return runs


def already_rendered(scene_id: str) -> bool:
    out_dir = NEURAD_DIR / "renders" / scene_id / "full" / "noshift"
    return out_dir.exists() and any(out_dir.rglob("*.png"))


def main():
    runs = get_completed_runs()

    print("=" * 60)
    print("SplatAD Renderer  (first frame, train split)")
    print("=" * 60)
    print(f"Completed runs found: {len(runs)}")

    if not runs:
        print("No completed runs to render.")
        sys.exit(0)

    to_render = [(sid, cp) for sid, cp in runs if not already_rendered(sid)]
    already_done = len(runs) - len(to_render)

    print(f"Already rendered:     {already_done}")
    print(f"To render:            {len(to_render)}")
    print("=" * 60)

    if not to_render:
        print("\nAll scenes already rendered — check renders/<scene_id>/full/noshift/")
        sys.exit(0)

    for i, (scene_id, config_path) in enumerate(to_render, 1):
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(to_render)}] {scene_id}")
        print(f"    config: {config_path}")
        print(f"{'=' * 60}\n")

        output_path = NEURAD_DIR / "renders" / scene_id / "full" / "noshift"

        cmd = [
            "python", "nerfstudio/scripts/render.py", "dataset",
            "--load-config", str(config_path),
            "--data", str(NEURAD_DIR / "data" / "av2"),
            "--pose-source", "train",
            "--rendered-output-names", "rgb",
            "--max-images", "1",
            "--output-path", str(output_path),
        ]

        env = os.environ.copy()
        env["TORCHDYNAMO_DISABLE"] = "1"

        result = subprocess.run(cmd, cwd=str(NEURAD_DIR), env=env)
        if result.returncode != 0:
            print(f"\nWarning: render failed for {scene_id} (exit {result.returncode})")
        else:
            print(f"\nSaved to {output_path}")

    print(f"\n{'=' * 60}")
    print("Done! Review renders in:  renders/<scene_id>/full/noshift/")
    print("=" * 60)


if __name__ == "__main__":
    main()
