#!/usr/bin/env python3
"""Apply a fitted stereo grayscale sidecar to AV2 source-camera images.

This script writes converter-compatible caches under:

  <output_root>/<log_id>/<source_camera>_<output_suffix>/<timestamp>.jpg

The mapping is read from a sidecar produced by neurad-studio's
`av2_stereo_photometric.py`, selecting one target-camera entry such as
`stereo_front_left` or `stereo_front_right`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from PIL import Image


def _load_sidecar(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _apply_lut_to_gray(gray: np.ndarray, lut_y: np.ndarray) -> np.ndarray:
    gray = np.clip(gray, 0.0, 1.0)
    xp = np.linspace(0.0, 1.0, num=lut_y.size, dtype=np.float32)
    return np.interp(gray, xp, lut_y).astype(np.float32)


def _rgb_to_luminance(rgb: np.ndarray) -> np.ndarray:
    # Input is RGB in [0, 1].
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return (rgb * weights).sum(axis=-1, keepdims=True)


def _apply_mapping(rgb: np.ndarray, entry: Dict[str, Any]) -> np.ndarray:
    luminance = _rgb_to_luminance(rgb)
    mapping_type = str(entry.get("mapping_type", "global"))
    if mapping_type == "lut":
        lut_y = np.asarray(entry["lut_y"], dtype=np.float32)
        gray = _apply_lut_to_gray(luminance[..., 0], lut_y)[..., None]
    elif mapping_type == "global":
        scale = float(entry["scale"])
        bias = float(entry["bias"])
        gamma = max(float(entry.get("gamma", 1.0)), 1e-3)
        gray = np.clip(scale * np.power(np.clip(luminance, 0.0, 1.0), gamma) + bias, 0.0, 1.0)
    else:
        raise ValueError(f"Unsupported mapping_type: {mapping_type!r}")
    return np.repeat(gray, 3, axis=-1)


def _iter_scene_dirs(split_root: Path, scene_ids: Optional[Iterable[str]], max_scenes: Optional[int]) -> List[Path]:
    selected = set(scene_ids or [])
    scene_dirs: List[Path] = []
    for path in sorted(p for p in split_root.iterdir() if p.is_dir()):
        if selected and path.name not in selected:
            continue
        scene_dirs.append(path)
        if max_scenes is not None and len(scene_dirs) >= max_scenes:
            break
    return scene_dirs


def _iter_image_paths(camera_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        paths.extend(sorted(camera_dir.glob(pattern)))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--av2-root", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--source-camera", type=str, default="ring_front_center")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-suffix", type=str, required=True)
    parser.add_argument("--sidecar-path", type=Path, required=True)
    parser.add_argument("--entry-camera", type=str, required=True,
                        help="Target-camera entry to read from the sidecar, e.g. stereo_front_left.")
    parser.add_argument("--scene-ids", nargs="*", default=[])
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    split_root = args.av2_root / args.split
    if not split_root.exists():
        raise FileNotFoundError(f"Split root not found: {split_root}")

    sidecar = _load_sidecar(args.sidecar_path)
    entries = sidecar.get("entries", {})
    if args.entry_camera not in entries:
        raise KeyError(f"Sidecar has no entry for {args.entry_camera!r}")
    entry = entries[args.entry_camera]

    scene_dirs = _iter_scene_dirs(split_root, args.scene_ids, args.max_scenes)
    if not scene_dirs:
        raise RuntimeError(f"No scenes found under {split_root}")

    total_written = 0
    for scene_idx, scene_dir in enumerate(scene_dirs, start=1):
        source_dir = scene_dir / "sensors" / "cameras" / args.source_camera
        if not source_dir.is_dir():
            continue

        output_dir = args.output_root / scene_dir.name / f"{args.source_camera}_{args.output_suffix}"
        output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = _iter_image_paths(source_dir)
        print(f"[{scene_idx}/{len(scene_dirs)}] {scene_dir.name}: {len(image_paths)} frames")
        for image_path in image_paths:
            target_path = output_dir / image_path.name
            if target_path.exists() and not args.overwrite:
                continue

            rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
            mapped = _apply_mapping(rgb, entry)
            out = np.clip(mapped * 255.0, 0.0, 255.0).astype(np.uint8)
            Image.fromarray(out).save(target_path)
            total_written += 1

    print(
        f"Applied sidecar entry {args.entry_camera} to {args.source_camera} "
        f"for split={args.split}; wrote {total_written} images to {args.output_root}"
    )


if __name__ == "__main__":
    main()
