#!/usr/bin/env python3
"""Apply Difix enhancement to cached AV2 stereo renders.

This script reads rendered stereo caches laid out as:

  <renders_root>/<scene>/<camera>/<timestamp>.jpg

and writes non-destructive Difix caches as:

  <renders_root>/<scene>/<camera>_<output_camera_dir_suffix>/<timestamp>.jpg

Each output cache also gets a copied ``render_manifest.json`` so later stages
can reuse the existing LUT + pickle tooling without special cases.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image

DEFAULT_TARGET_CAMERAS = ("stereo_front_left", "stereo_front_right")
DEFAULT_PROMPT = "remove degradation"
SCENE_ID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def _collect_scene_dirs(root: Path, scene_ids: Sequence[str], max_scenes: Optional[int]) -> List[Path]:
    selected_scene_ids = set(scene_ids)
    scene_dirs: List[Path] = []
    for scene_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if selected_scene_ids and scene_dir.name not in selected_scene_ids:
            continue
        if not selected_scene_ids and SCENE_ID_PATTERN.fullmatch(scene_dir.name) is None:
            continue
        scene_dirs.append(scene_dir)
        if max_scenes is not None and len(scene_dirs) >= max_scenes:
            break
    if not scene_dirs:
        raise ValueError(f"No scene directories found under {root}")
    return scene_dirs


def _load_manifest(camera_dir: Path) -> Dict[str, Any]:
    manifest_path = camera_dir / "render_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing render manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _get_render_frame_path(camera_dir: Path, timestamp_ns: int) -> Path:
    for suffix in (".jpg", ".jpeg", ".png"):
        candidate = camera_dir / f"{timestamp_ns}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find cached render for timestamp {timestamp_ns} in {camera_dir}")


def _select_indices(num_frames: int, start_frame: int, max_frames: Optional[int]) -> range:
    if start_frame < 0:
        raise ValueError("start_frame must be >= 0")
    end_frame = num_frames if max_frames is None else min(num_frames, start_frame + max_frames)
    if start_frame >= end_frame:
        raise ValueError("Selected frame range is empty")
    return range(start_frame, end_frame)


def _import_difix_model(difix_root: Optional[Path]):
    if difix_root is not None:
        sys.path.insert(0, str(difix_root))
        sys.path.insert(0, str(difix_root / "src"))

    try:
        from model import Difix
    except ImportError as exc:
        root_hint = f" under {difix_root}" if difix_root is not None else ""
        raise ImportError(
            "Could not import Difix model. Pass --difix-root to a checked-out nv-tlabs/Difix3D repo"
            f"{root_hint}, or run this script from an environment where its src/ is importable."
        ) from exc

    return Difix


def _resolve_inference_size(image: Image.Image, requested_width: Optional[int], requested_height: Optional[int]) -> Tuple[int, int]:
    if (requested_width is None) != (requested_height is None):
        raise ValueError("Provide both --width and --height, or neither.")

    if requested_width is not None and requested_height is not None:
        return requested_width, requested_height

    width, height = image.size
    resolved_width = width - (width % 8)
    resolved_height = height - (height % 8)
    if resolved_width <= 0 or resolved_height <= 0:
        raise ValueError(f"Image is too small for Difix inference: {image.size}")
    return resolved_width, resolved_height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renders-root", type=Path, required=True)
    parser.add_argument("--difix-root", type=Path, default=None,
                        help="Optional path to a checked-out nv-tlabs/Difix3D repo.")
    parser.add_argument("--target-cameras", nargs="+", default=list(DEFAULT_TARGET_CAMERAS))
    parser.add_argument("--scene-ids", nargs="*", default=[])
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--input-camera-dir-suffix", type=str, default=None,
                        help="Optional input cache suffix, e.g. foo reads <camera>_foo instead of <camera>.")
    parser.add_argument("--output-camera-dir-suffix", type=str, default="difix")
    parser.add_argument("--model-path", type=Path, required=True,
                        help="Path to a local Difix checkpoint (.pkl) for the upstream model.py inference path.")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--height", type=int, default=None,
                        help="Optional inference height. Omit to use the input height rounded down to a multiple of 8.")
    parser.add_argument("--width", type=int, default=None,
                        help="Optional inference width. Omit to use the input width rounded down to a multiple of 8.")
    parser.add_argument("--timestep", type=int, default=199)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Difix inference requires CUDA; run this script on the SSH GPU machine.")

    scene_dirs = _collect_scene_dirs(args.renders_root, args.scene_ids, args.max_scenes)
    Difix = _import_difix_model(args.difix_root)

    model_path = str(args.model_path)
    model = Difix(
        pretrained_path=model_path,
        timestep=args.timestep,
    )
    model.set_eval()

    total_written = 0
    for scene_idx, scene_dir in enumerate(scene_dirs, start=1):
        print(f"[{scene_idx}/{len(scene_dirs)}] {scene_dir.name}")
        for target_camera in args.target_cameras:
            input_dir_name = (
                f"{target_camera}_{args.input_camera_dir_suffix}"
                if args.input_camera_dir_suffix
                else target_camera
            )
            input_dir = scene_dir / input_dir_name
            output_dir = scene_dir / f"{target_camera}_{args.output_camera_dir_suffix}"
            output_dir.mkdir(parents=True, exist_ok=True)

            manifest = _load_manifest(input_dir)
            timestamps_ns = manifest["timestamps_ns"]
            frame_indices = _select_indices(len(timestamps_ns), args.start_frame, args.max_frames)

            written_for_camera = 0
            for frame_idx in frame_indices:
                timestamp_ns = int(timestamps_ns[frame_idx])
                source_path = _get_render_frame_path(input_dir, timestamp_ns)
                target_path = output_dir / source_path.name
                if target_path.exists() and not args.overwrite:
                    continue

                source_image = Image.open(source_path).convert("RGB")
                infer_width, infer_height = _resolve_inference_size(source_image, args.width, args.height)
                output_image = model.sample(
                    source_image,
                    width=infer_width,
                    height=infer_height,
                    prompt=args.prompt,
                )
                output_image.save(target_path)
                written_for_camera += 1
                total_written += 1

            output_manifest = dict(manifest)
            output_manifest["used_difix"] = True
            output_manifest["difix_source_render_dir"] = str(input_dir)
            output_manifest["difix_settings"] = {
                "difix_root": str(args.difix_root) if args.difix_root is not None else None,
                "model_path": model_path,
                "prompt": args.prompt,
                "height": args.height,
                "width": args.width,
                "timestep": args.timestep,
                "input_camera_dir_suffix": args.input_camera_dir_suffix,
                "output_camera_dir_suffix": args.output_camera_dir_suffix,
            }
            (output_dir / "render_manifest.json").write_text(json.dumps(output_manifest, indent=2), encoding="utf-8")
            print(
                f"  {target_camera}: wrote {written_for_camera} frames to {output_dir.name} "
                f"(overwrite={args.overwrite})"
            )

    print(f"Applied Difix to cached renders; wrote {total_written} images under {args.renders_root}")


if __name__ == "__main__":
    main()
