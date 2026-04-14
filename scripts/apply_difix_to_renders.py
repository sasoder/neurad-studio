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
import importlib.util
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
DEFAULT_CONDITIONING_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "data" / "av2" / "sensor" / "train"


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


def _get_conditioning_frame_path(conditioning_root: Path, scene_id: str, camera_name: str, timestamp_ns: int) -> Path:
    camera_dir = conditioning_root / scene_id / "sensors" / "cameras" / camera_name
    for suffix in (".jpg", ".jpeg", ".png"):
        candidate = camera_dir / f"{timestamp_ns}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find conditioning frame for timestamp {timestamp_ns} in {camera_dir}"
    )


def _select_indices(num_frames: int, start_frame: int, max_frames: Optional[int]) -> range:
    if start_frame < 0:
        raise ValueError("start_frame must be >= 0")
    end_frame = num_frames if max_frames is None else min(num_frames, start_frame + max_frames)
    if start_frame >= end_frame:
        raise ValueError("Selected frame range is empty")
    return range(start_frame, end_frame)


def _import_difix_pipeline_class(difix_root: Optional[Path]):
    if difix_root is not None:
        pipeline_path = difix_root / "src" / "pipeline_difix.py"
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Missing Difix pipeline file: {pipeline_path}")

        spec = importlib.util.spec_from_file_location("difix_pipeline_module", pipeline_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load Difix pipeline module from {pipeline_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.DifixPipeline

    try:
        from pipeline_difix import DifixPipeline  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Could not import DifixPipeline. Pass --difix-root to a checked-out nv-tlabs/Difix3D repo, "
            "or install a package that provides pipeline_difix."
        ) from exc

    return DifixPipeline


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
    parser.add_argument("--model-id", type=str, default="nvidia/difix",
                        help="Hugging Face model id for the public Difix diffusers pipeline.")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--height", type=int, default=None,
                        help="Optional inference height. Omit to use the input height rounded down to a multiple of 8.")
    parser.add_argument("--width", type=int, default=None,
                        help="Optional inference width. Omit to use the input width rounded down to a multiple of 8.")
    parser.add_argument("--timestep", type=int, default=199)
    parser.add_argument(
        "--use-conditioning-source-images",
        action="store_true",
        help=(
            "Use matching AV2 source camera images as the Difix conditioning input instead of the cached render "
            "frames. Looks up <conditioning-source-root>/<scene>/sensors/cameras/<camera>/<timestamp>.*."
        ),
    )
    parser.add_argument(
        "--conditioning-source-root",
        type=Path,
        default=DEFAULT_CONDITIONING_SOURCE_ROOT,
        help="Root directory for source conditioning images used with --use-conditioning-source-images.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Difix inference requires CUDA; run this script on the SSH GPU machine.")

    scene_dirs = _collect_scene_dirs(args.renders_root, args.scene_ids, args.max_scenes)
    DifixPipeline = _import_difix_pipeline_class(args.difix_root)

    pipe = DifixPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    pipe.enable_vae_slicing()
    pipe = pipe.to("cuda")

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

                conditioning_path = (
                    _get_conditioning_frame_path(
                        args.conditioning_source_root,
                        scene_dir.name,
                        target_camera,
                        timestamp_ns,
                    )
                    if args.use_conditioning_source_images
                    else source_path
                )

                source_image = Image.open(conditioning_path).convert("RGB")
                original_size = source_image.size
                infer_width, infer_height = _resolve_inference_size(source_image, args.width, args.height)
                output_image = pipe(
                    args.prompt,
                    image=source_image,
                    num_inference_steps=1,
                    timesteps=[args.timestep],
                    guidance_scale=0.0,
                    width=infer_width,
                    height=infer_height,
                ).images[0]
                if output_image.size != original_size:
                    output_image = output_image.resize(original_size, Image.LANCZOS)
                output_image.save(target_path)
                written_for_camera += 1
                total_written += 1

            output_manifest = dict(manifest)
            output_manifest["used_difix"] = True
            output_manifest["difix_source_render_dir"] = str(input_dir)
            output_manifest["difix_settings"] = {
                "difix_root": str(args.difix_root) if args.difix_root is not None else None,
                "model_id": args.model_id,
                "prompt": args.prompt,
                "height": args.height,
                "width": args.width,
                "timestep": args.timestep,
                "input_camera_dir_suffix": args.input_camera_dir_suffix,
                "output_camera_dir_suffix": args.output_camera_dir_suffix,
                "use_conditioning_source_images": args.use_conditioning_source_images,
                "conditioning_source_root": str(args.conditioning_source_root),
            }
            (output_dir / "render_manifest.json").write_text(json.dumps(output_manifest, indent=2), encoding="utf-8")
            print(
                f"  {target_camera}: wrote {written_for_camera} frames to {output_dir.name} "
                f"(overwrite={args.overwrite})"
            )

    print(f"Applied Difix to cached renders; wrote {total_written} images under {args.renders_root}")


if __name__ == "__main__":
    main()
