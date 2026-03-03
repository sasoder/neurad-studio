#!/usr/bin/env python3
"""Convert KITTI-360 fisheye images to pinhole images for reconstruction training.

The KITTI-360 fisheye cameras (image_02/image_03) use the Mei omnidirectional model.
NeuRAD's default camera models do not natively match that model, so this script remaps
fisheye images into a virtual pinhole camera and stores intrinsics for the remapped output.
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare KITTI-360 fisheye pinhole images")
    parser.add_argument("--data-root", type=Path, default=Path("data/kitti360"), help="KITTI-360 root")
    parser.add_argument("--sequence", type=str, required=True, help="Sequence id, e.g. 0004")
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["image_02", "image_03"],
        choices=["image_02", "image_03"],
        help="Fisheye cameras to preprocess",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="data_pinhole",
        help="Output subdir name under each fisheye camera folder",
    )
    parser.add_argument("--fov-deg", type=float, default=90.0, help="Virtual pinhole horizontal FOV")
    parser.add_argument("--width", type=int, default=0, help="Output width (0 keeps original width)")
    parser.add_argument("--height", type=int, default=0, help="Output height (0 keeps original height)")
    parser.add_argument("--max-frames", type=int, default=-1, help="Only preprocess first N frames")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output images")
    return parser.parse_args()


def read_opencv_yaml(path: Path) -> Dict[str, Dict[str, float]]:
    """Parse KITTI-360 OpenCV yaml file without requiring OpenCV-contrib."""
    data: Dict[str, Dict[str, float]] = {}
    current_section = ""
    kv_re = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:\s*(.*)\s*$")
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("%YAML") or line.startswith("---"):
                continue
            m = kv_re.match(raw)
            if not m:
                continue
            key, value = m.group(1), m.group(2).strip()
            if value == "":
                current_section = key
                data[current_section] = {}
            else:
                # Handle scalar at top level or inside current section.
                # KITTI-360 yaml includes strings like model_type: MEI.
                try:
                    parsed: float | str = float(value)
                except ValueError:
                    parsed = value
                if current_section and key in {"xi", "k1", "k2", "gamma1", "gamma2", "u0", "v0"}:
                    data[current_section][key] = float(parsed)
                else:
                    data[key] = {"value": parsed}
    return data


def fisheye_project(
    dirs: np.ndarray, xi: float, k1: float, k2: float, gamma1: float, gamma2: float, u0: float, v0: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Project unit ray directions to fisheye pixel coordinates (KITTI-360 model)."""
    x = dirs[..., 0]
    y = dirs[..., 1]
    z = dirs[..., 2]

    den = z + xi
    valid = den > 1e-6

    mx = np.zeros_like(x)
    my = np.zeros_like(y)
    mx[valid] = x[valid] / den[valid]
    my[valid] = y[valid] / den[valid]

    r2 = mx * mx + my * my
    dist = 1.0 + k1 * r2 + k2 * r2 * r2
    mx_d = mx * dist
    my_d = my * dist

    u = gamma1 * mx_d + u0
    v = gamma2 * my_d + v0
    return u.astype(np.float32), v.astype(np.float32)


def build_remap(
    width: int, height: int, fov_deg: float, params: Dict[str, float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build cv2.remap maps from pinhole output space to fisheye input space."""
    fx = (0.5 * width) / math.tan(math.radians(fov_deg) * 0.5)
    fy = fx
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5

    uu, vv = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    x = (uu - cx) / fx
    y = (vv - cy) / fy
    z = np.ones_like(x)

    dirs = np.stack([x, y, z], axis=-1)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    map_x, map_y = fisheye_project(
        dirs,
        params["xi"],
        params["k1"],
        params["k2"],
        params["gamma1"],
        params["gamma2"],
        params["u0"],
        params["v0"],
    )

    # Outside source image bounds become black.
    in_bounds = (
        (map_x >= 0.0)
        & (map_x <= (params["image_width"] - 1))
        & (map_y >= 0.0)
        & (map_y <= (params["image_height"] - 1))
    )
    map_x[~in_bounds] = -1.0
    map_y[~in_bounds] = -1.0

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return map_x, map_y, K


def parse_fisheye_params(calib_path: Path) -> Dict[str, float]:
    data = read_opencv_yaml(calib_path)
    params = {
        "image_width": int(data["image_width"]["value"]),
        "image_height": int(data["image_height"]["value"]),
        "xi": float(data["mirror_parameters"]["xi"]),
        "k1": float(data["distortion_parameters"]["k1"]),
        "k2": float(data["distortion_parameters"]["k2"]),
        "gamma1": float(data["projection_parameters"]["gamma1"]),
        "gamma2": float(data["projection_parameters"]["gamma2"]),
        "u0": float(data["projection_parameters"]["u0"]),
        "v0": float(data["projection_parameters"]["v0"]),
    }
    return params


def main() -> None:
    args = parse_args()
    seq_name = f"2013_05_28_drive_{args.sequence}_sync"
    seq_root = args.data_root / "data_2d_raw" / seq_name
    calib_root = args.data_root / "calibration"

    for cam_name in args.cameras:
        src_dir = seq_root / cam_name / "data_rgb"
        out_dir = seq_root / cam_name / args.output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        if not src_dir.exists():
            raise FileNotFoundError(f"Missing fisheye source directory: {src_dir}")

        cam_id = int(cam_name.split("_")[1])
        params = parse_fisheye_params(calib_root / f"image_{cam_id:02d}.yaml")
        out_w = args.width if args.width > 0 else int(params["image_width"])
        out_h = args.height if args.height > 0 else int(params["image_height"])
        map_x, map_y, K = build_remap(out_w, out_h, args.fov_deg, params)

        intrinsics_path = out_dir / "intrinsics.json"
        with open(intrinsics_path, "w") as f:
            json.dump(
                {
                    "camera": cam_name,
                    "width": out_w,
                    "height": out_h,
                    "fov_deg": args.fov_deg,
                    "K": K.tolist(),
                    "source_model": "kitti360_mei_omnidir",
                },
                f,
                indent=2,
            )

        images = sorted(src_dir.glob("*.png"))
        if args.max_frames > 0:
            images = images[: args.max_frames]

        print(f"[{cam_name}] remapping {len(images)} frames -> {out_dir}")
        for i, src in enumerate(images):
            dst = out_dir / src.name
            if dst.exists() and not args.overwrite:
                continue
            img = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if img is None:
                continue
            remapped = cv2.remap(
                img,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            cv2.imwrite(str(dst), remapped)
            if (i + 1) % 500 == 0:
                print(f"[{cam_name}] processed {i + 1}/{len(images)}")

    print("Done. You can now train with:")
    print(
        f"  ns-train neurad kitti360-data --data {args.data_root} --sequence {args.sequence} "
        "--pipeline.datamanager.dataparser.cameras image_02 image_03"
    )


if __name__ == "__main__":
    main()
