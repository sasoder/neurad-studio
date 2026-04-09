#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

DEFAULT_TARGET_CAMERAS = ("stereo_front_left", "stereo_front_right")
DEFAULT_LUT_BINS = 32
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


def _load_sidecar(path: Path) -> Dict[str, Any]:
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return torch.load(path, map_location="cpu")


def _save_sidecar(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        torch.save(payload, path)


def _get_render_frame_path(camera_dir: Path, timestamp_ns: int) -> Path:
    for suffix in (".jpg", ".png"):
        candidate = camera_dir / f"{timestamp_ns}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find cached render for timestamp {timestamp_ns} in {camera_dir}")


def _load_rgb_image(image_path: Path, size: Optional[Tuple[int, int]] = None, crop_to_size: bool = False) -> torch.Tensor:
    image = np.asarray(Image.open(image_path), dtype=np.float32)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    image = image[..., :3]
    if crop_to_size and size is not None:
        image = image[: size[0], : size[1]]
    image_tensor = torch.from_numpy(image / 255.0)
    if size is not None and (image_tensor.shape[0] != size[0] or image_tensor.shape[1] != size[1]):
        image_tensor = torch.nn.functional.interpolate(
            image_tensor.permute(2, 0, 1).unsqueeze(0),
            size=size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[0].permute(1, 2, 0)
    return image_tensor


def _extract_grayscale_target(image: torch.Tensor) -> torch.Tensor:
    if image.shape[-1] == 1:
        return image
    return image[..., :1]


def _rgb_to_luminance(rgb: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([0.299, 0.587, 0.114], dtype=rgb.dtype)
    return (rgb * weights).sum(dim=-1, keepdim=True)


def _apply_lut_to_gray(gray: torch.Tensor, lut_y: torch.Tensor) -> torch.Tensor:
    lut_y = lut_y.to(dtype=gray.dtype, device=gray.device)
    gray = torch.clamp(gray, 0.0, 1.0)
    if lut_y.numel() < 2:
        raise ValueError("lut_y must contain at least 2 points")
    position = gray * (lut_y.numel() - 1)
    idx0 = torch.floor(position).long().clamp(0, lut_y.numel() - 2)
    idx1 = idx0 + 1
    frac = position - idx0.to(position.dtype)
    return (1.0 - frac) * lut_y[idx0] + frac * lut_y[idx1]


def _get_mapping_type(entry: Dict[str, Any]) -> str:
    return str(entry.get("mapping_type", "global"))


def _apply_mapping(rgb: torch.Tensor, entry: Dict[str, Any]) -> torch.Tensor:
    luminance = _rgb_to_luminance(rgb)
    mapping_type = _get_mapping_type(entry)
    if mapping_type == "lut":
        lut_y = torch.as_tensor(entry["lut_y"], dtype=luminance.dtype, device=luminance.device)
        gray = _apply_lut_to_gray(luminance, lut_y)
    elif mapping_type == "global":
        scale = float(entry["scale"])
        bias = float(entry["bias"])
        gamma = max(float(entry.get("gamma", 1.0)), 1e-3)
        gray = torch.clamp(scale * torch.pow(torch.clamp(luminance, 0.0, 1.0), gamma) + bias, 0.0, 1.0)
    else:
        raise ValueError(f"Unsupported mapping_type {mapping_type!r}")
    return gray.expand(*rgb.shape[:-1], 3)


def _select_indices(num_frames: int, start_frame: int, max_frames: Optional[int]) -> range:
    if start_frame < 0:
        raise ValueError("start_frame must be >= 0")
    end_frame = num_frames if max_frames is None else min(num_frames, start_frame + max_frames)
    if start_frame >= end_frame:
        raise ValueError("Selected frame range is empty")
    return range(start_frame, end_frame)


def _fit_global_mapping(
    pred_gray: torch.Tensor,
    gt_gray: torch.Tensor,
    max_steps: int,
    learning_rate: float,
) -> Dict[str, Any]:
    baseline_mean_l1 = torch.abs(pred_gray - gt_gray).mean().item()
    scale_param = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
    bias_param = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
    log_gamma_param = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
    optimizer = torch.optim.Adam((scale_param, bias_param, log_gamma_param), lr=learning_rate)

    for _ in range(max_steps):
        gamma = torch.exp(log_gamma_param)
        mapped = torch.clamp(
            scale_param * torch.pow(torch.clamp(pred_gray, 0.0, 1.0), gamma) + bias_param,
            0.0,
            1.0,
        )
        loss = torch.abs(mapped - gt_gray).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    scale = float(scale_param.detach().item())
    bias = float(bias_param.detach().item())
    gamma = float(torch.exp(log_gamma_param.detach()).item())
    fitted_mean_l1 = torch.abs(
        torch.clamp(scale * torch.pow(torch.clamp(pred_gray, 0.0, 1.0), gamma) + bias, 0.0, 1.0) - gt_gray
    ).mean().item()
    return {
        "mapping_type": "global",
        "scale": scale,
        "bias": bias,
        "gamma": gamma,
        "baseline_mean_l1": baseline_mean_l1,
        "fitted_mean_l1": fitted_mean_l1,
    }


def _fit_lut_mapping(
    pred_gray: torch.Tensor,
    gt_gray: torch.Tensor,
    max_steps: int,
    learning_rate: float,
    lut_bins: int,
) -> Dict[str, Any]:
    baseline_mean_l1 = torch.abs(pred_gray - gt_gray).mean().item()
    if lut_bins < 2:
        raise ValueError("lut_bins must be >= 2")
    raw_delta = torch.nn.Parameter(torch.zeros(lut_bins - 1, dtype=torch.float32))
    optimizer = torch.optim.Adam((raw_delta,), lr=learning_rate)

    for _ in range(max_steps):
        delta = torch.nn.functional.softplus(raw_delta) + 1e-6
        lut_y = torch.cat((torch.zeros(1, dtype=delta.dtype), torch.cumsum(delta, dim=0)), dim=0)
        lut_y = lut_y / lut_y[-1].clamp_min(1e-6)
        mapped = _apply_lut_to_gray(pred_gray, lut_y)
        loss = torch.abs(mapped - gt_gray).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    delta = torch.nn.functional.softplus(raw_delta.detach()) + 1e-6
    lut_y = torch.cat((torch.zeros(1, dtype=delta.dtype), torch.cumsum(delta, dim=0)), dim=0)
    lut_y = lut_y / lut_y[-1].clamp_min(1e-6)
    fitted_mean_l1 = torch.abs(_apply_lut_to_gray(pred_gray, lut_y) - gt_gray).mean().item()
    return {
        "mapping_type": "lut",
        "lut_y": [float(v) for v in lut_y.tolist()],
        "lut_bins": lut_bins,
        "baseline_mean_l1": baseline_mean_l1,
        "fitted_mean_l1": fitted_mean_l1,
    }


def _append_sample_chunks(bucket: Dict[str, Any], pred_gray: torch.Tensor, gt_gray: torch.Tensor, max_total_samples: int) -> None:
    bucket["pred"].append(pred_gray.cpu())
    bucket["gt"].append(gt_gray.cpu())
    bucket["total_samples"] += pred_gray.numel()
    if bucket["total_samples"] <= max_total_samples * 2:
        return

    pred_all = torch.cat(bucket["pred"], dim=0)
    gt_all = torch.cat(bucket["gt"], dim=0)
    sample_idx = torch.randperm(pred_all.numel())[:max_total_samples]
    bucket["pred"] = [pred_all[sample_idx]]
    bucket["gt"] = [gt_all[sample_idx]]
    bucket["total_samples"] = max_total_samples


def _fit_mapping(
    pred_gray: torch.Tensor,
    gt_gray: torch.Tensor,
    mapping_mode: str,
    max_steps: int,
    learning_rate: float,
    lut_bins: int,
) -> Dict[str, Any]:
    if mapping_mode == "global":
        return _fit_global_mapping(pred_gray, gt_gray, max_steps=max_steps, learning_rate=learning_rate)
    if mapping_mode == "lut":
        return _fit_lut_mapping(
            pred_gray,
            gt_gray,
            max_steps=max_steps,
            learning_rate=learning_rate,
            lut_bins=lut_bins,
        )
    raise ValueError(f"Unsupported mapping_mode {mapping_mode!r}")


def _validate_sidecar_entries(entries: Dict[str, Dict[str, Any]], expected_mapping_type: Optional[str]) -> None:
    if expected_mapping_type is None:
        return
    mismatched = [
        camera
        for camera, entry in entries.items()
        if _get_mapping_type(entry) != expected_mapping_type
    ]
    if mismatched:
        mismatch_text = ", ".join(sorted(mismatched))
        raise ValueError(
            f"Sidecar contains mapping types incompatible with expected {expected_mapping_type!r}: {mismatch_text}"
        )


def fit_command(args: argparse.Namespace) -> None:
    scene_dirs = _collect_scene_dirs(args.renders_root, args.scene_ids, args.max_scenes)
    camera_samples: Dict[str, Dict[str, Any]] = {
        camera: {"pred": [], "gt": [], "total_samples": 0, "num_frames": 0, "scene_ids": []}
        for camera in args.target_cameras
    }

    print(f"Collecting cached stereo photometric samples from {len(scene_dirs)} scenes")
    for scene_idx, scene_dir in enumerate(scene_dirs, start=1):
        print(f"[{scene_idx}/{len(scene_dirs)}] {scene_dir.name}")
        for target_camera in args.target_cameras:
            camera_dir = scene_dir / target_camera
            manifest = _load_manifest(camera_dir)
            if manifest.get("target_camera") != target_camera:
                raise ValueError(f"Manifest in {camera_dir} declares {manifest.get('target_camera')}, expected {target_camera}")
            if args.appearance_sensor and manifest.get("appearance_sensor") != args.appearance_sensor:
                raise ValueError(
                    f"{camera_dir} was rendered with {manifest.get('appearance_sensor')}, expected {args.appearance_sensor}"
                )

            timestamps_ns = manifest["timestamps_ns"]
            source_images = manifest["source_images"]
            frame_indices = _select_indices(len(timestamps_ns), args.start_frame, args.max_frames)

            for frame_idx in frame_indices:
                timestamp_ns = int(timestamps_ns[frame_idx])
                render_path = _get_render_frame_path(camera_dir, timestamp_ns)
                target_path = Path(source_images[frame_idx])
                pred_rgb = _load_rgb_image(render_path)
                gt_rgb = _load_rgb_image(
                    target_path,
                    size=(pred_rgb.shape[0], pred_rgb.shape[1]),
                    crop_to_size=True,
                )

                pred_gray = _rgb_to_luminance(pred_rgb).reshape(-1)
                gt_gray = _extract_grayscale_target(gt_rgb).reshape(-1)

                if pred_gray.numel() > args.sample_pixels_per_frame:
                    sample_idx = torch.randperm(pred_gray.numel())[: args.sample_pixels_per_frame]
                    pred_gray = pred_gray[sample_idx]
                    gt_gray = gt_gray[sample_idx]

                _append_sample_chunks(
                    camera_samples[target_camera],
                    pred_gray,
                    gt_gray,
                    max_total_samples=args.max_total_samples_per_camera,
                )
                camera_samples[target_camera]["num_frames"] += 1

            camera_samples[target_camera]["scene_ids"].append(scene_dir.name)

    entries: Dict[str, Dict[str, Any]] = {}
    for target_camera in args.target_cameras:
        pred_chunks = camera_samples[target_camera]["pred"]
        gt_chunks = camera_samples[target_camera]["gt"]
        if not pred_chunks:
            print(f"Skipping {target_camera}: no samples collected")
            continue

        pred_gray = torch.cat(pred_chunks, dim=0).float()
        gt_gray = torch.cat(gt_chunks, dim=0).float()
        if pred_gray.numel() > args.max_total_samples_per_camera:
            sample_idx = torch.randperm(pred_gray.numel())[: args.max_total_samples_per_camera]
            pred_gray = pred_gray[sample_idx]
            gt_gray = gt_gray[sample_idx]
        fit = _fit_mapping(
            pred_gray,
            gt_gray,
            mapping_mode=args.mapping_mode,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            lut_bins=args.lut_bins,
        )
        entries[target_camera] = {
            "appearance_sensor": args.appearance_sensor,
            "mapping_type": fit["mapping_type"],
            "baseline_mean_l1": fit["baseline_mean_l1"],
            "fitted_mean_l1": fit["fitted_mean_l1"],
            "num_scenes": len(camera_samples[target_camera]["scene_ids"]),
            "num_frames": camera_samples[target_camera]["num_frames"],
        }
        if fit["mapping_type"] == "lut":
            entries[target_camera]["lut_y"] = fit["lut_y"]
            entries[target_camera]["lut_bins"] = fit["lut_bins"]
        else:
            entries[target_camera]["scale"] = fit["scale"]
            entries[target_camera]["bias"] = fit["bias"]
            entries[target_camera]["gamma"] = fit["gamma"]
        print(
            (
                f"{target_camera}: lut_bins={fit['lut_bins']}, "
                if fit["mapping_type"] == "lut"
                else f"{target_camera}: scale={fit['scale']:.6f}, bias={fit['bias']:.6f}, gamma={fit['gamma']:.6f}, "
            )
            + f"mean L1 {fit['baseline_mean_l1']:.6f} -> {fit['fitted_mean_l1']:.6f}"
        )

    if not entries:
        raise RuntimeError("No usable samples were collected")

    payload = {
        "format": "stereo_photometric_v1",
        "mapping_mode": args.mapping_mode,
        "entries": entries,
        "fit_settings": {
            "renders_root": str(args.renders_root),
            "appearance_sensor": args.appearance_sensor,
            "target_cameras": list(args.target_cameras),
            "scene_ids": [scene_dir.name for scene_dir in scene_dirs],
            "start_frame": args.start_frame,
            "max_frames": args.max_frames,
            "sample_pixels_per_frame": args.sample_pixels_per_frame,
            "max_total_samples_per_camera": args.max_total_samples_per_camera,
            "mapping_mode": args.mapping_mode,
            "lut_bins": args.lut_bins if args.mapping_mode == "lut" else None,
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
        },
    }
    _save_sidecar(args.output_path, payload)
    print(f"Saved stereo photometric sidecar to {args.output_path}")


def apply_command(args: argparse.Namespace) -> None:
    sidecar = _load_sidecar(args.sidecar_path)
    entries = sidecar["entries"]
    _validate_sidecar_entries(entries, args.expected_mapping_type)
    scene_dirs = _collect_scene_dirs(args.renders_root, args.scene_ids, args.max_scenes)

    for scene_idx, scene_dir in enumerate(scene_dirs, start=1):
        print(f"[{scene_idx}/{len(scene_dirs)}] {scene_dir.name}")
        for target_camera in args.target_cameras:
            if target_camera not in entries:
                raise ValueError(f"Sidecar does not contain an entry for {target_camera}")

            camera_dir = scene_dir / target_camera
            output_dir = scene_dir / f"{target_camera}_{args.output_suffix}"
            output_dir.mkdir(parents=True, exist_ok=True)

            manifest = _load_manifest(camera_dir)
            timestamps_ns = manifest["timestamps_ns"]
            frame_indices = _select_indices(len(timestamps_ns), args.start_frame, args.max_frames)
            entry = entries[target_camera]

            for frame_idx in frame_indices:
                timestamp_ns = int(timestamps_ns[frame_idx])
                source_path = _get_render_frame_path(camera_dir, timestamp_ns)
                target_path = output_dir / source_path.name
                if target_path.exists() and not args.overwrite:
                    continue

                rgb = _load_rgb_image(source_path)
                mapped = _apply_mapping(rgb, entry)
                image = (mapped.numpy() * 255.0).clip(0, 255).astype(np.uint8)
                Image.fromarray(image).save(target_path)

            output_manifest = dict(manifest)
            output_manifest["used_photometric_mapping"] = True
            output_manifest["photometric_sidecar"] = str(args.sidecar_path)
            output_manifest["source_render_dir"] = str(camera_dir)
            (output_dir / "render_manifest.json").write_text(json.dumps(output_manifest, indent=2), encoding="utf-8")

    print(f"Applied stereo photometric mapping from {args.sidecar_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit and apply AV2 stereo photometric mappings from cached renders.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit", help="Fit global left/right grayscale mappings from cached renders.")
    fit_parser.add_argument("--renders-root", type=Path, required=True)
    fit_parser.add_argument("--output-path", type=Path, required=True)
    fit_parser.add_argument("--appearance-sensor", type=str, default="ring_front_center")
    fit_parser.add_argument("--target-cameras", nargs="+", default=list(DEFAULT_TARGET_CAMERAS))
    fit_parser.add_argument("--scene-ids", nargs="*", default=[])
    fit_parser.add_argument("--max-scenes", type=int, default=None)
    fit_parser.add_argument("--start-frame", type=int, default=0)
    fit_parser.add_argument("--max-frames", type=int, default=None)
    fit_parser.add_argument("--sample-pixels-per-frame", type=int, default=1000)
    fit_parser.add_argument("--max-total-samples-per-camera", type=int, default=500000)
    fit_parser.add_argument("--max-steps", type=int, default=300)
    fit_parser.add_argument("--learning-rate", type=float, default=0.03)
    fit_parser.set_defaults(func=fit_command, mapping_mode="global", lut_bins=DEFAULT_LUT_BINS)

    fit_global_parser = subparsers.add_parser(
        "fit-global",
        help="Fit the current simple global grayscale mapping from cached renders.",
    )
    fit_global_parser.add_argument("--renders-root", type=Path, required=True)
    fit_global_parser.add_argument("--output-path", type=Path, required=True)
    fit_global_parser.add_argument("--appearance-sensor", type=str, default="ring_front_center")
    fit_global_parser.add_argument("--target-cameras", nargs="+", default=list(DEFAULT_TARGET_CAMERAS))
    fit_global_parser.add_argument("--scene-ids", nargs="*", default=[])
    fit_global_parser.add_argument("--max-scenes", type=int, default=None)
    fit_global_parser.add_argument("--start-frame", type=int, default=0)
    fit_global_parser.add_argument("--max-frames", type=int, default=None)
    fit_global_parser.add_argument("--sample-pixels-per-frame", type=int, default=1000)
    fit_global_parser.add_argument("--max-total-samples-per-camera", type=int, default=500000)
    fit_global_parser.add_argument("--max-steps", type=int, default=300)
    fit_global_parser.add_argument("--learning-rate", type=float, default=0.03)
    fit_global_parser.set_defaults(func=fit_command, mapping_mode="global", lut_bins=DEFAULT_LUT_BINS)

    fit_lut_parser = subparsers.add_parser(
        "fit-lut",
        help="Fit the revived monotonic LUT grayscale mapping from cached renders.",
    )
    fit_lut_parser.add_argument("--renders-root", type=Path, required=True)
    fit_lut_parser.add_argument("--output-path", type=Path, required=True)
    fit_lut_parser.add_argument("--appearance-sensor", type=str, default="ring_front_center")
    fit_lut_parser.add_argument("--target-cameras", nargs="+", default=list(DEFAULT_TARGET_CAMERAS))
    fit_lut_parser.add_argument("--scene-ids", nargs="*", default=[])
    fit_lut_parser.add_argument("--max-scenes", type=int, default=None)
    fit_lut_parser.add_argument("--start-frame", type=int, default=0)
    fit_lut_parser.add_argument("--max-frames", type=int, default=None)
    fit_lut_parser.add_argument("--sample-pixels-per-frame", type=int, default=1000)
    fit_lut_parser.add_argument("--max-total-samples-per-camera", type=int, default=500000)
    fit_lut_parser.add_argument("--lut-bins", type=int, default=DEFAULT_LUT_BINS)
    fit_lut_parser.add_argument("--max-steps", type=int, default=300)
    fit_lut_parser.add_argument("--learning-rate", type=float, default=0.03)
    fit_lut_parser.set_defaults(func=fit_command, mapping_mode="lut")

    apply_parser = subparsers.add_parser("apply", help="Apply a fitted mapping to cached renders.")
    apply_parser.add_argument("--renders-root", type=Path, required=True)
    apply_parser.add_argument("--sidecar-path", type=Path, required=True)
    apply_parser.add_argument("--target-cameras", nargs="+", default=list(DEFAULT_TARGET_CAMERAS))
    apply_parser.add_argument("--scene-ids", nargs="*", default=[])
    apply_parser.add_argument("--max-scenes", type=int, default=None)
    apply_parser.add_argument("--start-frame", type=int, default=0)
    apply_parser.add_argument("--max-frames", type=int, default=None)
    apply_parser.add_argument("--output-suffix", type=str, default="photometric")
    apply_parser.add_argument("--overwrite", action="store_true")
    apply_parser.set_defaults(func=apply_command, expected_mapping_type=None)

    apply_global_parser = subparsers.add_parser(
        "apply-global",
        help="Apply the simple global mapping into explicitly named photometric_global outputs.",
    )
    apply_global_parser.add_argument("--renders-root", type=Path, required=True)
    apply_global_parser.add_argument("--sidecar-path", type=Path, required=True)
    apply_global_parser.add_argument("--target-cameras", nargs="+", default=list(DEFAULT_TARGET_CAMERAS))
    apply_global_parser.add_argument("--scene-ids", nargs="*", default=[])
    apply_global_parser.add_argument("--max-scenes", type=int, default=None)
    apply_global_parser.add_argument("--start-frame", type=int, default=0)
    apply_global_parser.add_argument("--max-frames", type=int, default=None)
    apply_global_parser.add_argument("--output-suffix", type=str, default="photometric_global")
    apply_global_parser.add_argument("--overwrite", action="store_true")
    apply_global_parser.set_defaults(func=apply_command, expected_mapping_type="global")

    apply_lut_parser = subparsers.add_parser(
        "apply-lut",
        help="Apply the revived monotonic LUT mapping into explicitly named photometric_lut outputs.",
    )
    apply_lut_parser.add_argument("--renders-root", type=Path, required=True)
    apply_lut_parser.add_argument("--sidecar-path", type=Path, required=True)
    apply_lut_parser.add_argument("--target-cameras", nargs="+", default=list(DEFAULT_TARGET_CAMERAS))
    apply_lut_parser.add_argument("--scene-ids", nargs="*", default=[])
    apply_lut_parser.add_argument("--max-scenes", type=int, default=None)
    apply_lut_parser.add_argument("--start-frame", type=int, default=0)
    apply_lut_parser.add_argument("--max-frames", type=int, default=None)
    apply_lut_parser.add_argument("--output-suffix", type=str, default="photometric_lut")
    apply_lut_parser.add_argument("--overwrite", action="store_true")
    apply_lut_parser.set_defaults(func=apply_command, expected_mapping_type="lut")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
