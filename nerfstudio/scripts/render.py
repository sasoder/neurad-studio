# Copyright 2024 the authors of NeuRAD and contributors.
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
"""
render.py
"""

from __future__ import annotations

import gzip
import json
import os
import pickle
import shutil
import struct
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import mediapy as media
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import torch
import tyro
import viser.transforms as tf
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader, convert_pose_dataframe_to_SE3
from jaxtyping import Float
from PIL import Image
from rich import box, style
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from torch import Tensor
from typing_extensions import Annotated

from nerfstudio.cameras.camera_paths import (
    get_interpolated_camera_path,
    get_interpolated_spiral_camera_path,
    get_path_from_json,
    get_spiral_path,
)
from nerfstudio.cameras.camera_utils import rotmat_to_unitquat, unitquat_to_rotmat
from nerfstudio.cameras.cameras import Cameras, CameraType, RayBundle
from nerfstudio.cameras.lidars import transform_points
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
from nerfstudio.data.datamanagers.random_cameras_datamanager import RandomCamerasDataManager
from nerfstudio.data.dataparsers.ad_dataparser import OPENCV_TO_NERFSTUDIO
from nerfstudio.data.datasets.base_dataset import Dataset
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.data.utils.dataloaders import FixedIndicesEvalDataloader
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.model_components import renderers
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import poses as pose_utils
from nerfstudio.utils import colormaps, install_checks
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE, ItersPerSecColumn
from nerfstudio.utils.scripts import run_command

AV2_SPLITS = ("train", "val", "test", "mini", "mini_val", "mini_test")
DEFAULT_AV2_STEREO_APPEARANCE_SENSORS = {
    "stereo_front_left": "ring_front_left",
    "stereo_front_right": "ring_front_right",
}
AV2_CAMERA_DELTA_CACHE: Dict[Tuple[str, str, str], Dict[str, Any]] = {}


def _render_trajectory_video(
    pipeline: Pipeline,
    cameras: Cameras,
    output_filename: Path,
    rendered_output_names: List[str],
    crop_data: Optional[CropData] = None,
    rendered_resolution_scaling_factor: float = 1.0,
    seconds: float = 5.0,
    output_format: Literal["images", "video"] = "video",
    image_format: Literal["jpeg", "png"] = "jpeg",
    jpeg_quality: int = 100,
    depth_near_plane: Optional[float] = None,
    depth_far_plane: Optional[float] = None,
    colormap_options: colormaps.ColormapOptions = colormaps.ColormapOptions(),
    render_nearest_camera=False,
    check_occlusions: bool = False,
    rgb_postprocess_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> None:
    """Helper function to create a video of the spiral trajectory.

    Args:
        pipeline: Pipeline to evaluate with.
        cameras: Cameras to render.
        output_filename: Name of the output file.
        rendered_output_names: List of outputs to visualise.
        crop_data: Crop data to apply to the rendered images.
        rendered_resolution_scaling_factor: Scaling factor to apply to the camera image resolution.
        seconds: Length of output video.
        output_format: How to save output data.
        depth_near_plane: Closest depth to consider when using the colormap for depth. If None, use min value.
        depth_far_plane: Furthest depth to consider when using the colormap for depth. If None, use max value.
        colormap_options: Options for colormap.
        render_nearest_camera: Whether to render the nearest training camera to the rendered camera.
        check_occlusions: If true, checks line-of-sight occlusions when computing camera distance and rejects cameras not visible to each other
    """
    CONSOLE.print("[bold green]Creating trajectory " + output_format)
    cameras.rescale_output_resolution(rendered_resolution_scaling_factor)
    cameras = cameras.to(pipeline.device)
    fps = len(cameras) / seconds

    progress = Progress(
        TextColumn(":movie_camera: Rendering :movie_camera:"),
        BarColumn(),
        TaskProgressColumn(
            text_format="[progress.percentage]{task.completed}/{task.total:>.0f}({task.percentage:>3.1f}%)",
            show_speed=True,
        ),
        ItersPerSecColumn(suffix="fps"),
        TimeRemainingColumn(elapsed_when_finished=False, compact=False),
        TimeElapsedColumn(),
    )
    output_image_dir = output_filename.parent / output_filename.stem
    if output_format == "images":
        output_image_dir.mkdir(parents=True, exist_ok=True)
    if output_format == "video":
        # make the folder if it doesn't exist
        output_filename.parent.mkdir(parents=True, exist_ok=True)
        # NOTE:
        # we could use ffmpeg_args "-movflags faststart" for progressive download,
        # which would force moov atom into known position before mdat,
        # but then we would have to move all of mdat to insert metadata atom
        # (unless we reserve enough space to overwrite with our uuid tag,
        # but we don't know how big the video file will be, so it's not certain!)

    with ExitStack() as stack:
        writer = None

        if render_nearest_camera:
            assert pipeline.datamanager.train_dataset is not None
            train_dataset = pipeline.datamanager.train_dataset
            train_cameras = train_dataset.cameras.to(pipeline.device)
        else:
            train_dataset = None
            train_cameras = None

        with progress:
            for camera_idx in progress.track(range(cameras.size), description=""):
                obb_box = None
                if crop_data is not None:
                    obb_box = crop_data.obb

                max_dist, max_idx = -1, -1
                true_max_dist, true_max_idx = -1, -1

                if render_nearest_camera:
                    assert pipeline.datamanager.train_dataset is not None
                    assert train_dataset is not None
                    assert train_cameras is not None
                    cam_pos = cameras[camera_idx].camera_to_worlds[:, 3].cpu()
                    cam_quat = tf.SO3.from_matrix(cameras[camera_idx].camera_to_worlds[:3, :3].numpy(force=True)).wxyz

                    for i in range(len(train_cameras)):
                        train_cam_pos = train_cameras[i].camera_to_worlds[:, 3].cpu()
                        # Make sure the line of sight from rendered cam to training cam is not blocked by any object
                        bundle = RayBundle(
                            origins=cam_pos.view(1, 3),
                            directions=((cam_pos - train_cam_pos) / (cam_pos - train_cam_pos).norm()).view(1, 3),
                            pixel_area=torch.tensor(1).view(1, 1),
                            nears=torch.tensor(0.05).view(1, 1),
                            fars=torch.tensor(100).view(1, 1),
                            camera_indices=torch.tensor(0).view(1, 1),
                            metadata={},
                        ).to(pipeline.device)
                        outputs = pipeline.model.get_outputs(bundle)

                        q = tf.SO3.from_matrix(train_cameras[i].camera_to_worlds[:3, :3].numpy(force=True)).wxyz
                        # calculate distance between two quaternions
                        rot_dist = 1 - np.dot(q, cam_quat) ** 2
                        pos_dist = torch.norm(train_cam_pos - cam_pos)
                        dist = 0.3 * rot_dist + 0.7 * pos_dist

                        if true_max_dist == -1 or dist < true_max_dist:
                            true_max_dist = dist
                            true_max_idx = i

                        if outputs["depth"][0] < torch.norm(cam_pos - train_cam_pos).item():
                            continue

                        if check_occlusions and (max_dist == -1 or dist < max_dist):
                            max_dist = dist
                            max_idx = i

                    if max_idx == -1:
                        max_idx = true_max_idx

                if crop_data is not None:
                    with renderers.background_color_override_context(
                        crop_data.background_color.to(pipeline.device)
                    ), torch.no_grad():
                        outputs = pipeline.model.get_outputs_for_camera(
                            cameras[camera_idx : camera_idx + 1], obb_box=obb_box
                        )
                else:
                    with torch.no_grad():
                        outputs = pipeline.model.get_outputs_for_camera(
                            cameras[camera_idx : camera_idx + 1], obb_box=obb_box
                        )

                render_image = []
                for rendered_output_name in rendered_output_names:
                    if rendered_output_name not in outputs:
                        CONSOLE.rule("Error", style="red")
                        CONSOLE.print(f"Could not find {rendered_output_name} in the model outputs", justify="center")
                        CONSOLE.print(
                            f"Please set --rendered_output_name to one of: {outputs.keys()}", justify="center"
                        )
                        sys.exit(1)
                    output_image = outputs[rendered_output_name]
                    is_depth = rendered_output_name.find("depth") != -1
                    if is_depth:
                        output_image = (
                            colormaps.apply_depth_colormap(
                                output_image,
                                accumulation=outputs["accumulation"],
                                near_plane=depth_near_plane,
                                far_plane=depth_far_plane,
                                colormap_options=colormap_options,
                            )
                            .cpu()
                            .numpy()
                        )
                    else:
                        if rendered_output_name == "rgb" and rgb_postprocess_fn is not None:
                            output_image = rgb_postprocess_fn(output_image)
                        output_image = (
                            colormaps.apply_colormap(
                                image=output_image,
                                colormap_options=colormap_options,
                            )
                            .cpu()
                            .numpy()
                        )
                    render_image.append(output_image)

                # Add closest training image to the right of the rendered image
                if render_nearest_camera:
                    assert train_dataset is not None
                    assert train_cameras is not None
                    img = train_dataset.get_image_float32(max_idx)
                    height = cameras.image_height[0]
                    # maintain the resolution of the img to calculate the width from the height
                    width = int(img.shape[1] * (height / img.shape[0]))
                    resized_image = torch.nn.functional.interpolate(
                        img.permute(2, 0, 1)[None], size=(int(height), int(width))
                    )[0].permute(1, 2, 0)
                    resized_image = (
                        colormaps.apply_colormap(
                            image=resized_image,
                            colormap_options=colormap_options,
                        )
                        .cpu()
                        .numpy()
                    )
                    render_image.append(resized_image)

                render_image = np.concatenate(render_image, axis=1)
                if output_format == "images":
                    if image_format == "png":
                        media.write_image(output_image_dir / f"{camera_idx:05d}.png", render_image, fmt="png")
                    if image_format == "jpeg":
                        media.write_image(
                            output_image_dir / f"{camera_idx:05d}.jpg", render_image, fmt="jpeg", quality=jpeg_quality
                        )
                if output_format == "video":
                    if writer is None:
                        render_width = int(render_image.shape[1])
                        render_height = int(render_image.shape[0])
                        writer = stack.enter_context(
                            media.VideoWriter(
                                path=output_filename,
                                shape=(render_height, render_width),
                                fps=fps,
                            )
                        )
                    writer.add_image(render_image)

    table = Table(
        title=None,
        show_header=False,
        box=box.MINIMAL,
        title_style=style.Style(bold=True),
    )
    if output_format == "video":
        if cameras.camera_type[0] == CameraType.EQUIRECTANGULAR.value:
            CONSOLE.print("Adding spherical camera data")
            insert_spherical_metadata_into_file(output_filename)
        table.add_row("Video", str(output_filename))
    else:
        table.add_row("Images", str(output_image_dir))
    CONSOLE.print(Panel(table, title="[bold][green]:tada: Render Complete :tada:[/bold]", expand=False))


def insert_spherical_metadata_into_file(
    output_filename: Path,
) -> None:
    """Inserts spherical metadata into MP4 video file in-place.
    Args:
        output_filename: Name of the (input and) output file.
    """
    # NOTE:
    # because we didn't use faststart, the moov atom will be at the end;
    # to insert our metadata, we need to find (skip atoms until we get to) moov.
    # we should have 0x00000020 ftyp, then 0x00000008 free, then variable mdat.
    spherical_uuid = b"\xff\xcc\x82\x63\xf8\x55\x4a\x93\x88\x14\x58\x7a\x02\x52\x1f\xdd"
    spherical_metadata = bytes(
        """<rdf:SphericalVideo
xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'
xmlns:GSpherical='http://ns.google.com/videos/1.0/spherical/'>
<GSpherical:ProjectionType>equirectangular</GSpherical:ProjectionType>
<GSpherical:Spherical>True</GSpherical:Spherical>
<GSpherical:Stitched>True</GSpherical:Stitched>
<GSpherical:StitchingSoftware>nerfstudio</GSpherical:StitchingSoftware>
</rdf:SphericalVideo>""",
        "utf-8",
    )
    insert_size = len(spherical_metadata) + 8 + 16
    with open(output_filename, mode="r+b") as mp4file:
        try:
            # get file size
            mp4file_size = os.stat(output_filename).st_size

            # find moov container (probably after ftyp, free, mdat)
            while True:
                pos = mp4file.tell()
                size, tag = struct.unpack(">I4s", mp4file.read(8))
                if tag == b"moov":
                    break
                mp4file.seek(pos + size)
            # if moov isn't at end, bail
            if pos + size != mp4file_size:
                # TODO: to support faststart, rewrite all stco offsets
                raise Exception("moov container not at end of file")
            # go back and write inserted size
            mp4file.seek(pos)
            mp4file.write(struct.pack(">I", size + insert_size))
            # go inside moov
            mp4file.seek(pos + 8)
            # find trak container (probably after mvhd)
            while True:
                pos = mp4file.tell()
                size, tag = struct.unpack(">I4s", mp4file.read(8))
                if tag == b"trak":
                    break
                mp4file.seek(pos + size)
            # go back and write inserted size
            mp4file.seek(pos)
            mp4file.write(struct.pack(">I", size + insert_size))
            # we need to read everything from end of trak to end of file in order to insert
            # TODO: to support faststart, make more efficient (may load nearly all data)
            mp4file.seek(pos + size)
            rest_of_file = mp4file.read(mp4file_size - pos - size)
            # go to end of trak (again)
            mp4file.seek(pos + size)
            # insert our uuid atom with spherical metadata
            mp4file.write(struct.pack(">I4s16s", insert_size, b"uuid", spherical_uuid))
            mp4file.write(spherical_metadata)
            # write rest of file
            mp4file.write(rest_of_file)
        finally:
            mp4file.close()


@dataclass
class CropData:
    """Data for cropping an image."""

    background_color: Float[Tensor, "3"] = torch.Tensor([0.0, 0.0, 0.0])
    """background color"""
    obb: OrientedBox = field(default_factory=lambda: OrientedBox(R=torch.eye(3), T=torch.zeros(3), S=torch.ones(3) * 2))
    """Oriented box representing the crop region"""

    # properties for backwards-compatibility interface
    @property
    def center(self):
        return self.obb.T

    @property
    def scale(self):
        return self.obb.S


def get_crop_from_json(camera_json: Dict[str, Any]) -> Optional[CropData]:
    """Load crop data from a camera path JSON

    args:
        camera_json: camera path data
    returns:
        Crop data
    """
    if "crop" not in camera_json or camera_json["crop"] is None:
        return None
    bg_color = camera_json["crop"]["crop_bg_color"]
    center = camera_json["crop"]["crop_center"]
    scale = camera_json["crop"]["crop_scale"]
    rot = (0.0, 0.0, 0.0) if "crop_rot" not in camera_json["crop"] else tuple(camera_json["crop"]["crop_rot"])
    assert len(center) == 3
    assert len(scale) == 3
    assert len(rot) == 3
    return CropData(
        background_color=torch.Tensor([bg_color["r"] / 255.0, bg_color["g"] / 255.0, bg_color["b"] / 255.0]),
        obb=OrientedBox.from_params(center, rot, scale),
    )


@dataclass
class BaseRender:
    """Base class for rendering."""

    load_config: Path
    """Path to config YAML file."""
    output_path: Path = Path("renders/output.mp4")
    """Path to output video file."""
    image_format: Literal["jpeg", "png"] = "jpeg"
    """Image format"""
    jpeg_quality: int = 100
    """JPEG quality"""
    downscale_factor: float = 1.0
    """Scaling factor to apply to the camera image resolution."""
    eval_num_rays_per_chunk: Optional[int] = None
    """Specifies number of rays per chunk during eval. If None, use the value in the config file."""
    rendered_output_names: List[str] = field(default_factory=lambda: ["rgb"])
    """Name of the renderer outputs to use. rgb, depth, etc. concatenates them along y axis"""
    depth_near_plane: Optional[float] = None
    """Closest depth to consider when using the colormap for depth. If None, use min value."""
    depth_far_plane: Optional[float] = None
    """Furthest depth to consider when using the colormap for depth. If None, use max value."""
    colormap_options: colormaps.ColormapOptions = colormaps.ColormapOptions()
    """Colormap options."""
    render_nearest_camera: bool = False
    """Whether to render the nearest training camera to the rendered camera."""
    check_occlusions: bool = False
    """If true, checks line-of-sight occlusions when computing camera distance and rejects cameras not visible to each other"""


def _infer_scene_id_from_load_config(load_config: Path) -> str:
    """Infer the AV2 scene id from either .../{scene_id}/config.yml or .../{scene_id}/{timestamp}/config.yml."""
    if load_config.name != "config.yml":
        raise ValueError(f"Expected load_config to point to config.yml, got {load_config}")
    if (load_config.parent / "nerfstudio_models").exists():
        return load_config.parent.name
    if (load_config.parent.parent / "nerfstudio_models").exists():
        return load_config.parent.parent.name
    scene_id = load_config.parent.parent.name
    if not scene_id:
        raise ValueError(f"Could not infer scene id from {load_config}")
    return scene_id


def _get_default_stereo_sidecar_path(load_config: Path) -> Path:
    if (load_config.parent / "nerfstudio_models").exists():
        return load_config.parent / "nerfstudio_models" / "sidecars" / "stereo_appearance.pt"
    return load_config.parent.parent / "nerfstudio_models" / "sidecars" / "stereo_appearance.pt"


def _get_default_stereo_photometric_sidecar_path(load_config: Path) -> Path:
    if (load_config.parent / "nerfstudio_models").exists():
        return load_config.parent / "nerfstudio_models" / "sidecars" / "stereo_photometric.pt"
    return load_config.parent.parent / "nerfstudio_models" / "sidecars" / "stereo_photometric.pt"


def _get_default_av2_appearance_sensor(target_camera: str) -> str:
    return DEFAULT_AV2_STEREO_APPEARANCE_SENSORS.get(target_camera, "ring_front_right")


def _load_stereo_appearance_sidecar(sidecar_path: Path) -> Dict[str, Any]:
    sidecar = torch.load(sidecar_path, map_location="cpu")
    if not isinstance(sidecar, dict) or "entries" not in sidecar:
        raise ValueError(f"Invalid stereo appearance sidecar: {sidecar_path}")
    return sidecar


def _get_sidecar_appearance_embedding(
    sidecar: Dict[str, Any], target_camera: str, appearance_dim: int
) -> Optional[torch.Tensor]:
    entries = sidecar.get("entries", {})
    if target_camera not in entries:
        return None
    embedding = entries[target_camera].get("embedding")
    if embedding is None:
        raise ValueError(f"Stereo appearance sidecar entry for {target_camera} is missing an embedding.")
    embedding = torch.as_tensor(embedding, dtype=torch.float32)
    if embedding.ndim != 1 or embedding.shape[0] != appearance_dim:
        raise ValueError(
            f"Stereo appearance sidecar entry for {target_camera} has shape {tuple(embedding.shape)}, "
            f"expected ({appearance_dim},)."
        )
    return embedding


def _save_stereo_appearance_sidecar(
    sidecar_path: Path,
    *,
    load_config: Path,
    sequence: str,
    appearance_dim: int,
    entries: Dict[str, Dict[str, Any]],
    fit_settings: Dict[str, Any],
) -> None:
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "splatad_stereo_appearance_v1",
            "load_config": str(load_config),
            "sequence": sequence,
            "appearance_dim": appearance_dim,
            "entries": entries,
            "fit_settings": fit_settings,
        },
        sidecar_path,
    )


def _load_stereo_photometric_sidecar(sidecar_path: Path) -> Dict[str, Any]:
    sidecar = torch.load(sidecar_path, map_location="cpu")
    if not isinstance(sidecar, dict) or "entries" not in sidecar:
        raise ValueError(f"Invalid stereo photometric sidecar: {sidecar_path}")
    return sidecar


def _save_stereo_photometric_sidecar(
    sidecar_path: Path,
    *,
    load_config: Path,
    sequence: str,
    entries: Dict[str, Dict[str, Any]],
    fit_settings: Dict[str, Any],
) -> None:
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "splatad_stereo_photometric_v1",
            "load_config": str(load_config),
            "sequence": sequence,
            "entries": entries,
            "fit_settings": fit_settings,
        },
        sidecar_path,
    )


def _rgb_to_luminance(rgb: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([0.299, 0.587, 0.114], dtype=rgb.dtype, device=rgb.device)
    return (rgb * weights).sum(dim=-1, keepdim=True)


def _extract_grayscale_target(image: torch.Tensor) -> torch.Tensor:
    if image.shape[-1] == 1:
        return image
    return image[..., :1]


def _apply_affine_grayscale_mapping(rgb: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    gray = torch.clamp(_rgb_to_luminance(rgb) * scale + bias, 0.0, 1.0)
    return gray.expand(*rgb.shape[:-1], 3)


def _build_photometric_postprocess(entry: Dict[str, Any]) -> Callable[[torch.Tensor], torch.Tensor]:
    scale = torch.as_tensor(entry["scale"], dtype=torch.float32)
    bias = torch.as_tensor(entry["bias"], dtype=torch.float32)

    def _postprocess(rgb: torch.Tensor) -> torch.Tensor:
        return _apply_affine_grayscale_mapping(rgb, scale.to(rgb.device), bias.to(rgb.device))

    return _postprocess


def _resolve_av2_split_root(
    data_root: Path, sequence: str, split: Optional[Literal["train", "val", "test", "mini", "mini_val", "mini_test"]]
) -> Tuple[Path, str]:
    candidate_splits = [split] if split is not None else list(AV2_SPLITS)
    for candidate in candidate_splits:
        split_root = data_root / "sensor" / candidate
        if (split_root / sequence).exists():
            return split_root, candidate
    raise FileNotFoundError(f"Could not find AV2 sequence {sequence} under {data_root / 'sensor'}")


def _slice_frame_paths(paths: List[Path], start_frame: int, max_frames: Optional[int]) -> List[Path]:
    if start_frame < 0:
        raise ValueError("start_frame must be >= 0")
    end_frame = None if max_frames is None else start_frame + max_frames
    selected = paths[start_frame:end_frame]
    if not selected:
        raise ValueError("Selected frame range is empty")
    return selected


def _infer_video_seconds_from_times(times: torch.Tensor) -> float:
    if times.numel() <= 1:
        return 1.0
    deltas = times[1:, 0] - times[:-1, 0]
    positive_deltas = deltas[deltas > 0]
    if positive_deltas.numel() == 0:
        return float(times.shape[0])
    fps = 1.0 / torch.median(positive_deltas).item()
    return times.shape[0] / fps


def _read_av2_intrinsics(scene_dir: Path, camera_name: str) -> Tuple[np.ndarray, int, int]:
    intrinsics_df = pd.read_feather(scene_dir / "calibration" / "intrinsics.feather")
    rows = intrinsics_df[intrinsics_df["sensor_name"] == camera_name]
    if len(rows) != 1:
        raise KeyError(camera_name)
    params = rows.iloc[0]
    intrinsics = np.eye(3, dtype=np.float32)
    intrinsics[0, 0] = float(params["fx_px"])
    intrinsics[1, 1] = float(params["fy_px"])
    intrinsics[0, 2] = float(params["cx_px"])
    intrinsics[1, 2] = float(params["cy_px"])
    return intrinsics, int(params["width_px"]), int(params["height_px"])


def _read_av2_sensor_extrinsic(scene_dir: Path, camera_name: str) -> np.ndarray:
    extrinsics_df = pd.read_feather(scene_dir / "calibration" / "egovehicle_SE3_sensor.feather")
    rows = extrinsics_df[extrinsics_df["sensor_name"] == camera_name]
    if len(rows) != 1:
        raise KeyError(camera_name)
    return convert_pose_dataframe_to_SE3(rows).transform_matrix.astype(np.float32)


def _compute_average_av2_camera_delta(split_root: Path, reference_camera: str, target_camera: str) -> Dict[str, Any]:
    cache_key = (str(split_root), reference_camera, target_camera)
    if cache_key in AV2_CAMERA_DELTA_CACHE:
        return AV2_CAMERA_DELTA_CACHE[cache_key]

    deltas = []
    scene_ids = []
    for scene_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
        calibration_dir = scene_dir / "calibration"
        if not calibration_dir.exists():
            continue
        try:
            reference_extrinsic = _read_av2_sensor_extrinsic(scene_dir, reference_camera)
            target_extrinsic = _read_av2_sensor_extrinsic(scene_dir, target_camera)
        except KeyError:
            continue
        deltas.append(torch.tensor(np.linalg.inv(reference_extrinsic) @ target_extrinsic, dtype=torch.float32))
        scene_ids.append(scene_dir.name)

    if not deltas:
        raise ValueError(f"Could not compute fallback delta for {reference_camera} -> {target_camera}")

    deltas_tensor = torch.stack(deltas, dim=0)
    translations = deltas_tensor[:, :3, 3]
    quaternions = rotmat_to_unitquat(deltas_tensor[:, :3, :3])
    reference_quaternion = quaternions[0:1]
    signs = torch.where((quaternions * reference_quaternion).sum(dim=-1, keepdim=True) < 0, -1.0, 1.0)
    mean_quaternion = torch.nn.functional.normalize((quaternions * signs).mean(dim=0, keepdim=True), dim=-1)[0]
    mean_rotation = unitquat_to_rotmat(mean_quaternion.unsqueeze(0))[0].float()

    mean_delta = torch.eye(4, dtype=torch.float32)
    mean_delta[:3, :3] = mean_rotation
    mean_delta[:3, 3] = translations.mean(dim=0)
    result = {
        "transform_matrix": mean_delta,
        "num_scenes_used": len(scene_ids),
        "scene_ids_used": scene_ids,
    }
    AV2_CAMERA_DELTA_CACHE[cache_key] = result
    return result


def _get_av2_camera_calibration(
    split_root: Path,
    sequence: str,
    camera_name: str,
    fallback_reference_camera: Optional[str] = None,
) -> Tuple[np.ndarray, int, int, np.ndarray, Dict[str, Any]]:
    scene_dir = split_root / sequence
    intrinsics, width_px, height_px = _read_av2_intrinsics(scene_dir, camera_name)
    try:
        extrinsic = _read_av2_sensor_extrinsic(scene_dir, camera_name)
        return intrinsics, width_px, height_px, extrinsic, {"used_fallback": False}
    except KeyError:
        if fallback_reference_camera is None:
            raise
        reference_extrinsic = _read_av2_sensor_extrinsic(scene_dir, fallback_reference_camera)
        fallback_delta = _compute_average_av2_camera_delta(split_root, fallback_reference_camera, camera_name)
        extrinsic = (reference_extrinsic @ fallback_delta["transform_matrix"].cpu().numpy()).astype(np.float32)
        return intrinsics, width_px, height_px, extrinsic, {
            "used_fallback": True,
            "reference_camera": fallback_reference_camera,
            "num_scenes_used": fallback_delta["num_scenes_used"],
        }


def _build_av2_target_cameras(
    split_root: Path,
    dataloader: AV2SensorDataLoader,
    sequence: str,
    target_camera: str,
    image_paths: List[Path],
    world_transform: torch.Tensor,
    time_offset: float,
    appearance_sensor_idx: int,
    rolling_shutter_time: float,
    time_to_center_pixel: float,
    appearance_embedding: Optional[torch.Tensor] = None,
    fallback_reference_camera: Optional[str] = None,
) -> Tuple[Cameras, List[int], Dict[str, Any]]:
    intrinsics, width_px, height_px, camera_extrinsics, calibration_info = _get_av2_camera_calibration(
        split_root=split_root,
        sequence=sequence,
        camera_name=target_camera,
        fallback_reference_camera=fallback_reference_camera,
    )
    camera_extrinsics[:3, :3] = camera_extrinsics[:3, :3] @ OPENCV_TO_NERFSTUDIO

    timestamps_ns: List[int] = []
    poses = []
    relative_times = []
    for image_path in image_paths:
        timestamp_ns = int(image_path.stem)
        ego_to_world = dataloader.get_city_SE3_ego(sequence, timestamp_ns)
        raw_camera_to_world = torch.tensor(ego_to_world.transform_matrix @ camera_extrinsics, dtype=torch.float32)
        poses.append((world_transform @ raw_camera_to_world)[:3, :4].float())
        timestamps_ns.append(timestamp_ns)
        relative_times.append((timestamp_ns / 1e9) - time_offset)

    num_frames = len(image_paths)
    poses_tensor = torch.stack(poses, dim=0).float()
    times_tensor = torch.tensor(relative_times, dtype=torch.float32).unsqueeze(-1)
    velocities = torch.zeros((num_frames, 3), dtype=torch.float32)
    linear_velocities_local = torch.zeros((num_frames, 3), dtype=torch.float32)
    angular_velocities_local = torch.zeros((num_frames, 3), dtype=torch.float32)
    if num_frames > 1:
        translation_velo = (poses_tensor[1:, :3, 3] - poses_tensor[:-1, :3, 3]) / (times_tensor[1:] - times_tensor[:-1])
        next_cam = poses_tensor[1:]
        prev_cam = poses_tensor[:-1]
        next_cam_in_prev_cam = pose_utils.to4x4(pose_utils.inverse(prev_cam)) @ pose_utils.to4x4(next_cam)
        translation_velo_cam_ref = next_cam_in_prev_cam[:, :3, 3] / (times_tensor[1:] - times_tensor[:-1])
        angular_velo = pose_utils.rotation_difference(poses_tensor[:-1, :3, :3], poses_tensor[1:, :3, :3]) / (
            times_tensor[1:] - times_tensor[:-1]
        )
        velocities = torch.cat((translation_velo, translation_velo[-1:]), 0).float()
        linear_velocities_local = torch.cat((translation_velo_cam_ref, translation_velo_cam_ref[-1:]), 0).float()
        angular_velocities_local = torch.cat((angular_velo, angular_velo[-1:]), 0).float()

    metadata = {
        "sensor_idxs": torch.full((num_frames, 1), appearance_sensor_idx, dtype=torch.int64),
        "timestamp_ns": torch.tensor(timestamps_ns, dtype=torch.int64).unsqueeze(-1),
        "velocities": velocities,
        "linear_velocities_local": linear_velocities_local,
        "angular_velocities_local": angular_velocities_local,
        "rolling_shutter_time": torch.full((num_frames, 1), rolling_shutter_time, dtype=torch.float32),
        "time_to_center_pixel": torch.full((num_frames, 1), time_to_center_pixel, dtype=torch.float32),
    }
    if appearance_embedding is not None:
        appearance_embedding = appearance_embedding.float()
        metadata["appearance_embedding"] = appearance_embedding.unsqueeze(0).repeat(num_frames, 1)

    cameras = Cameras(
        camera_to_worlds=poses_tensor,
        fx=torch.full((num_frames, 1), float(intrinsics[0, 0]), dtype=torch.float32),
        fy=torch.full((num_frames, 1), float(intrinsics[1, 1]), dtype=torch.float32),
        cx=torch.full((num_frames, 1), float(intrinsics[0, 2]), dtype=torch.float32),
        cy=torch.full((num_frames, 1), float(intrinsics[1, 2]), dtype=torch.float32),
        width=torch.full((num_frames, 1), width_px, dtype=torch.int64),
        height=torch.full((num_frames, 1), height_px, dtype=torch.int64),
        camera_type=CameraType.PERSPECTIVE,
        times=times_tensor,
        metadata=metadata,
    )
    return cameras, timestamps_ns, calibration_info


def _get_av2_camera_to_worlds(
    split_root: Path,
    dataloader: AV2SensorDataLoader,
    sequence: str,
    camera_name: str,
    timestamps_ns: List[int],
    world_transform: torch.Tensor,
    fallback_reference_camera: Optional[str] = None,
) -> torch.Tensor:
    _, _, _, camera_extrinsics, _ = _get_av2_camera_calibration(
        split_root=split_root,
        sequence=sequence,
        camera_name=camera_name,
        fallback_reference_camera=fallback_reference_camera,
    )
    camera_extrinsics[:3, :3] = camera_extrinsics[:3, :3] @ OPENCV_TO_NERFSTUDIO

    poses = []
    for timestamp_ns in timestamps_ns:
        ego_to_world = dataloader.get_city_SE3_ego(sequence, timestamp_ns)
        raw_camera_to_world = torch.tensor(ego_to_world.transform_matrix @ camera_extrinsics, dtype=torch.float32)
        poses.append((world_transform @ raw_camera_to_world)[:3, :4].float())
    return torch.stack(poses, dim=0)


def _compute_reference_camera_delta(
    reference_camera_to_worlds: torch.Tensor,
    target_camera_to_worlds: torch.Tensor,
    timestamps_ns: List[int],
) -> Dict[str, Any]:
    ref_to_target = pose_utils.multiply(pose_utils.inverse(reference_camera_to_worlds), target_camera_to_worlds)
    quaternions_xyzw = rotmat_to_unitquat(ref_to_target[:, :3, :3]).float()

    first_translation = ref_to_target[0, :3, 3]
    translation_deviation = (ref_to_target[:, :3, 3] - first_translation).norm(dim=-1)
    first_rotation = ref_to_target[0, :3, :3].unsqueeze(0).expand(ref_to_target.shape[0], -1, -1)
    rotation_deviation_rad = pose_utils.rotation_difference(first_rotation, ref_to_target[:, :3, :3]).norm(dim=-1)
    rotation_deviation_deg = torch.rad2deg(rotation_deviation_rad)

    per_frame = []
    for idx, timestamp_ns in enumerate(timestamps_ns):
        per_frame.append(
            {
                "timestamp_ns": int(timestamp_ns),
                "translation_xyz_m": [float(v) for v in ref_to_target[idx, :3, 3].tolist()],
                "quaternion_xyzw": [float(v) for v in quaternions_xyzw[idx].tolist()],
                "camera_to_camera_3x4": [[float(v) for v in row] for row in ref_to_target[idx].tolist()],
            }
        )

    return {
        "translation_xyz_m_first": [float(v) for v in first_translation.tolist()],
        "quaternion_xyzw_first": [float(v) for v in quaternions_xyzw[0].tolist()],
        "max_translation_deviation_m": float(translation_deviation.max().item()),
        "max_rotation_deviation_deg": float(rotation_deviation_deg.max().item()),
        "per_frame": per_frame,
    }


def _rename_rendered_images_to_timestamps(
    output_path: Path, image_format: Literal["jpeg", "png"], timestamps_ns: List[int]
) -> Path:
    image_suffix = ".png" if image_format == "png" else ".jpg"
    output_dir = output_path.parent / output_path.stem
    for frame_idx, timestamp_ns in enumerate(timestamps_ns):
        source = output_dir / f"{frame_idx:05d}{image_suffix}"
        target = output_dir / f"{timestamp_ns}{image_suffix}"
        if not source.exists():
            raise FileNotFoundError(f"Expected rendered frame {source} was not created")
        source.rename(target)
    return output_dir


def _resolve_av2_output_path(
    output_path: Path,
    output_format: Literal["images", "video"],
    scene_id: str,
    target_camera: str,
) -> Path:
    if output_path != BaseRender.output_path:
        return output_path
    if output_format == "video":
        return Path("renders") / scene_id / f"{target_camera}.mp4"
    return Path("renders") / scene_id / target_camera


def _write_av2_render_manifest(
    manifest_path: Path,
    *,
    load_config: Path,
    sequence: str,
    data_split: str,
    target_camera: str,
    appearance_sensor: str,
    reference_camera: Optional[str],
    calibration_info: Optional[Dict[str, Any]],
    timestamps_ns: List[int],
    image_paths: List[Path],
    relative_times: torch.Tensor,
    sensor_idx_to_name: Dict[int, str],
    reference_delta: Optional[Dict[str, Any]] = None,
    appearance_sidecar: Optional[Path] = None,
    used_fitted_appearance: bool = False,
    photometric_sidecar: Optional[Path] = None,
    used_photometric_mapping: bool = False,
) -> None:
    manifest = {
        "load_config": str(load_config),
        "sequence": sequence,
        "data_split": data_split,
        "target_camera": target_camera,
        "appearance_sensor": appearance_sensor,
        "reference_camera": reference_camera,
        "calibration_info": calibration_info,
        "num_frames": len(timestamps_ns),
        "timestamps_ns": timestamps_ns,
        "source_images": [str(path) for path in image_paths],
        "relative_times_seconds": [float(value) for value in relative_times[:, 0].tolist()],
        "sensor_idx_to_name": {str(idx): name for idx, name in sensor_idx_to_name.items()},
        "used_fitted_appearance": used_fitted_appearance,
        "used_photometric_mapping": used_photometric_mapping,
    }
    if appearance_sidecar is not None:
        manifest["appearance_sidecar"] = str(appearance_sidecar)
    if photometric_sidecar is not None:
        manifest["photometric_sidecar"] = str(photometric_sidecar)
    if reference_delta is not None:
        manifest["reference_to_target_delta"] = reference_delta
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _load_av2_rgb_image(image_path: Path, height: int, width: int) -> torch.Tensor:
    image = np.asarray(Image.open(image_path), dtype=np.float32)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    image = torch.from_numpy(image[..., :3] / 255.0)
    image = image[:height, :width]
    if image.shape[0] != height or image.shape[1] != width:
        image = torch.nn.functional.interpolate(
            image.permute(2, 0, 1).unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[0].permute(1, 2, 0)
    return image


def _set_camera_appearance_override(camera: Cameras, appearance_embedding: torch.Tensor) -> Cameras:
    camera.metadata = camera.metadata or {}
    camera.metadata["appearance_embedding"] = appearance_embedding.float().unsqueeze(0)
    return camera


@dataclass
class RenderCameraPath(BaseRender):
    """Render a camera path generated by the viewer or blender add-on."""

    camera_path_filename: Path = Path("camera_path.json")
    """Filename of the camera path to render."""
    output_format: Literal["images", "video"] = "video"
    """How to save output data."""

    def main(self) -> None:
        """Main function."""
        _, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="inference",
            update_config_callback=streamline_ad_config,
        )

        install_checks.check_ffmpeg_installed()

        with open(self.camera_path_filename, "r", encoding="utf-8") as f:
            camera_path = json.load(f)
        seconds = camera_path["seconds"]
        crop_data = get_crop_from_json(camera_path)
        camera_path = get_path_from_json(camera_path)

        if (
            camera_path.camera_type[0] == CameraType.OMNIDIRECTIONALSTEREO_L.value
            or camera_path.camera_type[0] == CameraType.VR180_L.value
        ):
            # temp folder for writing left and right view renders
            temp_folder_path = self.output_path.parent / (self.output_path.stem + "_temp")

            Path(temp_folder_path).mkdir(parents=True, exist_ok=True)
            left_eye_path = temp_folder_path / "render_left.mp4"

            self.output_path = left_eye_path

            if camera_path.camera_type[0] == CameraType.OMNIDIRECTIONALSTEREO_L.value:
                CONSOLE.print("[bold green]:goggles: Omni-directional Stereo VR :goggles:")
            else:
                CONSOLE.print("[bold green]:goggles: VR180 :goggles:")

            CONSOLE.print("Rendering left eye view")

        # add mp4 suffix to video output if none is specified
        if self.output_format == "video" and str(self.output_path.suffix) == "":
            self.output_path = self.output_path.with_suffix(".mp4")

        _render_trajectory_video(
            pipeline,
            camera_path,
            output_filename=self.output_path,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            crop_data=crop_data,
            seconds=seconds,
            output_format=self.output_format,
            image_format=self.image_format,
            jpeg_quality=self.jpeg_quality,
            depth_near_plane=self.depth_near_plane,
            depth_far_plane=self.depth_far_plane,
            colormap_options=self.colormap_options,
            render_nearest_camera=self.render_nearest_camera,
            check_occlusions=self.check_occlusions,
        )

        if (
            camera_path.camera_type[0] == CameraType.OMNIDIRECTIONALSTEREO_L.value
            or camera_path.camera_type[0] == CameraType.VR180_L.value
        ):
            # declare paths for left and right renders

            left_eye_path = self.output_path
            right_eye_path = left_eye_path.parent / "render_right.mp4"

            self.output_path = right_eye_path

            if camera_path.camera_type[0] == CameraType.OMNIDIRECTIONALSTEREO_L.value:
                camera_path.camera_type[0] = CameraType.OMNIDIRECTIONALSTEREO_R.value
            else:
                camera_path.camera_type[0] = CameraType.VR180_R.value

            CONSOLE.print("Rendering right eye view")
            _render_trajectory_video(
                pipeline,
                camera_path,
                output_filename=self.output_path,
                rendered_output_names=self.rendered_output_names,
                rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
                crop_data=crop_data,
                seconds=seconds,
                output_format=self.output_format,
                image_format=self.image_format,
                jpeg_quality=self.jpeg_quality,
                depth_near_plane=self.depth_near_plane,
                depth_far_plane=self.depth_far_plane,
                colormap_options=self.colormap_options,
                render_nearest_camera=self.render_nearest_camera,
                check_occlusions=self.check_occlusions,
            )

            self.output_path = Path(str(left_eye_path.parent)[:-5] + ".mp4")

            if camera_path.camera_type[0] == CameraType.OMNIDIRECTIONALSTEREO_R.value:
                # stack the left and right eye renders vertically for ODS final output
                ffmpeg_ods_command = ""
                if self.output_format == "video":
                    ffmpeg_ods_command = f'ffmpeg -y -i "{left_eye_path}" -i "{right_eye_path}" -filter_complex "[0:v]pad=iw:2*ih[int];[int][1:v]overlay=0:h" -c:v libx264 -crf 23 -preset veryfast "{self.output_path}"'
                    run_command(ffmpeg_ods_command, verbose=False)
                if self.output_format == "images":
                    # create a folder for the stacked renders
                    self.output_path = Path(str(left_eye_path.parent)[:-5])
                    self.output_path.mkdir(parents=True, exist_ok=True)
                    if self.image_format == "png":
                        ffmpeg_ods_command = f'ffmpeg -y -pattern_type glob -i "{str(left_eye_path.with_suffix("") / "*.png")}"  -pattern_type glob -i "{str(right_eye_path.with_suffix("") / "*.png")}" -filter_complex vstack -start_number 0 "{str(self.output_path)+"//%05d.png"}"'
                    elif self.image_format == "jpeg":
                        ffmpeg_ods_command = f'ffmpeg -y -pattern_type glob -i "{str(left_eye_path.with_suffix("") / "*.jpg")}"  -pattern_type glob -i "{str(right_eye_path.with_suffix("") / "*.jpg")}" -filter_complex vstack -start_number 0 "{str(self.output_path)+"//%05d.jpg"}"'
                    run_command(ffmpeg_ods_command, verbose=False)

                # remove the temp files directory
                if str(left_eye_path.parent)[-5:] == "_temp":
                    shutil.rmtree(left_eye_path.parent, ignore_errors=True)
                CONSOLE.print("[bold green]Final ODS Render Complete")
            else:
                # stack the left and right eye renders horizontally for VR180 final output
                self.output_path = Path(str(left_eye_path.parent)[:-5] + ".mp4")
                ffmpeg_vr180_command = ""
                if self.output_format == "video":
                    ffmpeg_vr180_command = f'ffmpeg -y -i "{right_eye_path}" -i "{left_eye_path}" -filter_complex "[1:v]hstack=inputs=2" -c:a copy "{self.output_path}"'
                    run_command(ffmpeg_vr180_command, verbose=False)
                if self.output_format == "images":
                    # create a folder for the stacked renders
                    self.output_path = Path(str(left_eye_path.parent)[:-5])
                    self.output_path.mkdir(parents=True, exist_ok=True)
                    if self.image_format == "png":
                        ffmpeg_vr180_command = f'ffmpeg -y -pattern_type glob -i "{str(left_eye_path.with_suffix("") / "*.png")}"  -pattern_type glob -i "{str(right_eye_path.with_suffix("") / "*.png")}" -filter_complex hstack -start_number 0 "{str(self.output_path)+"//%05d.png"}"'
                    elif self.image_format == "jpeg":
                        ffmpeg_vr180_command = f'ffmpeg -y -pattern_type glob -i "{str(left_eye_path.with_suffix("") / "*.jpg")}"  -pattern_type glob -i "{str(right_eye_path.with_suffix("") / "*.jpg")}" -filter_complex hstack -start_number 0 "{str(self.output_path)+"//%05d.jpg"}"'
                    run_command(ffmpeg_vr180_command, verbose=False)

                # remove the temp files directory
                if str(left_eye_path.parent)[-5:] == "_temp":
                    shutil.rmtree(left_eye_path.parent, ignore_errors=True)
                CONSOLE.print("[bold green]Final VR180 Render Complete")


@dataclass
class RenderInterpolated(BaseRender):
    """Render a trajectory that interpolates between training or eval dataset images."""

    pose_source: Literal["eval", "train", "train+eval"] = "eval"
    """Pose source to render."""
    interpolation_steps: int = 10
    """Number of interpolation steps between eval dataset cameras."""
    order_poses: bool = False
    """Whether to order camera poses by proximity."""
    sensor_index: Optional[int] = None
    """Sensor index to render. If None, render all sensors."""
    frame_rate: int = 24
    """Frame rate of the output video."""
    output_format: Literal["images", "video"] = "video"
    """How to save output data."""

    spiral_radius: float = 0.0
    """Radius of the spiral."""
    spiral_rotations: float = 1.0
    """Number of rotations of the spiral."""

    shift: Tuple[float, ...] = (0.0, 0.0, 0.0)
    """Shift to apply to the camera pose."""
    shift_time: float = 0.0
    """Time at which to apply the shift."""
    shift_steps: int = 0
    """Number of steps to interpolate the shift over."""

    actor_shift: Tuple[float, ...] = (0.0, 0.0, 0.0)
    """Shift to apply to all actor poses."""
    actor_removal_time: Optional[float] = None
    """Time at which to remove all actors."""
    actor_stop_time: Optional[float] = None
    """Time at which to stop all actors."""
    actor_indices: Optional[List[int]] = None
    """Indices of actors to modify. If None, modify all actors."""

    def main(self) -> None:
        """Main function."""
        _, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="test",
            update_config_callback=streamline_ad_config,
        )

        if self.spiral_radius and any(self.shift):
            CONSOLE.print(
                "Warning: Rendering with both spiral and shift is not supported. Only spiral will be rendered."
            )

        install_checks.check_ffmpeg_installed()

        if self.pose_source == "eval":
            assert pipeline.datamanager.eval_dataset is not None
            cameras = pipeline.datamanager.eval_dataset.cameras
        elif self.pose_source == "train":
            assert pipeline.datamanager.train_dataset is not None
            cameras = pipeline.datamanager.train_dataset.cameras
        else:
            assert pipeline.datamanager.train_dataset is not None
            assert pipeline.datamanager.eval_dataset is not None
            cameras = pipeline.datamanager.train_dataset.cameras.cat([pipeline.datamanager.eval_dataset.cameras])

        if cameras.times is not None:
            cameras = cameras[torch.argsort(cameras.times.squeeze(-1))]

        if cameras.metadata and "sensor_idxs" in cameras.metadata:
            sensor_indices = (
                torch.tensor(self.sensor_index).unsqueeze(0)
                if self.sensor_index is not None
                else cameras.metadata["sensor_idxs"].unique()
            )
        else:
            sensor_indices = torch.tensor([0]).unsqueeze(0)
            cameras.metadata["sensor_idxs"] = torch.zeros_like(cameras.camera_type, dtype=torch.int64)

        modify_actors(pipeline, self.actor_shift, self.actor_removal_time, self.actor_stop_time, self.actor_indices)

        for sensor_index in sensor_indices:
            mask = (cameras.metadata["sensor_idxs"] == sensor_index).squeeze(-1)
            curr_cameras = cameras[mask]

            seconds = self.interpolation_steps * len(curr_cameras) / self.frame_rate
            if self.spiral_radius:
                camera_path = get_interpolated_spiral_camera_path(
                    cameras=curr_cameras,
                    steps=self.interpolation_steps,
                    radius=self.spiral_radius,
                    rotations=self.spiral_rotations,
                )
            elif any(self.shift):
                camera_path = get_shifted_camera_path(
                    cameras=curr_cameras,
                    shift=self.shift,
                    shift_time=self.shift_time,
                    shift_steps=self.shift_steps,
                    interpolation_steps=self.interpolation_steps,
                )
            else:
                camera_path = get_interpolated_camera_path(
                    cameras=curr_cameras,
                    steps=self.interpolation_steps,
                    order_poses=self.order_poses,
                )
                if curr_cameras.times is not None:
                    times, stepsize = curr_cameras.times[..., 0], 1.0 / self.interpolation_steps
                    camera_path.times = torch.from_numpy(
                        np.interp(
                            np.append(np.arange(0, len(times) - 1, stepsize), len(times) - 1),
                            np.arange(len(times)),
                            times,
                        )[..., None]
                    ).float()

            camera_path.metadata = camera_path.metadata or {}
            if curr_cameras.metadata and "sensor_idxs" in curr_cameras.metadata:
                camera_path.metadata["sensor_idxs"] = torch.full_like(camera_path.width, sensor_index.item())

            print("Rendering sensor index ", sensor_index.item())
            _render_trajectory_video(
                pipeline,
                camera_path,
                output_filename=self.output_path.with_stem(f"{self.output_path.stem}_sensor_idx_{sensor_index}"),
                rendered_output_names=self.rendered_output_names,
                rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
                seconds=seconds,
                output_format=self.output_format,
                image_format=self.image_format,
                depth_near_plane=self.depth_near_plane,
                depth_far_plane=self.depth_far_plane,
                colormap_options=self.colormap_options,
                render_nearest_camera=self.render_nearest_camera,
                check_occlusions=self.check_occlusions,
            )


def modify_actors(pipeline, actor_shift, actor_removal_time, actor_stop_time, actor_indices):
    actor_shift = torch.nn.Parameter(torch.tensor(actor_shift, dtype=torch.float32, device=pipeline.model.device))
    with torch.no_grad():
        if actor_indices is not None:
            indices = torch.tensor(actor_indices, device=pipeline.model.device, dtype=torch.int)
        else:
            indices = torch.arange(pipeline.model.dynamic_actors.actor_positions.shape[1], device=pipeline.model.device)

        pipeline.model.dynamic_actors.actor_positions[:, indices, :] += actor_shift
        if actor_removal_time is not None:
            no_actor_mask = pipeline.model.dynamic_actors.unique_timestamps > actor_removal_time
            pipeline.model.dynamic_actors.actor_present_at_time[no_actor_mask, indices] = False
        if actor_stop_time is not None:
            actor_stop_idx = torch.searchsorted(pipeline.model.dynamic_actors.unique_timestamps, actor_stop_time)
            freeze_position = pipeline.model.dynamic_actors.actor_positions[actor_stop_idx, indices].unsqueeze(0)
            freeze_rotation = pipeline.model.dynamic_actors.actor_rotations_6d[actor_stop_idx, indices].unsqueeze(0)
            pipeline.model.dynamic_actors.actor_positions[actor_stop_idx:, indices] = freeze_position
            pipeline.model.dynamic_actors.actor_rotations_6d[actor_stop_idx:, indices] = freeze_rotation


def get_shifted_camera_path(cameras, shift, shift_time, shift_steps, interpolation_steps):
    if cameras.times is not None:
        # find index of time closest to shift_time
        shift_idx = torch.argmin(torch.abs(cameras.times - shift_time))
    else:
        # warn user that we are assuming shift_time is the middle of the trajectory
        CONSOLE.print(
            "Warning: Assuming shift_time is the middle of the trajectory. "
            "If this is not the case, please specify times in the camera path JSON."
        )
        shift_idx = int(shift_time * len(cameras))
    pre_shift_cams = cameras[:shift_idx]
    post_shift_cams = cameras[shift_idx:]
    post_shift_cams.camera_to_worlds = post_shift_cams.camera_to_worlds.clone()
    post_shift_cams.camera_to_worlds[..., :3, 3] = post_shift_cams.camera_to_worlds[..., :3, 3] + torch.tensor(shift)
    post_shift_camera_path = get_interpolated_camera_path(post_shift_cams, steps=interpolation_steps, order_poses=False)
    if (times := post_shift_cams.times) is not None:
        post_shift_camera_path.times = torch.from_numpy(
            np.interp(
                np.append(np.arange(0, len(times) - 1, 1 / interpolation_steps), len(times) - 1),
                np.arange(len(times)),
                times.squeeze(-1),
            )[..., None]
        ).float()

    if len(pre_shift_cams) == 0:
        return post_shift_camera_path

    pre_shift_camera_path = get_interpolated_camera_path(pre_shift_cams, steps=interpolation_steps, order_poses=False)
    mid_shift_camera_path = get_interpolated_camera_path(
        pre_shift_cams[-1:].cat([post_shift_cams[:1]]), steps=shift_steps, order_poses=False
    )
    if (times := pre_shift_cams.times) is not None:
        pre_shift_camera_path.times = torch.from_numpy(
            np.interp(
                np.append(np.arange(0, len(times) - 1, 1 / interpolation_steps), len(times) - 1),
                np.arange(len(times)),
                times.squeeze(-1),
            )[..., None]
        ).float()
        mid_shift_camera_path.times = torch.full_like(mid_shift_camera_path.cy, pre_shift_camera_path.times[-1].item())
    return pre_shift_camera_path.cat([mid_shift_camera_path, post_shift_camera_path])


@dataclass
class SpiralRender(BaseRender):
    """Render a spiral trajectory (often not great)."""

    seconds: float = 3.0
    """How long the video should be."""
    output_format: Literal["images", "video"] = "video"
    """How to save output data."""
    frame_rate: int = 24
    """Frame rate of the output video (only for interpolate trajectory)."""
    radius: float = 0.1
    """Radius of the spiral."""

    def main(self) -> None:
        """Main function."""
        _, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="test",
            update_config_callback=streamline_ad_config,
        )

        install_checks.check_ffmpeg_installed()

        assert isinstance(
            pipeline.datamanager,
            (
                VanillaDataManager,
                ParallelDataManager,
                RandomCamerasDataManager,
            ),
        )
        steps = int(self.frame_rate * self.seconds)
        camera_start, _ = pipeline.datamanager.eval_dataloader.get_camera(image_idx=0)
        camera_path = get_spiral_path(camera_start, steps=steps, radius=self.radius)

        _render_trajectory_video(
            pipeline,
            camera_path,
            output_filename=self.output_path,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            seconds=self.seconds,
            output_format=self.output_format,
            image_format=self.image_format,
            depth_near_plane=self.depth_near_plane,
            depth_far_plane=self.depth_far_plane,
            colormap_options=self.colormap_options,
            render_nearest_camera=self.render_nearest_camera,
            check_occlusions=self.check_occlusions,
        )


@contextmanager
def _disable_datamanager_setup(cls):
    """
    Disables setup_train or setup_eval for faster initialization.
    """
    old_setup_train = getattr(cls, "setup_train")
    old_setup_eval = getattr(cls, "setup_eval")
    setattr(cls, "setup_train", lambda *args, **kwargs: None)
    setattr(cls, "setup_eval", lambda *args, **kwargs: None)
    yield cls
    setattr(cls, "setup_train", old_setup_train)
    setattr(cls, "setup_eval", old_setup_eval)


@dataclass
class DatasetRender(BaseRender):
    """Render all images in the dataset."""

    pose_source: Literal["train", "val", "test", "train+test", "train+val"] = "test"
    """Split to render."""
    output_path: Path = Path("renders")
    """Path to output video file."""
    data: Optional[Path] = None
    """Override path to the dataset."""
    config_output_dir: Optional[Path] = None
    """Override the config output dir. Used to load the model."""
    downscale_factor: Optional[float] = None
    """Scaling factor to apply to the camera image resolution."""
    rendered_output_names: List[str] = field(default_factory=lambda: ["all"])
    """Name of the renderer outputs to use. rgb, depth, raw-depth, gt-rgb etc. By default all outputs are rendered."""
    strict_load: bool = True
    """Whether to strictly load the config."""
    load_ignore_keys: Optional[List[str]] = field(
        default_factory=lambda: []
    )  # e.g. ["model.camera_optimizer.pose_adjustment", "_model.camera_optimizer.pose_adjustment"]
    """Keys to ignore when loading the config."""

    render_height: Optional[int] = None
    """Height to render the images at."""
    render_width: Optional[int] = None
    """Width to render the images at."""
    output_height: Optional[int] = None
    """Height to crop the output images at."""
    output_width: Optional[int] = None
    """Width to crop the output images at."""
    max_images: Optional[int] = None
    """Maximum number of images to render per split. If None, render all."""

    shift: Tuple[float, float, float] = (0, 0, 0)
    """Shift to apply to the camera pose."""

    actor_shift: Tuple[float, ...] = (0.0, 0.0, 0.0)
    """Shift to apply to all actor poses."""
    actor_removal_time: Optional[float] = None
    """Time at which to remove all actors."""
    actor_stop_time: Optional[float] = None
    """Time at which to stop all actors."""
    actor_indices: Optional[List[int]] = None
    """Indices of actors to modify. If None, modify all actors."""

    calculate_and_save_metrics: bool = False
    """Whether to calculate and save metrics."""
    metrics_filename: Path = Path("metrics.pkl")
    """Filename to save the metrics to."""

    render_point_clouds: bool = False
    """Whether to render point clouds."""

    def main(self):
        config: TrainerConfig

        def update_config(config: TrainerConfig) -> TrainerConfig:
            config = streamline_ad_config(config)
            data_manager_config = config.pipeline.datamanager
            assert isinstance(data_manager_config, (VanillaDataManagerConfig, FullImageDatamanagerConfig))
            data_manager_config.eval_num_images_to_sample_from = -1
            data_manager_config.eval_num_times_to_repeat_images = -1
            if isinstance(data_manager_config, VanillaDataManagerConfig):
                data_manager_config.train_num_images_to_sample_from = -1
                data_manager_config.train_num_times_to_repeat_images = -1
            if self.data is not None:
                data_manager_config.data = self.data
            if self.config_output_dir is not None:
                config.output_dir = self.config_output_dir
            if self.downscale_factor is not None:
                assert hasattr(data_manager_config.dataparser, "downscale_factor")
                setattr(data_manager_config.dataparser, "downscale_factor", self.downscale_factor)
            # Remove any frame limit on the the dataparser
            config.pipeline.datamanager.dataparser.max_eval_frames = None
            return config

        config, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="inference",
            update_config_callback=update_config,
            strict_load=self.strict_load,
            ignore_keys=self.load_ignore_keys,
        )
        data_manager_config = config.pipeline.datamanager
        assert isinstance(data_manager_config, (VanillaDataManagerConfig, FullImageDatamanagerConfig))

        modify_actors(pipeline, self.actor_shift, self.actor_removal_time, self.actor_stop_time, self.actor_indices)

        self.output_path.mkdir(exist_ok=True, parents=True)
        metrics_out = dict()
        for split in self.pose_source.split("+"):
            datamanager: VanillaDataManager
            dataset: Dataset
            if split == "train":
                with _disable_datamanager_setup(data_manager_config._target):  # pylint: disable=protected-access
                    datamanager = data_manager_config.setup(test_mode="test", device=pipeline.device)

                dataset = datamanager.train_dataset
                dataparser_outputs = getattr(dataset, "_dataparser_outputs", datamanager.train_dataparser_outputs)
                lidar_dataset = datamanager.train_lidar_dataset
            else:
                with _disable_datamanager_setup(data_manager_config._target):  # pylint: disable=protected-access
                    datamanager = data_manager_config.setup(test_mode=split, device=pipeline.device)

                dataset = datamanager.eval_dataset
                dataparser_outputs = getattr(dataset, "_dataparser_outputs", None)
                lidar_dataset = datamanager.eval_lidar_dataset
                if dataparser_outputs is None:
                    dataparser_outputs = datamanager.dataparser.get_dataparser_outputs(split=datamanager.test_split)
            dataset.cameras.height = (
                torch.full_like(dataset.cameras.height, self.render_height)
                if self.render_height is not None
                else dataset.cameras.height
            )
            dataset.cameras.width = (
                torch.full_like(dataset.cameras.width, self.render_width)
                if self.render_width is not None
                else dataset.cameras.width
            )
            shift_relative_to_cam = torch.tensor(self.shift, dtype=torch.float32)
            # add homogenous point
            shift_relative_to_cam = torch.cat([shift_relative_to_cam, torch.tensor([1.0], dtype=torch.float32)])
            shift_relative_to_cam = shift_relative_to_cam.to(dataset.cameras.camera_to_worlds.device)
            # shift the camera poses
            dataset.cameras.camera_to_worlds[..., :3, 3:4] = (
                dataset.cameras.camera_to_worlds @ shift_relative_to_cam.reshape(1, 4, 1)
            )

            dataloader = FixedIndicesEvalDataloader(
                dataset=dataset,
                device=datamanager.device,
                num_workers=datamanager.world_size * 4,
            )
            lidar_dataloader = FixedIndicesEvalDataloader(
                dataset=lidar_dataset,
                device=datamanager.device,
                num_workers=datamanager.world_size * 4,
            )
            images_root = Path(os.path.commonpath(dataparser_outputs.image_filenames))
            with Progress(
                TextColumn(f":movie_camera: Rendering split {split} :movie_camera:"),
                BarColumn(),
                TaskProgressColumn(
                    text_format="[progress.percentage]{task.completed}/{task.total:>.0f}({task.percentage:>3.1f}%)",
                    show_speed=True,
                ),
                ItersPerSecColumn(suffix="fps"),
                TimeRemainingColumn(elapsed_when_finished=False, compact=False),
                TimeElapsedColumn(),
            ) as progress:
                for camera_idx, (camera, batch) in enumerate(progress.track(dataloader, total=len(dataset))):
                    if self.max_images is not None and camera_idx >= self.max_images:
                        break
                    # Try to get the original filename
                    image_name = (
                        Path(dataparser_outputs.image_filenames[camera_idx]).with_suffix("").relative_to(images_root)
                    )

                    with torch.no_grad():
                        outputs = pipeline.model.get_outputs_for_camera(camera)

                    if self.output_height is not None:
                        dataset.cameras.height[batch["image_idx"]] = torch.full_like(
                            dataset.cameras.height[0:1], self.output_height
                        )
                        batch["image"] = batch["image"][..., : self.output_height, :, :]
                        outputs["rgb"] = outputs["rgb"][..., : self.output_height, :, :]

                    if self.output_width is not None:
                        dataset.cameras.width[batch["image_idx"]] = torch.full_like(
                            dataset.cameras.width[0:1], self.output_width
                        )
                        batch["image"] = batch["image"][..., : self.output_width, :]
                        outputs["rgb"] = outputs["rgb"][..., : self.output_width, :]

                    if self.calculate_and_save_metrics:
                        with torch.no_grad():
                            metrics_dict, _ = pipeline.model.get_image_metrics_and_images(outputs, batch)
                            metrics_out[str(image_name)] = metrics_dict

                    gt_batch = batch.copy()
                    gt_batch["rgb"] = gt_batch.pop("image")
                    all_outputs = (
                        list(outputs.keys())
                        + [f"raw-{x}" for x in outputs.keys()]
                        + [f"gt-{x}" for x in gt_batch.keys()]
                        + [f"raw-gt-{x}" for x in gt_batch.keys()]
                    )
                    rendered_output_names = self.rendered_output_names
                    if "all" in rendered_output_names:
                        rendered_output_names = ["gt-rgb"] + list(outputs.keys())
                    elif rendered_output_names == ["none"]:
                        rendered_output_names = []
                    for rendered_output_name in rendered_output_names:
                        if rendered_output_name not in all_outputs:
                            CONSOLE.rule("Error", style="red")
                            CONSOLE.print(
                                f"Could not find {rendered_output_name} in the model outputs", justify="center"
                            )
                            CONSOLE.print(
                                f"Please set --rendered-output-name to one of: {all_outputs}", justify="center"
                            )
                            sys.exit(1)

                        is_raw = False
                        is_depth = rendered_output_name.find("depth") != -1

                        output_path = self.output_path / split / rendered_output_name / image_name
                        output_path.parent.mkdir(exist_ok=True, parents=True)

                        output_name = rendered_output_name
                        if output_name.startswith("raw-"):
                            output_name = output_name[4:]
                            is_raw = True
                            if output_name.startswith("gt-"):
                                output_name = output_name[3:]
                                output_image = gt_batch[output_name]
                            else:
                                output_image = outputs[output_name]
                                if is_depth:
                                    # Divide by the dataparser scale factor
                                    output_image.div_(dataparser_outputs.dataparser_scale)
                        else:
                            if output_name.startswith("gt-"):
                                output_name = output_name[3:]
                                output_image = gt_batch[output_name]
                            else:
                                output_image = outputs[output_name]
                        del output_name

                        # Map to color spaces / numpy
                        if is_raw:
                            output_image = output_image.cpu().numpy()
                        elif is_depth:
                            output_image = (
                                colormaps.apply_depth_colormap(
                                    output_image,
                                    accumulation=outputs["accumulation"],
                                    near_plane=self.depth_near_plane,
                                    far_plane=self.depth_far_plane,
                                    colormap_options=self.colormap_options,
                                )
                                .cpu()
                                .numpy()
                            )
                        else:
                            output_image = (
                                colormaps.apply_colormap(
                                    image=output_image,
                                    colormap_options=self.colormap_options,
                                )
                                .cpu()
                                .numpy()
                            )

                        # Save to file
                        height = (
                            min(output_image.shape[0], self.output_height)
                            if self.output_height
                            else output_image.shape[0]
                        )
                        width = (
                            min(output_image.shape[1], self.output_width)
                            if self.output_width
                            else output_image.shape[1]
                        )
                        output_image = output_image[:height, :width]
                        if is_raw:
                            with gzip.open(output_path.parent / (output_path.name + ".npy.gz"), "wb") as f:
                                np.save(f, output_image)
                        elif self.image_format == "png":
                            media.write_image(output_path.parent / (output_path.name + ".png"), output_image, fmt="png")
                        elif self.image_format == "jpeg":
                            media.write_image(
                                output_path.parent / (output_path.name + ".jpg"),
                                output_image,
                                fmt="jpeg",
                                quality=self.jpeg_quality,
                            )
                        else:
                            raise ValueError(f"Unknown image format {self.image_format}")

            if self.render_point_clouds:
                with Progress(
                    TextColumn(f":movie_camera: Rendering lidars for split {split} :movie_camera:"),
                    BarColumn(),
                    TaskProgressColumn(
                        text_format="[progress.percentage]{task.completed}/{task.total:>.0f}({task.percentage:>3.1f}%)",
                        show_speed=True,
                    ),
                    ItersPerSecColumn(suffix="fps"),
                    TimeRemainingColumn(elapsed_when_finished=False, compact=False),
                    TimeElapsedColumn(),
                ) as progress:
                    with torch.no_grad():
                        output_path = self.output_path / split / "lidar"
                        output_path.mkdir(exist_ok=True, parents=True)
                        for lidar_idx, (lidar, batch) in enumerate(
                            progress.track(lidar_dataloader, total=len(lidar_dataloader))
                        ):
                            lidar_output, _ = pipeline.model.get_outputs_for_lidar(lidar, batch=batch)
                            points_in_local = lidar_output["points"]
                            if "ray_drop_prob" in lidar_output:
                                points_in_local = points_in_local[(lidar_output["ray_drop_prob"] < 0.5).squeeze(-1)]

                            points_in_world = transform_points(points_in_local, lidar.lidar_to_worlds[0])
                            # get ground truth for comparison
                            gt_point_in_world = transform_points(batch["lidar"][..., :3], lidar.lidar_to_worlds[0])
                            plot_lidar_points(
                                gt_point_in_world.cpu().detach().numpy(), output_path / f"gt-lidar_{lidar_idx}.png"
                            )
                            plot_lidar_points(
                                points_in_world.cpu().detach().numpy(), output_path / f"lidar_{lidar_idx}.png"
                            )

        if self.calculate_and_save_metrics:
            metrics_out_path = Path(self.output_path, self.metrics_filename)
            with open(metrics_out_path, "wb") as f:
                pickle.dump(metrics_out, f)
            CONSOLE.print(f"[bold][green]:glowing_star: Metrics saved to {metrics_out_path}")

        table = Table(
            title=None,
            show_header=False,
            box=box.MINIMAL,
            title_style=style.Style(bold=True),
        )
        for split in self.pose_source.split("+"):
            table.add_row(f"Outputs {split}", str(self.output_path / split))
        CONSOLE.print(Panel(table, title="[bold][green]:tada: Render on split {} Complete :tada:[/bold]", expand=False))


@dataclass
class RenderAV2TargetCamera(BaseRender):
    """Render an optimized scene from an AV2 target camera trajectory."""

    sequence: str = ""
    """AV2 sequence UUID to render. Defaults to the scene id inferred from load_config."""

    target_camera: str = "stereo_front_right"
    """AV2 camera stream to use for intrinsics and poses."""

    appearance_sensor: str = "ring_front_right"
    """Training camera whose appearance embedding should be reused."""

    appearance_sidecar: Optional[Path] = None
    """Optional sidecar file containing fitted stereo appearance embeddings."""

    photometric_sidecar: Optional[Path] = None
    """Optional sidecar file containing fitted stereo photometric grayscale mappings."""

    reference_camera: Optional[str] = None
    """Optional camera used when exporting rigid camera-to-camera deltas. Defaults to appearance_sensor."""

    data_root: Path = Path("data/av2")
    """Root directory containing the AV2 dataset."""

    split: Optional[Literal["train", "val", "test", "mini", "mini_val", "mini_test"]] = None
    """Dataset split override. If omitted, infer the split by searching the dataset."""

    start_frame: int = 0
    """First frame index to render."""

    max_frames: Optional[int] = None
    """Maximum number of frames to render."""

    output_format: Literal["images", "video"] = "images"
    """How to save output data."""

    video_seconds: Optional[float] = None
    """Override video duration. If omitted, infer it from AV2 timestamps."""

    def main(self) -> None:
        scene_id = _infer_scene_id_from_load_config(self.load_config)
        sequence = self.sequence or scene_id
        self.output_path = _resolve_av2_output_path(self.output_path, self.output_format, scene_id, self.target_camera)

        _, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="inference",
            update_config_callback=streamline_ad_config,
            strict_load=False,
        )

        dataparser_outputs = pipeline.datamanager.train_dataparser_outputs
        sensor_idx_to_name = dataparser_outputs.metadata["sensor_idx_to_name"]
        sensor_name_to_idx = {name: idx for idx, name in sensor_idx_to_name.items()}
        default_appearance_sensor = _get_default_av2_appearance_sensor(self.target_camera)
        photometric_sidecar = None
        photometric_entry = None
        if self.photometric_sidecar is not None:
            photometric_sidecar = _load_stereo_photometric_sidecar(self.photometric_sidecar)
            photometric_entry = photometric_sidecar.get("entries", {}).get(self.target_camera)
            donor_sensor = photometric_entry.get("appearance_sensor") if photometric_entry is not None else None
            if donor_sensor is not None and self.appearance_sensor in {"ring_front_right", default_appearance_sensor}:
                self.appearance_sensor = donor_sensor
        if self.appearance_sensor == "ring_front_right" and self.target_camera in DEFAULT_AV2_STEREO_APPEARANCE_SENSORS:
            self.appearance_sensor = _get_default_av2_appearance_sensor(self.target_camera)
        if self.appearance_sensor not in sensor_name_to_idx:
            available = ", ".join(sensor_name_to_idx)
            raise ValueError(f"appearance_sensor {self.appearance_sensor!r} not found. Available sensors: {available}")

        split_root, data_split = _resolve_av2_split_root(self.data_root, sequence, self.split)
        dataloader = AV2SensorDataLoader(split_root, split_root)
        image_paths = dataloader.get_ordered_log_cam_fpaths(sequence, self.target_camera)
        selected_image_paths = _slice_frame_paths(image_paths, self.start_frame, self.max_frames)

        world_transform = pose_utils.to4x4(dataparser_outputs.dataparser_transform).cpu().float()
        time_offset = float(dataparser_outputs.time_offset)
        appearance_sensor_idx = int(sensor_name_to_idx[self.appearance_sensor])
        appearance_override = None
        used_fitted_appearance = False
        rgb_postprocess_fn = None
        used_photometric_mapping = False
        if self.appearance_sidecar is not None:
            sidecar = _load_stereo_appearance_sidecar(self.appearance_sidecar)
            appearance_override = _get_sidecar_appearance_embedding(
                sidecar, self.target_camera, pipeline.model.config.appearance_dim
            )
            used_fitted_appearance = appearance_override is not None
        if photometric_entry is not None:
            donor_sensor = photometric_entry.get("appearance_sensor")
            if donor_sensor is not None and donor_sensor != self.appearance_sensor:
                CONSOLE.print(
                    f"[yellow]Photometric sidecar for {self.target_camera} was fit using {donor_sensor}, "
                    f"but render is using {self.appearance_sensor} appearance.[/yellow]"
                )
            rgb_postprocess_fn = _build_photometric_postprocess(photometric_entry)
            used_photometric_mapping = True
        dataparser_config = pipeline.datamanager.dataparser.config
        cameras, timestamps_ns, calibration_info = _build_av2_target_cameras(
            split_root=split_root,
            dataloader=dataloader,
            sequence=sequence,
            target_camera=self.target_camera,
            image_paths=selected_image_paths,
            world_transform=world_transform,
            time_offset=time_offset,
            appearance_sensor_idx=appearance_sensor_idx,
            rolling_shutter_time=float(dataparser_config.rolling_shutter_time),
            time_to_center_pixel=float(dataparser_config.time_to_center_pixel),
            appearance_embedding=appearance_override,
            fallback_reference_camera=self.reference_camera or self.appearance_sensor,
        )
        reference_camera = self.reference_camera or self.appearance_sensor
        reference_delta = None
        if reference_camera:
            reference_camera_to_worlds = _get_av2_camera_to_worlds(
                split_root=split_root,
                dataloader=dataloader,
                sequence=sequence,
                camera_name=reference_camera,
                timestamps_ns=timestamps_ns,
                world_transform=world_transform,
                fallback_reference_camera=reference_camera,
            )
            reference_delta = _compute_reference_camera_delta(
                reference_camera_to_worlds=reference_camera_to_worlds,
                target_camera_to_worlds=cameras.camera_to_worlds,
                timestamps_ns=timestamps_ns,
            )

        CONSOLE.print(
            f"Rendering {len(selected_image_paths)} frames from {self.target_camera} using {self.appearance_sensor} appearance"
        )
        if used_fitted_appearance:
            CONSOLE.print(f"Using fitted appearance override from {self.appearance_sidecar}")
        if used_photometric_mapping:
            CONSOLE.print(f"Using fitted photometric mapping from {self.photometric_sidecar}")
        CONSOLE.print(f"First timestamp: {timestamps_ns[0]}, last timestamp: {timestamps_ns[-1]}, split: {data_split}")
        if calibration_info.get("used_fallback", False):
            CONSOLE.print(
                f"Using fallback extrinsic for {self.target_camera} from {calibration_info['reference_camera']} "
                f"(estimated from {calibration_info['num_scenes_used']} scenes)"
            )
        if reference_delta is not None:
            CONSOLE.print(
                f"{reference_camera} -> {self.target_camera} translation (m): "
                f"{reference_delta['translation_xyz_m_first']}"
            )
            CONSOLE.print(
                f"{reference_camera} -> {self.target_camera} max drift: "
                f"{reference_delta['max_translation_deviation_m']:.6f} m, "
                f"{reference_delta['max_rotation_deviation_deg']:.6f} deg"
            )

        if self.output_format == "video" and str(self.output_path.suffix) == "":
            self.output_path = self.output_path.with_suffix(".mp4")

        seconds = self.video_seconds if self.video_seconds is not None else _infer_video_seconds_from_times(cameras.times)
        _render_trajectory_video(
            pipeline,
            cameras,
            output_filename=self.output_path,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            seconds=seconds,
            output_format=self.output_format,
            image_format=self.image_format,
            jpeg_quality=self.jpeg_quality,
            depth_near_plane=self.depth_near_plane,
            depth_far_plane=self.depth_far_plane,
            colormap_options=self.colormap_options,
            render_nearest_camera=self.render_nearest_camera,
            check_occlusions=self.check_occlusions,
            rgb_postprocess_fn=rgb_postprocess_fn,
        )

        if self.output_format == "images":
            manifest_dir = _rename_rendered_images_to_timestamps(self.output_path, self.image_format, timestamps_ns)
            manifest_path = manifest_dir / "render_manifest.json"
        else:
            manifest_path = self.output_path.with_suffix(".manifest.json")

        _write_av2_render_manifest(
            manifest_path,
            load_config=self.load_config,
            sequence=sequence,
            data_split=data_split,
            target_camera=self.target_camera,
            appearance_sensor=self.appearance_sensor,
            reference_camera=reference_camera,
            calibration_info=calibration_info,
            timestamps_ns=timestamps_ns,
            image_paths=selected_image_paths,
            relative_times=cameras.times,
            sensor_idx_to_name={int(idx): name for idx, name in sensor_idx_to_name.items()},
            reference_delta=reference_delta,
            appearance_sidecar=self.appearance_sidecar,
            used_fitted_appearance=used_fitted_appearance,
            photometric_sidecar=self.photometric_sidecar,
            used_photometric_mapping=used_photometric_mapping,
        )
        CONSOLE.print(f"[bold][green]:glowing_star: Render manifest saved to {manifest_path}")


@dataclass
class FitAV2StereoAppearance:
    """Fit stereo-only appearance vectors for an AV2 scene without changing the base checkpoint."""

    load_config: Path
    """Path to the base config YAML file."""

    output_path: Optional[Path] = None
    """Optional sidecar output path. Defaults to <run-dir>/sidecars/stereo_appearance.pt."""

    sequence: str = ""
    """AV2 sequence UUID to fit. Defaults to the scene id inferred from load_config."""

    target_cameras: Tuple[str, ...] = ("stereo_front_left", "stereo_front_right")
    """AV2 camera streams to fit appearance vectors for."""

    appearance_sensors: Tuple[str, ...] = tuple()
    """Optional donor sensors used to initialize each target camera embedding."""

    data_root: Path = Path("data/av2")
    """Root directory containing the AV2 dataset."""

    split: Optional[Literal["train", "val", "test", "mini", "mini_val", "mini_test"]] = None
    """Dataset split override. If omitted, infer the split by searching the dataset."""

    start_frame: int = 0
    """First frame index to use for fitting."""

    max_frames: Optional[int] = 24
    """Maximum number of frames to use per stereo camera."""

    downscale_factor: float = 4.0
    """Downscale factor applied during fitting."""

    max_steps: int = 400
    """Number of optimization steps."""

    learning_rate: float = 5e-2
    """Learning rate for the stereo appearance vectors."""

    eval_num_rays_per_chunk: Optional[int] = None
    """Optional eval chunk size override."""

    log_every: int = 25
    """How often to print progress."""

    seed: int = 42
    """Random seed for frame sampling."""

    def main(self) -> None:
        scene_id = _infer_scene_id_from_load_config(self.load_config)
        sequence = self.sequence or scene_id
        output_path = self.output_path or _get_default_stereo_sidecar_path(self.load_config)

        torch.manual_seed(self.seed)

        _, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="inference",
            update_config_callback=streamline_ad_config,
            strict_load=False,
        )

        model = pipeline.model
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

        if not hasattr(model, "appearance_embedding"):
            raise TypeError("Stereo appearance fitting currently only supports models with an appearance_embedding table.")

        appearance_dim = int(model.config.appearance_dim)
        dataparser_outputs = pipeline.datamanager.train_dataparser_outputs
        sensor_idx_to_name = dataparser_outputs.metadata["sensor_idx_to_name"]
        sensor_name_to_idx = {name: idx for idx, name in sensor_idx_to_name.items()}
        split_root, data_split = _resolve_av2_split_root(self.data_root, sequence, self.split)
        dataloader = AV2SensorDataLoader(split_root, split_root)
        world_transform = pose_utils.to4x4(dataparser_outputs.dataparser_transform).cpu().float()
        time_offset = float(dataparser_outputs.time_offset)
        dataparser_config = pipeline.datamanager.dataparser.config

        if self.appearance_sensors and len(self.appearance_sensors) != len(self.target_cameras):
            raise ValueError("appearance_sensors must be empty or have the same length as target_cameras.")

        donor_by_camera = {
            target_camera: (
                self.appearance_sensors[idx]
                if self.appearance_sensors
                else _get_default_av2_appearance_sensor(target_camera)
            )
            for idx, target_camera in enumerate(self.target_cameras)
        }

        camera_sets: Dict[str, Dict[str, Any]] = {}
        appearance_params = torch.nn.ParameterDict()
        for target_camera in self.target_cameras:
            donor_sensor = donor_by_camera[target_camera]
            if donor_sensor not in sensor_name_to_idx:
                available = ", ".join(sensor_name_to_idx)
                raise ValueError(f"appearance sensor {donor_sensor!r} not found. Available sensors: {available}")

            image_paths = dataloader.get_ordered_log_cam_fpaths(sequence, target_camera)
            selected_image_paths = _slice_frame_paths(image_paths, self.start_frame, self.max_frames)
            donor_sensor_idx = int(sensor_name_to_idx[donor_sensor])
            cameras, timestamps_ns, _ = _build_av2_target_cameras(
                split_root=split_root,
                dataloader=dataloader,
                sequence=sequence,
                target_camera=target_camera,
                image_paths=selected_image_paths,
                world_transform=world_transform,
                time_offset=time_offset,
                appearance_sensor_idx=donor_sensor_idx,
                rolling_shutter_time=float(dataparser_config.rolling_shutter_time),
                time_to_center_pixel=float(dataparser_config.time_to_center_pixel),
                fallback_reference_camera=donor_sensor,
            )
            if self.downscale_factor != 1.0:
                cameras.rescale_output_resolution(1.0 / self.downscale_factor)

            gt_images = [
                _load_av2_rgb_image(image_path, int(cameras.height[idx].item()), int(cameras.width[idx].item()))
                for idx, image_path in enumerate(selected_image_paths)
            ]
            init_embedding = model.appearance_embedding.weight[donor_sensor_idx].detach().clone().to(model.device)
            appearance_params[target_camera] = torch.nn.Parameter(init_embedding)
            camera_sets[target_camera] = {
                "cameras": cameras,
                "gt_images": gt_images,
                "timestamps_ns": timestamps_ns,
                "image_paths": selected_image_paths,
                "donor_sensor": donor_sensor,
            }

        optimizer = torch.optim.Adam(appearance_params.parameters(), lr=self.learning_rate)
        camera_names = list(camera_sets)

        def evaluate_mean_l1() -> Dict[str, float]:
            metrics = {}
            with torch.no_grad():
                for target_camera, data in camera_sets.items():
                    losses = []
                    for idx, gt_image in enumerate(data["gt_images"]):
                        camera = _set_camera_appearance_override(
                            data["cameras"][idx : idx + 1], appearance_params[target_camera].detach()
                        )
                        outputs = model.get_outputs(camera.to(model.device))
                        losses.append(torch.abs(outputs["rgb"] - gt_image.to(model.device)).mean().item())
                    metrics[target_camera] = float(np.mean(losses))
            return metrics

        baseline_l1 = evaluate_mean_l1()
        CONSOLE.print(f"Fitting stereo appearance for {sequence} on split {data_split}")
        for target_camera, value in baseline_l1.items():
            donor_sensor = camera_sets[target_camera]["donor_sensor"]
            CONSOLE.print(f"Initial mean L1 for {target_camera} (init from {donor_sensor}): {value:.6f}")

        for step in range(1, self.max_steps + 1):
            target_camera = camera_names[torch.randint(len(camera_names), size=(1,)).item()]
            data = camera_sets[target_camera]
            frame_idx = torch.randint(len(data["gt_images"]), size=(1,)).item()
            camera = _set_camera_appearance_override(data["cameras"][frame_idx : frame_idx + 1], appearance_params[target_camera])
            gt_image = data["gt_images"][frame_idx].to(model.device)
            outputs = model.get_outputs(camera.to(model.device))
            loss = model.get_loss_dict(outputs, {"image": gt_image})["main_loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if step % self.log_every == 0 or step == 1 or step == self.max_steps:
                CONSOLE.print(f"Step {step}/{self.max_steps}: {target_camera} frame {frame_idx} L1={loss.item():.6f}")

        fitted_l1 = evaluate_mean_l1()
        entries = {}
        for target_camera in self.target_cameras:
            entries[target_camera] = {
                "embedding": appearance_params[target_camera].detach().cpu(),
                "init_from": camera_sets[target_camera]["donor_sensor"],
                "baseline_mean_l1": baseline_l1[target_camera],
                "fitted_mean_l1": fitted_l1[target_camera],
                "num_frames": len(camera_sets[target_camera]["gt_images"]),
            }

        _save_stereo_appearance_sidecar(
            output_path,
            load_config=self.load_config,
            sequence=sequence,
            appearance_dim=appearance_dim,
            entries=entries,
            fit_settings={
                "data_root": str(self.data_root),
                "data_split": data_split,
                "target_cameras": list(self.target_cameras),
                "start_frame": self.start_frame,
                "max_frames": self.max_frames,
                "downscale_factor": self.downscale_factor,
                "max_steps": self.max_steps,
                "learning_rate": self.learning_rate,
                "seed": self.seed,
            },
        )
        CONSOLE.print(f"[bold][green]:glowing_star: Stereo appearance sidecar saved to {output_path}")
        for target_camera, value in fitted_l1.items():
            CONSOLE.print(f"Fitted mean L1 for {target_camera}: {value:.6f} (baseline {baseline_l1[target_camera]:.6f})")


@dataclass
class FitAV2StereoPhotometric:
    """Fit a grayscale photometric mapping for AV2 stereo cameras on top of frozen renders."""

    load_config: Path
    """Path to the base config YAML file."""

    output_path: Optional[Path] = None
    """Optional sidecar output path. Defaults to <run-dir>/sidecars/stereo_photometric.pt."""

    sequence: str = ""
    """AV2 sequence UUID to fit. Defaults to the scene id inferred from load_config."""

    target_cameras: Tuple[str, ...] = ("stereo_front_left", "stereo_front_right")
    """AV2 camera streams to fit grayscale mappings for."""

    appearance_sensor: str = "ring_front_center"
    """Training camera appearance used to produce the source stereo-view renders."""

    data_root: Path = Path("data/av2")
    """Root directory containing the AV2 dataset."""

    split: Optional[Literal["train", "val", "test", "mini", "mini_val", "mini_test"]] = None
    """Dataset split override. If omitted, infer the split by searching the dataset."""

    start_frame: int = 0
    """First frame index to use for fitting."""

    max_frames: Optional[int] = None
    """Maximum number of frames to use per stereo camera."""

    downscale_factor: float = 1.0
    """Downscale factor applied during fitting."""

    eval_num_rays_per_chunk: Optional[int] = None
    """Optional eval chunk size override."""

    def main(self) -> None:
        scene_id = _infer_scene_id_from_load_config(self.load_config)
        sequence = self.sequence or scene_id
        output_path = self.output_path or _get_default_stereo_photometric_sidecar_path(self.load_config)

        _, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="inference",
            update_config_callback=streamline_ad_config,
            strict_load=False,
        )

        model = pipeline.model
        model.eval()
        dataparser_outputs = pipeline.datamanager.train_dataparser_outputs
        sensor_idx_to_name = dataparser_outputs.metadata["sensor_idx_to_name"]
        sensor_name_to_idx = {name: idx for idx, name in sensor_idx_to_name.items()}
        if self.appearance_sensor not in sensor_name_to_idx:
            available = ", ".join(sensor_name_to_idx)
            raise ValueError(f"appearance_sensor {self.appearance_sensor!r} not found. Available sensors: {available}")

        split_root, data_split = _resolve_av2_split_root(self.data_root, sequence, self.split)
        dataloader = AV2SensorDataLoader(split_root, split_root)
        world_transform = pose_utils.to4x4(dataparser_outputs.dataparser_transform).cpu().float()
        time_offset = float(dataparser_outputs.time_offset)
        dataparser_config = pipeline.datamanager.dataparser.config
        donor_sensor_idx = int(sensor_name_to_idx[self.appearance_sensor])

        entries: Dict[str, Dict[str, Any]] = {}
        CONSOLE.print(f"Fitting stereo photometric mapping for {sequence} on split {data_split}")
        for target_camera in self.target_cameras:
            image_paths = dataloader.get_ordered_log_cam_fpaths(sequence, target_camera)
            selected_image_paths = _slice_frame_paths(image_paths, self.start_frame, self.max_frames)
            cameras, _, _ = _build_av2_target_cameras(
                split_root=split_root,
                dataloader=dataloader,
                sequence=sequence,
                target_camera=target_camera,
                image_paths=selected_image_paths,
                world_transform=world_transform,
                time_offset=time_offset,
                appearance_sensor_idx=donor_sensor_idx,
                rolling_shutter_time=float(dataparser_config.rolling_shutter_time),
                time_to_center_pixel=float(dataparser_config.time_to_center_pixel),
                fallback_reference_camera=self.appearance_sensor,
            )
            if self.downscale_factor != 1.0:
                cameras.rescale_output_resolution(1.0 / self.downscale_factor)

            xtx = torch.zeros((2, 2), dtype=torch.float64)
            xty = torch.zeros((2,), dtype=torch.float64)
            baseline_abs_error = 0.0
            pixel_count = 0

            with torch.no_grad():
                for idx, image_path in enumerate(selected_image_paths):
                    gt_image = _load_av2_rgb_image(image_path, int(cameras.height[idx].item()), int(cameras.width[idx].item()))
                    gt_gray = _extract_grayscale_target(gt_image)
                    outputs = model.get_outputs(cameras[idx : idx + 1].to(model.device))
                    pred_rgb = outputs["rgb"].detach().cpu()
                    pred_gray = _rgb_to_luminance(pred_rgb)

                    x = pred_gray.reshape(-1).double()
                    y = gt_gray.reshape(-1).double()
                    ones = torch.ones_like(x)
                    design = torch.stack((x, ones), dim=-1)
                    xtx += design.T @ design
                    xty += design.T @ y

                    baseline_abs_error += torch.abs(pred_gray - gt_gray).sum().item()
                    pixel_count += gt_gray.numel()

            ridge = torch.tensor([[1e-6, 0.0], [0.0, 1e-6]], dtype=torch.float64)
            solution = torch.linalg.solve(xtx + ridge, xty)
            scale = float(solution[0].item())
            bias = float(solution[1].item())

            fitted_abs_error = 0.0
            with torch.no_grad():
                for idx, image_path in enumerate(selected_image_paths):
                    gt_image = _load_av2_rgb_image(image_path, int(cameras.height[idx].item()), int(cameras.width[idx].item()))
                    gt_gray = _extract_grayscale_target(gt_image)
                    outputs = model.get_outputs(cameras[idx : idx + 1].to(model.device))
                    pred_rgb = outputs["rgb"].detach().cpu()
                    mapped_gray = _extract_grayscale_target(
                        _apply_affine_grayscale_mapping(
                            pred_rgb,
                            torch.tensor(scale, dtype=pred_rgb.dtype),
                            torch.tensor(bias, dtype=pred_rgb.dtype),
                        )
                    )
                    fitted_abs_error += torch.abs(mapped_gray - gt_gray).sum().item()

            baseline_mean_l1 = baseline_abs_error / pixel_count
            fitted_mean_l1 = fitted_abs_error / pixel_count
            entries[target_camera] = {
                "appearance_sensor": self.appearance_sensor,
                "scale": scale,
                "bias": bias,
                "baseline_mean_l1": baseline_mean_l1,
                "fitted_mean_l1": fitted_mean_l1,
                "num_frames": len(selected_image_paths),
                "downscale_factor": self.downscale_factor,
            }
            CONSOLE.print(
                f"{target_camera}: scale={scale:.6f}, bias={bias:.6f}, "
                f"mean L1 {baseline_mean_l1:.6f} -> {fitted_mean_l1:.6f}"
            )

        _save_stereo_photometric_sidecar(
            output_path,
            load_config=self.load_config,
            sequence=sequence,
            entries=entries,
            fit_settings={
                "data_root": str(self.data_root),
                "data_split": data_split,
                "target_cameras": list(self.target_cameras),
                "appearance_sensor": self.appearance_sensor,
                "start_frame": self.start_frame,
                "max_frames": self.max_frames,
                "downscale_factor": self.downscale_factor,
            },
        )
        CONSOLE.print(f"[bold][green]:glowing_star: Stereo photometric sidecar saved to {output_path}")


def plot_lidar_points(points, output_path, cmin=-6.0, cmax=5.0, width=1920, height=1080, ranges=[100, 200, 10]):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Create a 3D scatter plot
    trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker=dict(
            size=1.0,
            color=z,
            colorscale="Viridis",
            opacity=0.8,
            cmin=cmin,
            cmax=cmax,
        ),
    )

    x_range, y_range, z_range = ranges

    # Compute the aspect ratio
    max_range = 2 * max(x_range, y_range, z_range)
    aspect_ratio = dict(x=x_range / max_range, y=y_range / max_range, z=z_range / max_range)

    # Define the camera position
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0.0, y=-0.07, z=0.02),
    )
    layout = go.Layout(
        scene=dict(
            xaxis=dict(
                title="",
                range=[-x_range, x_range],
                showticklabels=False,
                ticks="",
                showline=False,
                showgrid=False,
            ),
            yaxis=dict(
                title="",
                range=[-y_range, y_range],
                showticklabels=False,
                ticks="",
                showline=False,
                showgrid=False,
            ),
            zaxis=dict(
                title="",
                range=[-z_range, z_range],
                showticklabels=False,
                ticks="",
                showline=False,
                showgrid=False,
            ),
            aspectmode="manual",
            aspectratio=aspect_ratio,
            camera=camera,
        )
    )

    fig = go.Figure(data=[trace], layout=layout)
    fig.write_image(output_path, width=width, height=height, scale=1)


def streamline_ad_config(config):
    if getattr(config.pipeline.datamanager, "num_processes", None):
        config.pipeline.datamanager.num_processes = 0
    config.pipeline.model.eval_num_rays_per_chunk = 2**17
    if getattr(config.pipeline.datamanager.dataparser, "add_missing_points", None):
        config.pipeline.datamanager.dataparser.add_missing_points = False
    return config


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[RenderCameraPath, tyro.conf.subcommand(name="camera-path")],
        Annotated[RenderInterpolated, tyro.conf.subcommand(name="interpolate")],
        Annotated[SpiralRender, tyro.conf.subcommand(name="spiral")],
        Annotated[DatasetRender, tyro.conf.subcommand(name="dataset")],
        Annotated[RenderAV2TargetCamera, tyro.conf.subcommand(name="av2-target-camera")],
        Annotated[FitAV2StereoAppearance, tyro.conf.subcommand(name="fit-av2-stereo-appearance")],
        Annotated[FitAV2StereoPhotometric, tyro.conf.subcommand(name="fit-av2-stereo-photometric")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(Commands)  # noqa
