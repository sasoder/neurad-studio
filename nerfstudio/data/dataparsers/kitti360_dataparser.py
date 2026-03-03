# Copyright 2024 the authors of NeuRAD and contributors.
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

"""Data parser for the KITTI-360 dataset."""

import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Type

import numpy as np
import torch
from torch import Tensor
from typing_extensions import Literal

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.lidars import Lidars, LidarType
from nerfstudio.data.dataparsers.ad_dataparser import OPENCV_TO_NERFSTUDIO, ADDataParser, ADDataParserConfig

HORIZONTAL_BEAM_DIVERGENCE = 3.0e-3  # radians
VERTICAL_BEAM_DIVERGENCE = 1.5e-3  # radians
MAX_INTENSITY_VALUE = 1.0
DATA_FREQUENCY = 10.0  # 10 Hz
LIDAR_ROTATION_TIME = 1.0 / DATA_FREQUENCY

# KITTI-360 world frame: x-forward, y-left, z-up
# SplatAD actor frame: x-right, y-forward, z-up
# actor_transform: SplatAD -> KITTI-360
RFZU_TO_FLZU = np.array(
    [[0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float32,
)

# KITTI-360 label categories for dynamic actors
ALLOWED_CATEGORIES = {
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "caravan",
    "trailer",
    "train",
}
SYMMETRIC_CATEGORIES = {
    "car",
    "truck",
    "bus",
    "caravan",
    "trailer",
    "train",
}
DEFORMABLE_CATEGORIES = {
    "person",
    "rider",
}

AVAILABLE_CAMERAS = ("image_00", "image_01", "image_02", "image_03")

# KITTI-360 kittiId -> (cityscapes_name, cityscapes_id)
# Only the ones relevant for 3D bboxes with instances
KITTI_ID_TO_NAME = {
    13: "car",
    14: "truck",
    34: "bus",
    16: "caravan",
    15: "trailer",
    33: "train",
    17: "motorcycle",
    18: "bicycle",
    19: "person",
    20: "rider",
    11: "building",
    7: "wall",
    8: "fence",
}


@dataclass
class Kitti360DataParserConfig(ADDataParserConfig):
    """KITTI-360 DatasetParser config."""

    _target: Type = field(default_factory=lambda: Kitti360)
    """target class to instantiate"""
    sequence: str = "0009"
    """Sequence number (e.g. '0009')."""
    data: Path = Path("data/kitti360")
    """Path to KITTI-360 dataset root."""
    cameras: Tuple[
        Literal["image_00", "image_01", "image_02", "image_03", "none", "all"],
        ...,
    ] = ("image_02", "image_03")
    """Which cameras to use."""
    fisheye_camera_model: str = "mei"
    """Camera model for fisheye cameras: 'mei' (native MEI) or 'pinhole' (pre-remapped)."""
    camera_data_subdir: str = "data_pinhole"
    """Subdir under image_02/03 containing preprocessed pinhole fisheye images (only used when fisheye_camera_model='pinhole')."""
    lidars: Tuple[Literal["velodyne", "none"], ...] = ("velodyne",)
    """Which lidars to use."""
    annotation_interval: float = 0.1
    """Interval between annotations in seconds."""
    allow_per_point_times: bool = True
    """Whether to allow per-point timestamps."""
    compute_sensor_velocities: bool = False
    """Whether to compute sensor velocities."""
    min_lidar_dist: Tuple[float, ...] = (2.0, 1.6, 2.0)
    """Minimum distance of lidar points."""


@dataclass
class Kitti360(ADDataParser):
    """KITTI-360 DatasetParser."""

    config: Kitti360DataParserConfig

    @property
    def actor_transform(self) -> Tensor:
        return torch.from_numpy(RFZU_TO_FLZU)

    def _get_cameras(self) -> Tuple[Cameras, List[Path]]:
        """Returns camera info and image filenames."""
        if "all" in self.config.cameras:
            self.config.cameras = AVAILABLE_CAMERAS

        use_native_mei = self.config.fisheye_camera_model == "mei"

        filenames, times, poses, idxs = [], [], [], []
        fxs, fys, cxs, cys, heights, widths = [], [], [], [], [], []
        cam_types: List[CameraType] = []
        distortion_params_list: List[List[float]] = []

        for camera_idx, cam_name in enumerate(self.config.cameras):
            cam_id = int(cam_name.split("_")[1])
            seq_name = f"2013_05_28_drive_{self.config.sequence}_sync"

            if cam_id in (0, 1):
                camera_folder = self.config.data / "data_2d_raw" / seq_name / cam_name / "data_rect"
                intrinsics = self.P_rect[cam_id]
                width = int(self.perspective_sizes[cam_id][0])
                height = int(self.perspective_sizes[cam_id][1])
                cam_type = CameraType.PERSPECTIVE
                dist_params = [0.0] * 6
                fx_val = float(intrinsics[0, 0])
                fy_val = float(intrinsics[1, 1])
                cx_val = float(intrinsics[0, 2])
                cy_val = float(intrinsics[1, 2])
            elif use_native_mei:
                camera_folder = self.config.data / "data_2d_raw" / seq_name / cam_name / "data_rgb"
                mei_params = _load_mei_calibration(self.config.data / "calibration" / f"image_{cam_id:02d}.yaml")
                width = mei_params["image_width"]
                height = mei_params["image_height"]
                cam_type = CameraType.MEI
                fx_val = mei_params["gamma1"]
                fy_val = mei_params["gamma2"]
                cx_val = mei_params["u0"]
                cy_val = mei_params["v0"]
                dist_params = [
                    mei_params["xi"],
                    mei_params["k1"],
                    mei_params["k2"],
                    mei_params["p1"],
                    mei_params["p2"],
                    0.0,
                ]
            else:
                camera_folder = self.config.data / "data_2d_raw" / seq_name / cam_name / self.config.camera_data_subdir
                intrinsics_path = camera_folder / "intrinsics.json"
                if not intrinsics_path.exists():
                    raise FileNotFoundError(
                        f"Missing fisheye pinhole intrinsics at {intrinsics_path}. "
                        "Run scripts/prepare_kitti360_fisheye.py first."
                    )
                fisheye_intrinsics = _load_pinhole_intrinsics(intrinsics_path)
                intrinsics = fisheye_intrinsics["K"]
                width = int(fisheye_intrinsics["width"])
                height = int(fisheye_intrinsics["height"])
                cam_type = CameraType.PERSPECTIVE
                dist_params = [0.0] * 6
                fx_val = float(intrinsics[0, 0])
                fy_val = float(intrinsics[1, 1])
                cx_val = float(intrinsics[0, 2])
                cy_val = float(intrinsics[1, 2])

            if not camera_folder.exists():
                raise FileNotFoundError(f"Camera folder not found: {camera_folder}")

            camera_files = sorted(camera_folder.glob("*.png"))

            for camera_file in camera_files:
                frame_id = int(camera_file.stem)

                if frame_id not in self.frame_to_pose:
                    continue

                filenames.append(camera_file)
                times.append(self.timestamps_cam.get(cam_name, {}).get(frame_id, frame_id / DATA_FREQUENCY))

                ego_pose = torch.from_numpy(self.frame_to_pose[frame_id].copy()).double()

                cam2imu = torch.from_numpy(self.cam_to_pose[f"image_{cam_id:02d}"].copy()).double()

                if cam_id in (0, 1):
                    rect_inv = torch.eye(4, dtype=torch.double)
                    rect_inv[:3, :3] = torch.from_numpy(
                        np.linalg.inv(self.R_rect[cam_id]).copy()
                    ).double()
                    cam_pose = ego_pose @ cam2imu @ rect_inv
                else:
                    cam_pose = ego_pose @ cam2imu

                cam_pose[:3, :3] = cam_pose[:3, :3] @ torch.from_numpy(OPENCV_TO_NERFSTUDIO).double()
                poses.append(cam_pose.float()[:3, :4])
                idxs.append(camera_idx)
                fxs.append(fx_val)
                fys.append(fy_val)
                cxs.append(cx_val)
                cys.append(cy_val)
                widths.append(width)
                heights.append(height)
                cam_types.append(cam_type)
                distortion_params_list.append(dist_params)

        poses = torch.stack(poses)
        times = torch.tensor(times, dtype=torch.float64)
        idxs = torch.tensor(idxs).int().unsqueeze(-1)

        cameras = Cameras(
            fx=torch.tensor(fxs, dtype=torch.float32),
            fy=torch.tensor(fys, dtype=torch.float32),
            cx=torch.tensor(cxs, dtype=torch.float32),
            cy=torch.tensor(cys, dtype=torch.float32),
            height=torch.tensor(heights, dtype=torch.int32),
            width=torch.tensor(widths, dtype=torch.int32),
            distortion_params=torch.tensor(distortion_params_list, dtype=torch.float32),
            camera_to_worlds=poses[:, :3, :4],
            camera_type=cam_types,
            times=times,
            metadata={"sensor_idxs": idxs},
        )
        return cameras, filenames

    def _get_lidars(self) -> Tuple[Lidars, List[Path]]:
        """Returns lidar info and loaded point clouds."""
        times, poses, idxs, lidar_filenames = [], [], [], []

        seq_name = f"2013_05_28_drive_{self.config.sequence}_sync"

        # cam0_to_velo gives cam0 -> velo, so velo2cam0 = inv(cam0_to_velo)
        # We want velo2world = imu2world @ velo2imu
        # velo2imu = cam2imu @ cam2velo_inv ... but simpler:
        # velo2imu can be derived from: cam0_to_velo (cam0->velo) and cam_to_pose (cam0->imu)
        # velo2cam0 = inv(cam0_to_velo)
        # cam0_to_imu = cam_to_pose['image_00']
        # velo2imu = cam0_to_imu @ velo2cam0
        cam0_to_velo = torch.from_numpy(self.cam_to_velo.copy()).float()
        velo2cam0 = cam0_to_velo.inverse()
        cam0_to_imu = torch.from_numpy(self.cam_to_pose["image_00"].copy()).float()
        velo2imu = cam0_to_imu @ velo2cam0

        for lidar_idx, lidar_name in enumerate(self.config.lidars):
            lidar_folder = self.config.data / "data_3d_raw" / seq_name / "velodyne_points" / "data"

            if not lidar_folder.exists():
                raise FileNotFoundError(f"Lidar folder not found: {lidar_folder}")

            lidar_files = sorted(lidar_folder.glob("*.bin"))

            for lidar_file in lidar_files:
                frame_id = int(lidar_file.stem)

                if frame_id not in self.frame_to_pose:
                    continue

                if "velodyne" in self.timestamps_lidar and frame_id in self.timestamps_lidar["velodyne"]:
                    times.append(self.timestamps_lidar["velodyne"][frame_id])
                else:
                    times.append(frame_id / DATA_FREQUENCY)

                ego_pose = torch.from_numpy(self.frame_to_pose[frame_id].copy()).float()
                pose = ego_pose @ velo2imu
                poses.append(pose)
                idxs.append(lidar_idx)
                lidar_filenames.append(lidar_file)

        poses = torch.stack(poses)
        times = torch.tensor(times, dtype=torch.float64)
        idxs = torch.tensor(idxs).int().unsqueeze(-1)

        lidars = Lidars(
            lidar_to_worlds=poses[:, :3, :4],
            lidar_type=LidarType.VELODYNE64E,
            times=times,
            assume_ego_compensated=False,
            metadata={"sensor_idxs": idxs},
            horizontal_beam_divergence=HORIZONTAL_BEAM_DIVERGENCE,
            vertical_beam_divergence=VERTICAL_BEAM_DIVERGENCE,
        )

        return lidars, lidar_filenames

    def _read_lidars(self, lidars: Lidars, filepaths: List[Path]) -> List[torch.Tensor]:
        point_clouds = []
        for filepath in filepaths:
            pc = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)
            xyz = pc[:, :3]
            intensity = pc[:, 3] / MAX_INTENSITY_VALUE
            t = _get_mock_timestamps(xyz)
            pc = np.hstack((xyz, intensity[:, None], t[:, None]))
            point_clouds.append(torch.from_numpy(pc).float())

        lidars.lidar_to_worlds = lidars.lidar_to_worlds.float()
        return point_clouds

    def _get_actor_trajectories(self) -> List[Dict]:
        """Parse 3D bounding boxes from KITTI-360 XML annotations."""
        seq_name = f"2013_05_28_drive_{self.config.sequence}_sync"

        if self.config.include_deformable_actors:
            allowed_cats = ALLOWED_CATEGORIES.union(DEFORMABLE_CATEGORIES)
        else:
            allowed_cats = ALLOWED_CATEGORIES

        bbox_path = self.config.data / "data_3d_bboxes" / "train" / f"{seq_name}.xml"
        if not bbox_path.exists():
            bbox_path = self.config.data / "data_3d_bboxes" / "train_full" / f"{seq_name}.xml"
        if not bbox_path.exists():
            return []

        tree = ET.parse(bbox_path)
        root = tree.getroot()

        # Group objects by globalId
        objects: Dict[int, List[dict]] = defaultdict(list)

        for child in root:
            if child.find("transform") is None:
                continue

            kitti_id = int(child.find("semanticId").text)
            if kitti_id not in KITTI_ID_TO_NAME:
                continue

            label = KITTI_ID_TO_NAME[kitti_id]
            if label not in allowed_cats:
                continue

            instance_id = int(child.find("instanceId").text)
            timestamp = int(child.find("timestamp").text)

            # Skip static objects (timestamp == -1)
            if timestamp == -1:
                continue

            transform = _parse_opencv_matrix(child.find("transform"))
            R = transform[:3, :3]
            T = transform[:3, 3]

            vertices_local = _parse_opencv_matrix(child.find("vertices"))

            global_id = kitti_id * 1000 + instance_id

            objects[global_id].append(
                {
                    "label": label,
                    "timestamp": timestamp,
                    "R": R,
                    "T": T,
                    "vertices_local": vertices_local,
                }
            )

        trajs = []
        for global_id, entries in objects.items():
            if len(entries) < 2:
                continue

            entries_sorted = sorted(entries, key=lambda x: x["timestamp"])
            label = entries_sorted[0]["label"]
            deformable = label in DEFORMABLE_CATEGORIES
            symmetric = label in SYMMETRIC_CATEGORIES

            poses_list, timestamps, wlh_list = [], [], []

            for entry in entries_sorted:
                frame_id = entry["timestamp"]
                if frame_id not in self.frame_to_pose:
                    continue

                # Get timestamp in seconds
                if "velodyne" in self.timestamps_lidar and frame_id in self.timestamps_lidar["velodyne"]:
                    t = self.timestamps_lidar["velodyne"][frame_id]
                else:
                    t = frame_id / DATA_FREQUENCY
                timestamps.append(t)

                # Build the object pose in world coordinates
                obj_pose = np.eye(4, dtype=np.float64)
                obj_pose[:3, :3] = entry["R"]
                obj_pose[:3, 3] = entry["T"]

                # The object is already in world coordinates (KITTI-360 world: x-fwd, y-left, z-up)
                # Apply actor_transform so local frame is (x-right, y-forward, z-up)
                obj_pose = obj_pose @ RFZU_TO_FLZU.astype(np.float64)
                poses_list.append(obj_pose)

                # Compute dimensions from local vertices
                verts = entry["vertices_local"]
                dims = verts.max(axis=0) - verts.min(axis=0)
                wlh_list.append(dims)

            if len(timestamps) < 2:
                continue

            poses_np = np.array(poses_list)
            dynamic = np.any(np.std(poses_np[:, :3, 3], axis=0) > 0.5)

            if not dynamic:
                continue

            # Median dimensions (w, l, h) in the local frame
            wlh_median = np.median(wlh_list, axis=0)
            # In KITTI-360 local frame (before actor_transform), the axes might be ordered differently.
            # The local vertices define a box; we interpret:
            # dim[0] = along local x (forward in KITTI-360 = length)
            # dim[1] = along local y (left in KITTI-360 = width)
            # dim[2] = along local z (up in KITTI-360 = height)
            # SplatAD expects WLH order: width, length, height
            wlh = np.array([wlh_median[1], wlh_median[0], wlh_median[2]], dtype=np.float32)

            trajs.append(
                {
                    "poses": torch.tensor(poses_np, dtype=torch.float32),
                    "timestamps": torch.tensor(timestamps, dtype=torch.float64),
                    "dims": torch.tensor(wlh, dtype=torch.float32),
                    "label": label,
                    "stationary": not dynamic,
                    "symmetric": symmetric,
                    "deformable": deformable,
                }
            )

        return trajs

    def _generate_dataparser_outputs(self, split="train"):
        self._load_calibrations()
        self._load_poses()
        self._load_timestamps()

        out = super()._generate_dataparser_outputs(split=split)

        # Clean up
        del self.frame_to_pose
        del self.cam_to_pose
        del self.cam_to_velo
        del self.R_rect
        del self.P_rect

        return out

    def _load_calibrations(self):
        """Load KITTI-360 calibration files."""
        calib_dir = self.config.data / "calibration"

        # Camera-to-pose (cam -> IMU/GPS)
        self.cam_to_pose = _load_cam_to_pose(calib_dir / "calib_cam_to_pose.txt")

        # Camera-to-velodyne (cam0 -> velo)
        self.cam_to_velo = _load_rigid_transform(calib_dir / "calib_cam_to_velo.txt")

        # Perspective intrinsics and rectification
        self.P_rect, self.R_rect, self.perspective_sizes = _load_perspective_intrinsics(
            calib_dir / "perspective.txt"
        )

    def _load_poses(self):
        """Load optimized poses from poses.txt."""
        seq_name = f"2013_05_28_drive_{self.config.sequence}_sync"
        pose_file = self.config.data / "data_poses" / seq_name / "poses.txt"

        if not pose_file.exists():
            raise FileNotFoundError(f"Pose file not found: {pose_file}")

        data = np.loadtxt(pose_file)
        frames = data[:, 0].astype(int)
        matrices = data[:, 1:].reshape(-1, 3, 4)

        self.frame_to_pose = {}
        for frame, mat in zip(frames, matrices):
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :4] = mat
            self.frame_to_pose[int(frame)] = pose

    def _load_timestamps(self):
        """Load sensor timestamps."""
        seq_name = f"2013_05_28_drive_{self.config.sequence}_sync"

        self.timestamps_cam: Dict[str, Dict[int, float]] = {}
        self.timestamps_lidar: Dict[str, Dict[int, float]] = {}

        for cam_name in AVAILABLE_CAMERAS:
            ts_file = self.config.data / "data_2d_raw" / seq_name / cam_name / "timestamps.txt"
            if ts_file.exists():
                self.timestamps_cam[cam_name] = _parse_timestamps(ts_file)

        velo_ts_file = self.config.data / "data_3d_raw" / seq_name / "velodyne_points" / "timestamps.txt"
        if velo_ts_file.exists():
            self.timestamps_lidar["velodyne"] = _parse_timestamps(velo_ts_file)


def _load_cam_to_pose(filepath: Path) -> Dict[str, np.ndarray]:
    """Load camera-to-pose (camera -> IMU/GPS) transforms."""
    cameras = ["image_00", "image_01", "image_02", "image_03"]
    result = {}
    lastrow = np.array([0, 0, 0, 1], dtype=np.float64).reshape(1, 4)

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for cam in cameras:
                if line.startswith(cam + ":"):
                    values = [float(x) for x in line.split(":")[1].strip().split()]
                    mat = np.array(values, dtype=np.float64).reshape(3, 4)
                    result[cam] = np.concatenate((mat, lastrow), axis=0)
                    break

    return result


def _load_rigid_transform(filepath: Path) -> np.ndarray:
    """Load a rigid 3x4 transform and return as 4x4."""
    lastrow = np.array([0, 0, 0, 1], dtype=np.float64).reshape(1, 4)
    mat = np.loadtxt(filepath).reshape(3, 4)
    return np.concatenate((mat, lastrow), axis=0)


def _load_perspective_intrinsics(
    filepath: Path,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, Tuple[int, int]]]:
    """Load perspective camera intrinsics and rectification matrices."""
    P_rect = {}
    R_rect = {}
    sizes: Dict[int, Tuple[int, int]] = {}

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            key = parts[0].rstrip(":")

            if key.startswith("P_rect_"):
                cam_id = int(key[-2:])
                values = [float(x) for x in parts[1:]]
                P_rect[cam_id] = np.array(values, dtype=np.float32).reshape(3, 4)

            elif key.startswith("R_rect_"):
                cam_id = int(key[-2:])
                values = [float(x) for x in parts[1:]]
                R_rect[cam_id] = np.array(values, dtype=np.float64).reshape(3, 3)

            elif key.startswith("S_rect_"):
                cam_id = int(key[-2:])
                width = int(float(parts[1]))
                height = int(float(parts[2]))
                sizes[cam_id] = (width, height)

    assert 0 in sizes and 1 in sizes, "Failed to read perspective image sizes from perspective.txt"
    return P_rect, R_rect, sizes


def _parse_timestamps(filepath: Path) -> Dict[int, float]:
    """Parse timestamps file and return frame_id -> timestamp_seconds mapping."""
    timestamps = {}
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                # KITTI-360 format: "2013-05-28 09:45:58.073869926"
                # Remove nanosecond precision (Python datetime handles up to microseconds)
                line_trimmed = line[:26]
                dt = datetime.strptime(line_trimmed, "%Y-%m-%d %H:%M:%S.%f")
                timestamps[i] = dt.timestamp()
            except ValueError:
                continue

    return timestamps


def _load_pinhole_intrinsics(filepath: Path) -> Dict[str, np.ndarray]:
    with open(filepath, "r") as f:
        data = json.load(f)
    return {
        "K": np.array(data["K"], dtype=np.float32),
        "width": int(data["width"]),
        "height": int(data["height"]),
    }


def _load_mei_calibration(filepath: Path) -> Dict[str, float]:
    """Load MEI omnidirectional camera calibration from KITTI-360 OpenCV YAML file."""
    params: Dict[str, float] = {}
    section = ""
    kv_re = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:\s*(.*)\s*$")

    with open(filepath, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("%YAML") or line.startswith("---"):
                continue
            m = kv_re.match(raw)
            if not m:
                continue
            key, value = m.group(1), m.group(2).strip()
            if value == "":
                section = key
            elif key in ("image_width", "image_height"):
                params[key] = int(float(value))
            elif section == "mirror_parameters" and key == "xi":
                params["xi"] = float(value)
            elif section == "distortion_parameters" and key in ("k1", "k2", "p1", "p2"):
                params[key] = float(value)
            elif section == "projection_parameters" and key in ("gamma1", "gamma2", "u0", "v0"):
                params[key] = float(value)

    required = {"image_width", "image_height", "xi", "k1", "k2", "p1", "p2", "gamma1", "gamma2", "u0", "v0"}
    missing = required - set(params.keys())
    if missing:
        raise ValueError(f"Missing MEI calibration parameters in {filepath}: {missing}")
    return params


def _parse_opencv_matrix(node) -> np.ndarray:
    """Parse an OpenCV matrix from XML."""
    rows = int(node.find("rows").text)
    cols = int(node.find("cols").text)
    data_text = node.find("data").text.split()
    values = [float(d) for d in data_text if d.strip()]
    return np.array(values, dtype=np.float64).reshape(rows, cols)


def _get_mock_timestamps(points: np.ndarray) -> np.ndarray:
    """Get mock relative timestamps for velodyne points based on azimuth angle."""
    angles = np.arctan2(points[:, 1], points[:, 0])
    angles += np.pi
    fraction = angles / (2 * np.pi)
    return (fraction * LIDAR_ROTATION_TIME).astype(np.float32)
