#!/usr/bin/env python3
"""Generate average AV2 camera extrinsics and pairwise deltas as CSV."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import pyarrow.feather as feather


CAMERA_NAMES = [
    "ring_front_center",
    "ring_front_left",
    "ring_front_right",
    "ring_rear_left",
    "ring_rear_right",
    "ring_side_left",
    "ring_side_right",
    "stereo_front_left",
    "stereo_front_right",
]

DEFAULT_DATASET_ROOT = Path("/home/samuelsoderberg/neurad-studio/data/av2/sensor/train")
DEFAULT_OUTPUT_CSV = Path("/home/samuelsoderberg/neurad-studio/av2_train_camera_deltas.csv")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Average AV2 camera extrinsics over all train logs and export "
            "ego-to-camera plus ordered camera-to-camera deltas as CSV."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Path to the AV2 sensor train split.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Destination CSV path.",
    )
    return parser.parse_args()


def distance_xyz(x_m: float, y_m: float, z_m: float) -> float:
    """Return Euclidean norm for a translation vector."""
    return math.sqrt(x_m * x_m + y_m * y_m + z_m * z_m)


def read_sensor_rows(log_dir: Path) -> dict[str, dict[str, float | str]]:
    """Read the per-sensor calibration rows for one AV2 log."""
    table = feather.read_table(log_dir / "calibration" / "egovehicle_SE3_sensor.feather")
    columns = {name: table[name].to_pylist() for name in table.column_names}
    rows = []
    for idx in range(len(columns["sensor_name"])):
        row = {name: columns[name][idx] for name in table.column_names}
        rows.append(row)
    return {str(row["sensor_name"]): row for row in rows}


def collect_camera_positions(
    dataset_root: Path,
) -> tuple[dict[str, list[tuple[float, float, float]]], dict[str, set[str]], list[str], int]:
    """Collect all available camera positions and logs missing stereo extrinsics."""
    positions: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    camera_logs: dict[str, set[str]] = defaultdict(set)
    missing_stereo_logs: list[str] = []
    logs_scanned = 0

    for log_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        extrinsics_path = log_dir / "calibration" / "egovehicle_SE3_sensor.feather"
        if not extrinsics_path.exists():
            continue

        logs_scanned += 1
        rows_by_sensor = read_sensor_rows(log_dir)
        has_stereo_left = "stereo_front_left" in rows_by_sensor
        has_stereo_right = "stereo_front_right" in rows_by_sensor
        if not (has_stereo_left and has_stereo_right):
            missing_stereo_logs.append(log_dir.name)

        for camera_name in CAMERA_NAMES:
            row = rows_by_sensor.get(camera_name)
            if row is None:
                continue
            positions[camera_name].append((float(row["tx_m"]), float(row["ty_m"]), float(row["tz_m"])))
            camera_logs[camera_name].add(log_dir.name)

    return positions, camera_logs, missing_stereo_logs, logs_scanned


def mean_position(samples: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    """Compute the mean position over all samples."""
    count = len(samples)
    if count == 0:
        raise ValueError("Cannot average an empty sample list.")
    return tuple(sum(sample[axis] for sample in samples) / count for axis in range(3))


def build_csv_rows(
    positions: dict[str, list[tuple[float, float, float]]],
    camera_logs: dict[str, set[str]],
) -> list[dict[str, str | int | float]]:
    """Build output rows for ego-to-camera and camera-to-camera deltas."""
    means = {camera_name: mean_position(samples) for camera_name, samples in positions.items() if samples}

    rows: list[dict[str, str | int | float]] = []

    for camera_name in CAMERA_NAMES:
        if camera_name not in means:
            continue
        tx_m, ty_m, tz_m = means[camera_name]
        rows.append(
            {
                "row_type": "ego_to_camera",
                "from_sensor": "ego",
                "to_sensor": camera_name,
                "from_count": 0,
                "to_count": len(positions[camera_name]),
                "shared_logs": len(positions[camera_name]),
                "dx_m": tx_m,
                "dy_m": ty_m,
                "dz_m": tz_m,
                "distance_m": distance_xyz(tx_m, ty_m, tz_m),
            }
        )

    for from_camera in CAMERA_NAMES:
        if from_camera not in means:
            continue
        from_x, from_y, from_z = means[from_camera]
        for to_camera in CAMERA_NAMES:
            if to_camera not in means:
                continue
            to_x, to_y, to_z = means[to_camera]
            dx_m = to_x - from_x
            dy_m = to_y - from_y
            dz_m = to_z - from_z
            rows.append(
                {
                    "row_type": "camera_to_camera",
                    "from_sensor": from_camera,
                    "to_sensor": to_camera,
                    "from_count": len(positions[from_camera]),
                    "to_count": len(positions[to_camera]),
                    "shared_logs": len(camera_logs[from_camera] & camera_logs[to_camera]),
                    "dx_m": dx_m,
                    "dy_m": dy_m,
                    "dz_m": dz_m,
                    "distance_m": distance_xyz(dx_m, dy_m, dz_m),
                }
            )

    return rows


def write_csv(output_csv: Path, rows: list[dict[str, str | int | float]]) -> None:
    """Write the CSV output."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_type",
        "from_sensor",
        "to_sensor",
        "from_count",
        "to_count",
        "shared_logs",
        "dx_m",
        "dy_m",
        "dz_m",
        "distance_m",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    """Generate the CSV and print a short summary."""
    args = parse_args()
    positions, camera_logs, missing_stereo_logs, logs_scanned = collect_camera_positions(args.dataset_root)
    rows = build_csv_rows(positions, camera_logs)
    write_csv(args.output_csv, rows)

    print(f"Scanned {logs_scanned} logs.")
    print(f"Wrote {len(rows)} rows to {args.output_csv}")
    if missing_stereo_logs:
        print("Logs missing stereo extrinsic rows:")
        for log_id in missing_stereo_logs:
            print(f"  - {log_id}")


if __name__ == "__main__":
    main()
