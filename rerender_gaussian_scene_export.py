#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


REFERENCE_ROTATION_AXIS_FLIP = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
DEFAULT_CONDA_EXE = Path("/opt/conda/bin/conda")
DEFAULT_CONDA_ENV = "saga"
DEFAULT_SPOT_CHECK_VIEWS = ("000", "075", "149")


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Build and run a SegAnyGaussians render job for an exported scene Gaussian "
            "using cameras.json and a reference transforms.json for validation."
        )
    )
    parser.add_argument("--ply-path", type=Path, required=True, help="Path to gaussian_scene_full.ply.")
    parser.add_argument("--camera-json", type=Path, required=True, help="Path to cameras.json.")
    parser.add_argument(
        "--reference-transforms",
        type=Path,
        required=True,
        help="Path to the reference transforms.json used only for view-order validation.",
    )
    parser.add_argument(
        "--job-json",
        type=Path,
        default=None,
        help="Where to write the render job JSON. Defaults to <ply_dir>/render_job_150views.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write the rendered outputs. Defaults to <ply_dir>/renders_150views.",
    )
    parser.add_argument("--expected-views", type=int, default=150, help="Expected number of views.")
    parser.add_argument("--width", type=int, default=512, help="Output image width.")
    parser.add_argument("--height", type=int, default=512, help="Output image height.")
    parser.add_argument("--cx", type=float, default=None, help="Principal point x. Defaults to width / 2.")
    parser.add_argument("--cy", type=float, default=None, help="Principal point y. Defaults to height / 2.")
    parser.add_argument(
        "--background",
        choices=("white", "black"),
        default="white",
        help="Background color for the render job.",
    )
    parser.add_argument(
        "--render-script",
        type=Path,
        default=root_dir / "render_gaussian_export.py",
        help="Path to render_gaussian_export.py.",
    )
    parser.add_argument(
        "--conda-exe",
        type=Path,
        default=DEFAULT_CONDA_EXE,
        help="Path to the conda executable used to launch the render job.",
    )
    parser.add_argument(
        "--conda-env",
        default=DEFAULT_CONDA_ENV,
        help="Conda environment used to launch the render job.",
    )
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Only validate inputs and write the render job JSON without launching the render.",
    )
    parser.add_argument(
        "--spot-check-views",
        nargs="*",
        default=list(DEFAULT_SPOT_CHECK_VIEWS),
        help="View ids to verify against the existing render directory after rendering.",
    )
    return parser.parse_args()


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_json(path: str | Path):
    return json.loads(Path(path).read_text())


def dump_json(path: str | Path, payload) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(payload, indent=2))
    return target


def resolve_view_id(camera_entry: dict, fallback_index: int) -> str:
    if camera_entry.get("img_name"):
        return str(camera_entry["img_name"])
    if "id" in camera_entry:
        return f"{int(camera_entry['id']):03d}"
    return f"{fallback_index:03d}"


def validate_camera_json(camera_entries: list[dict], expected_views: int, width: int, height: int) -> list[str]:
    if len(camera_entries) != expected_views:
        raise ValueError(
            f"Expected {expected_views} camera entries in cameras.json, but found {len(camera_entries)}."
        )

    view_ids: list[str] = []
    for index, entry in enumerate(camera_entries):
        view_id = resolve_view_id(entry, index)
        expected_view_id = f"{index:03d}"
        if view_id != expected_view_id:
            raise ValueError(
                f"Camera entry {index} had view id '{view_id}', expected '{expected_view_id}'."
            )
        if int(entry["width"]) != width or int(entry["height"]) != height:
            raise ValueError(
                f"Camera entry {view_id} had size {(entry['width'], entry['height'])}, "
                f"expected {(width, height)}."
            )
        view_ids.append(view_id)
    return view_ids


def validate_reference_transforms(
    camera_entries: list[dict],
    reference_frames: list[dict],
    expected_views: int,
    position_tolerance: float = 1e-5,
    rotation_tolerance: float = 1e-5,
) -> None:
    if len(reference_frames) != expected_views:
        raise ValueError(
            f"Expected {expected_views} frames in reference transforms, but found {len(reference_frames)}."
        )

    for index, (camera_entry, frame) in enumerate(zip(camera_entries, reference_frames)):
        view_id = resolve_view_id(camera_entry, index)
        reference_view_id = Path(str(frame["file_path"])).stem
        if reference_view_id != view_id:
            raise ValueError(
                f"Reference frame order mismatch at index {index}: cameras.json has '{view_id}', "
                f"transforms.json has '{reference_view_id}'."
            )

        reference_transform = np.asarray(frame["transform_matrix"], dtype=np.float32)
        reference_position = reference_transform[:3, 3]
        camera_position = np.asarray(camera_entry["position"], dtype=np.float32)
        position_error = float(np.max(np.abs(reference_position - camera_position)))
        if position_error > position_tolerance:
            raise ValueError(
                f"Position mismatch for view {view_id}: max abs error {position_error:.6g} "
                f"exceeded tolerance {position_tolerance:.6g}."
            )

        reference_rotation = reference_transform[:3, :3] @ REFERENCE_ROTATION_AXIS_FLIP
        camera_rotation = np.asarray(camera_entry["rotation"], dtype=np.float32)
        rotation_error = float(np.max(np.abs(reference_rotation - camera_rotation)))
        if rotation_error > rotation_tolerance:
            raise ValueError(
                f"Rotation mismatch for view {view_id}: max abs error {rotation_error:.6g} "
                f"exceeded tolerance {rotation_tolerance:.6g}."
            )


def build_views(
    camera_entries: list[dict],
    output_dir: str | Path,
    width: int,
    height: int,
    cx: float,
    cy: float,
) -> list[dict]:
    output_root = Path(output_dir)
    views: list[dict] = []
    for index, camera_entry in enumerate(camera_entries):
        view_id = resolve_view_id(camera_entry, index)
        views.append(
            {
                "view_id": view_id,
                "width": width,
                "height": height,
                "fx": float(camera_entry["fx"]),
                "fy": float(camera_entry["fy"]),
                "cx": float(cx),
                "cy": float(cy),
                "camera_position": camera_entry["position"],
                "camera_rotation": camera_entry["rotation"],
                "output_rgb": str(output_root / f"{view_id}.png"),
                "output_mask": str(output_root / f"{view_id}_mask.png"),
                "output_depth_npy": str(output_root / f"{view_id}_depth.npy"),
                "output_depth_png": str(output_root / f"{view_id}_depth.png"),
            }
        )
    return views


def validate_render_outputs(views: list[dict]) -> None:
    for view in views:
        rgb_path = Path(view["output_rgb"])
        mask_path = Path(view["output_mask"])
        depth_npy_path = Path(view["output_depth_npy"])
        depth_png_path = Path(view["output_depth_png"])
        for path in (rgb_path, mask_path, depth_npy_path, depth_png_path):
            if not path.exists():
                raise FileNotFoundError(f"Missing render output for view {view['view_id']}: {path}")
        depth = np.load(depth_npy_path)
        if not np.any(depth > 0):
            raise RuntimeError(f"Rendered depth was empty for view {view['view_id']}: {depth_npy_path}")


def verify_spot_check_views(reference_render_dir: str | Path, views: list[dict], spot_check_views: list[str]) -> None:
    view_lookup = {view["view_id"]: view for view in views}
    reference_root = Path(reference_render_dir)
    for view_id in spot_check_views:
        if view_id not in view_lookup:
            raise KeyError(f"Spot-check view '{view_id}' was not present in the render job.")
        reference_image = reference_root / f"{view_id}.png"
        if not reference_image.exists():
            raise FileNotFoundError(f"Reference render for spot check was missing: {reference_image}")
        rendered_image = Path(view_lookup[view_id]["output_rgb"])
        with Image.open(reference_image) as ref_image, Image.open(rendered_image) as out_image:
            if ref_image.size != out_image.size:
                raise ValueError(
                    f"Spot-check size mismatch for view {view_id}: "
                    f"reference {ref_image.size} vs rendered {out_image.size}."
                )


def launch_render_job(job_json: str | Path, conda_exe: str | Path, conda_env: str, render_script: str | Path) -> None:
    command = [
        str(conda_exe),
        "run",
        "--no-capture-output",
        "-n",
        conda_env,
        "python",
        str(render_script),
        "--job-json",
        str(job_json),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = "\n".join(part for part in (completed.stdout.strip(), completed.stderr.strip()) if part)
        raise RuntimeError(f"Render job failed for {job_json}.\n{detail}")


def main() -> None:
    args = parse_args()

    ply_path = args.ply_path.resolve()
    camera_json = args.camera_json.resolve()
    reference_transforms = args.reference_transforms.resolve()
    render_script = args.render_script.resolve()
    conda_exe = args.conda_exe.resolve()

    if not ply_path.exists():
        raise FileNotFoundError(f"Gaussian export was not found: {ply_path}")
    if not camera_json.exists():
        raise FileNotFoundError(f"Camera JSON was not found: {camera_json}")
    if not reference_transforms.exists():
        raise FileNotFoundError(f"Reference transforms were not found: {reference_transforms}")
    if not render_script.exists():
        raise FileNotFoundError(f"Render script was not found: {render_script}")
    if not conda_exe.exists():
        raise FileNotFoundError(f"Conda executable was not found: {conda_exe}")

    job_json = (args.job_json or (ply_path.parent / "render_job_150views.json")).resolve()
    output_dir = (args.output_dir or (ply_path.parent / "renders_150views")).resolve()
    cx = float(args.width / 2.0 if args.cx is None else args.cx)
    cy = float(args.height / 2.0 if args.cy is None else args.cy)
    background_rgb = [1.0, 1.0, 1.0] if args.background == "white" else [0.0, 0.0, 0.0]

    camera_entries = load_json(camera_json)
    if not isinstance(camera_entries, list):
        raise TypeError(f"Expected {camera_json} to contain a JSON list of camera entries.")
    reference_payload = load_json(reference_transforms)
    reference_frames = reference_payload.get("frames")
    if not isinstance(reference_frames, list):
        raise TypeError(f"Expected {reference_transforms} to contain a 'frames' list.")

    validate_camera_json(camera_entries, args.expected_views, args.width, args.height)
    validate_reference_transforms(camera_entries, reference_frames, args.expected_views)

    views = build_views(camera_entries, output_dir, args.width, args.height, cx, cy)
    payload = {
        "ply_path": str(ply_path),
        "background_rgb": background_rgb,
        "views": views,
    }
    dump_json(job_json, payload)

    print(f"Validated {len(views)} camera views against {reference_transforms}")
    print(f"Wrote render job JSON to {job_json}")

    if args.skip_render:
        return

    ensure_dir(output_dir)
    print(f"Rendering {len(views)} views to {output_dir}")
    launch_render_job(job_json, conda_exe, args.conda_env, render_script)
    validate_render_outputs(views)
    verify_spot_check_views(reference_transforms.parent, views, args.spot_check_views)
    print(f"Validated render outputs in {output_dir}")
    print(f"Spot-checked views: {', '.join(args.spot_check_views)}")


if __name__ == "__main__":
    main()
