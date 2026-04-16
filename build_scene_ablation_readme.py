#!/usr/bin/env python3
import json
from argparse import ArgumentParser
from pathlib import Path


def _load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def _rel(target, root):
    return str(Path(target).resolve().relative_to(Path(root).resolve()))


def _float_str(value):
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _markdown_table(rows, columns):
    if not rows:
        return "_No results available._\n"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_float_str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def _winner(rows, primary_key="connectivity_weighted_mean", secondary_key="mask_best_iou_mean"):
    if not rows:
        return None
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row.get(primary_key, 0.0) or 0.0),
            float(row.get(secondary_key, 0.0) or 0.0),
            float(row.get("cluster_confidence_mean", 0.0) or 0.0),
        ),
        reverse=True,
    )
    return ordered[0]


def _variant_order(label):
    order = {
        "single_scale_band_035": 0,
        "single_scale_band_050": 1,
        "single_scale_band_060": 2,
        "multiscale": 3,
    }
    return order.get(label, 99)


def _target_scale_for_label(label):
    mapping = {
        "single_scale_band_035": 0.35,
        "single_scale_band_050": 0.50,
        "single_scale_band_060": 0.60,
    }
    return mapping.get(label)


def _flat_scale_rows(results):
    rows = []
    for result in results:
        label = result.get("label")
        target_scale = _target_scale_for_label(label)
        if target_scale is None:
            for scale_row in result.get("scales", []):
                scale = float(scale_row.get("scale", 0.0))
                rows.append(
                    {
                        "row_label": f"{label} (scale={scale:.2f})",
                        "label": label,
                        "scale": scale,
                        "cluster_method": result.get("cluster_method"),
                        "connectivity_weighted_mean": scale_row.get("connectivity_weighted_mean", 0.0),
                        "mask_best_iou_mean": scale_row.get("mask_best_iou_mean", 0.0),
                        "mask_best_precision_mean": scale_row.get("mask_best_precision_mean", 0.0),
                        "mask_best_recall_mean": scale_row.get("mask_best_recall_mean", 0.0),
                        "num_clusters": scale_row.get("num_clusters", 0.0),
                        "cluster_confidence_mean": scale_row.get("cluster_confidence_mean", 0.0),
                        "selected_masks": scale_row.get("selected_masks", 0.0),
                        "evaluated_views": scale_row.get("evaluated_views", 0.0),
                    }
                )
            continue

        scale_row = next(
            (row for row in result.get("scales", []) if abs(float(row.get("scale", 0.0)) - target_scale) < 1e-6),
            None,
        )
        if scale_row is None:
            continue
        rows.append(
            {
                "row_label": f"{label} (scale={target_scale:.2f})",
                "label": label,
                "scale": target_scale,
                "cluster_method": result.get("cluster_method"),
                "connectivity_weighted_mean": scale_row.get("connectivity_weighted_mean", 0.0),
                "mask_best_iou_mean": scale_row.get("mask_best_iou_mean", 0.0),
                "mask_best_precision_mean": scale_row.get("mask_best_precision_mean", 0.0),
                "mask_best_recall_mean": scale_row.get("mask_best_recall_mean", 0.0),
                "num_clusters": scale_row.get("num_clusters", 0.0),
                "cluster_confidence_mean": scale_row.get("cluster_confidence_mean", 0.0),
                "selected_masks": scale_row.get("selected_masks", 0.0),
                "evaluated_views": scale_row.get("evaluated_views", 0.0),
            }
        )
    rows.sort(
        key=lambda row: (
            _variant_order(row.get("label")),
            float(row.get("scale", 0.0)),
        ),
        reverse=False,
    )
    return rows


def _collect_scene_results(scene_dir):
    results = []
    for json_path in sorted(scene_dir.glob("*/ablation_eval/*.json")):
        payload = _load_json(json_path)
        payload["_result_json"] = str(json_path)
        payload["_result_csv"] = str(json_path.with_suffix(".csv"))
        results.append(payload)
    return results


def _collect_scales(results):
    scales = []
    for result in results:
        for row in result.get("scales", []):
            scale = float(row.get("scale", 0.0))
            if scale not in scales:
                scales.append(scale)
    return sorted(scales)


def build_readme(scene_dir: Path, output_path: Path):
    results = _collect_scene_results(scene_dir)
    if not results:
        raise RuntimeError(f"No ablation JSON files found under {scene_dir}")

    first = results[0]
    scene_name = scene_dir.name
    run_context_path = scene_dir / "run_context.json"
    run_context = _load_json(run_context_path) if run_context_path.exists() else {}
    scales = _collect_scales(results)
    flat_rows = _flat_scale_rows(results)

    lines = []
    lines.append(f"# Stage A Scene Report: {scene_name}")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(
        "This README summarizes the completed Stage A supervision ablation for this scene, "
        "with per-scale comparisons for the multiscale and single-scale-band variants."
    )
    lines.append("")

    lines.append("## Setup")
    lines.append("")
    lines.append(f"- `source_path`: `{first.get('source_path')}`")
    lines.append(f"- `camera_split_used`: `{first.get('camera_split_used')}`")
    lines.append(f"- `render_downsample`: `{first.get('render_downsample')}`")
    lines.append(f"- `eval_scales`: `{', '.join(f'{scale:.2f}' for scale in scales)}`")
    if run_context:
        for key in (
            "num_sampled_rays",
            "mask_scale_tolerance",
            "mask_min_area",
            "mask_max_area_ratio",
            "mask_max_iou_overlap",
            "mask_boundary_erode_kernel",
            "min_valid_masks_per_view",
        ):
            if key in run_context:
                lines.append(f"- `{key}`: `{run_context[key]}`")
    lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append(
        _markdown_table(
            flat_rows,
            [
                "row_label",
                "cluster_method",
                "connectivity_weighted_mean",
                "mask_best_iou_mean",
                "mask_best_precision_mean",
                "mask_best_recall_mean",
                "num_clusters",
                "cluster_confidence_mean",
                "selected_masks",
                "evaluated_views",
            ],
        ).rstrip()
    )
    lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    summary_csv = scene_dir / "ablation_summary.csv"
    summary_md = scene_dir / "ablation_summary.md"
    if summary_csv.exists():
        lines.append(f"- [ablation_summary.csv]({_rel(summary_csv, scene_dir)})")
    if summary_md.exists():
        lines.append(f"- [ablation_summary.md]({_rel(summary_md, scene_dir)})")
    if run_context_path.exists():
        lines.append(f"- [run_context.json]({_rel(run_context_path, scene_dir)})")

    for result in results:
        variant_dir = Path(result.get("model_path", "")).resolve()
        variant_name = result.get("label")
        result_json = Path(result.get("_result_json", ""))
        result_csv = Path(result.get("_result_csv", ""))
        lines.append(f"- `{variant_name}`:")
        if result_json.exists():
            lines.append(f"  [JSON]({_rel(result_json, scene_dir)})")
        if result_csv.exists():
            lines.append(f"  [CSV]({_rel(result_csv, scene_dir)})")
        viz_dir = variant_dir / "ablation_eval" / "viz"
        for scale in scales:
            overlay = viz_dir / f"scale_{scale:.2f}_view_000_overlay.png"
            if overlay.exists():
                lines.append(f"  [scale {scale:.2f} overlay]({_rel(overlay, scene_dir)})")
    lines.append("")

    output_path.write_text("\n".join(lines))


def main():
    parser = ArgumentParser(description="Build a per-scene README for an ablation scene directory.")
    parser.add_argument("--scene_dir", type=str, default=None)
    parser.add_argument("--stage_dir", type=str, default=None)
    args = parser.parse_args()

    if bool(args.scene_dir) == bool(args.stage_dir):
        raise SystemExit("Provide exactly one of --scene_dir or --stage_dir")

    if args.scene_dir:
        scene_dir = Path(args.scene_dir)
        build_readme(scene_dir, scene_dir / "README.md")
        return

    stage_dir = Path(args.stage_dir)
    for scene_dir in sorted(path for path in stage_dir.iterdir() if path.is_dir()):
        if scene_dir.name.startswith("."):
            continue
        build_readme(scene_dir, scene_dir / "README.md")


if __name__ == "__main__":
    main()
