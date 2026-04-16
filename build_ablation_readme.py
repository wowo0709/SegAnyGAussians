#!/usr/bin/env python3
import csv
import json
from argparse import ArgumentParser
from pathlib import Path


def _load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def _load_jsonl(path):
    records = []
    if not Path(path).exists():
        return records
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _load_csv(path):
    rows = []
    if not Path(path).exists():
        return rows
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows.extend(reader)
    return rows


def _detect_scale_prefixes(rows):
    prefixes = set()
    for row in rows:
        for key in row.keys():
            if key.startswith("scale_") and key.endswith("_mask_best_iou_mean"):
                prefixes.add(key[: -len("_mask_best_iou_mean")])
    return sorted(prefixes)


def _float_str(value):
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _rel(target, root):
    return str(Path(target).resolve().relative_to(Path(root).resolve()))


def _markdown_table(rows, columns):
    if not rows:
        return "_No results available._\n"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_float_str(row.get(column, "")) for column in columns)
            + " |"
        )
    return "\n".join(lines) + "\n"


def _stage_scene_summary_rows(stage_dir):
    scene_rows = {}
    if not Path(stage_dir).exists():
        return scene_rows
    for summary_path in sorted(Path(stage_dir).glob("*/ablation_summary.csv")):
        scene_name = summary_path.parent.name
        rows = _load_csv(summary_path)
        if rows:
            scene_rows[scene_name] = rows
    return scene_rows


def _mean(values):
    numeric = []
    for value in values:
        if value in (None, ""):
            continue
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            continue
    if not numeric:
        return 0.0
    return float(sum(numeric) / len(numeric))


def _sort_rows(rows, primary="mean_connectivity_weighted", secondary="mean_mask_best_iou"):
    return sorted(
        rows,
        key=lambda row: (
            float(row.get(primary, 0.0) or 0.0),
            float(row.get(secondary, 0.0) or 0.0),
            float(row.get("mean_cluster_confidence", 0.0) or 0.0),
        ),
        reverse=True,
    )


def _per_scale_columns(prefixes):
    columns = ["label", "cluster_method", "mean_connectivity_weighted", "mean_mask_best_iou"]
    for prefix in prefixes:
        columns.extend(
            [
                f"{prefix}_connectivity_weighted_mean",
                f"{prefix}_mask_best_iou_mean",
            ]
        )
    return columns


def _aggregate_scene_rows(scene_rows):
    grouped = {}
    for scene_name, rows in scene_rows.items():
        for row in rows:
            key = (row.get("label", ""), row.get("cluster_method", ""))
            bucket = grouped.setdefault(
                key,
                {
                    "label": row.get("label", ""),
                    "cluster_method": row.get("cluster_method", ""),
                    "_rows": [],
                    "_scenes": set(),
                },
            )
            bucket["_rows"].append(row)
            bucket["_scenes"].add(scene_name)

    aggregate_rows = []
    for (_, _), bucket in grouped.items():
        rows = bucket.pop("_rows")
        scenes = sorted(bucket.pop("_scenes"))
        aggregate = {
            "label": bucket["label"],
            "cluster_method": bucket["cluster_method"],
            "num_scenes": len(scenes),
            "scenes": ",".join(scenes),
        }
        numeric_keys = sorted(
            {
                key
                for row in rows
                for key in row.keys()
                if key not in {"label", "cluster_method", "model_path", "result_json", "camera_split_used"}
            }
        )
        for key in numeric_keys:
            aggregate[key] = _mean(row.get(key) for row in rows)
        aggregate_rows.append(aggregate)
    return _sort_rows(aggregate_rows)


def _winner_for_keys(rows, primary_key, secondary_key):
    if not rows:
        return None
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row.get(primary_key, 0.0) or 0.0),
            float(row.get(secondary_key, 0.0) or 0.0),
            float(row.get("mean_cluster_confidence", 0.0) or 0.0),
        ),
        reverse=True,
    )
    return ordered[0]


def _scale_value_from_prefix(prefix):
    if not prefix.startswith("scale_"):
        return prefix
    return prefix[len("scale_"):].replace("p", ".")


def _oom_rows(status_records):
    rows = []
    for record in status_records:
        fallback_level = int(record.get("oom_fallback_level", 0) or 0)
        detected_oom = bool(record.get("detected_oom", False))
        if fallback_level <= 0 and not detected_oom:
            continue
        actual = record.get("actual_config", {})
        rows.append(
            {
                "job_id": record.get("job_id"),
                "stage": record.get("stage"),
                "scene": record.get("scene"),
                "status": record.get("status"),
                "oom_fallback_level": fallback_level,
                "num_sampled_rays": actual.get("NUM_SAMPLED_RAYS"),
                "graph_laplacian_samples": actual.get("GRAPH_LAPLACIAN_SAMPLES"),
                "cluster_sample_size": actual.get("CLUSTER_SAMPLE_SIZE"),
                "render_downsample": actual.get("EVAL_RENDER_DOWNSAMPLE"),
                "save_viz_views": actual.get("SAVE_VIZ_VIEWS"),
                "log_path": record.get("log_path"),
            }
        )
    return rows


def _winner_line(winner_payload):
    winner = winner_payload.get("winner") if winner_payload else None
    if not winner:
        return "No winner selected."
    return (
        f"`{winner['label']}` "
        f"(Conn={float(winner.get('mean_connectivity_weighted', 0.0)):.4f}, "
        f"IoU={float(winner.get('mean_mask_best_iou', 0.0)):.4f}, "
        f"Scenes={int(winner.get('num_scenes', 0))})"
    )


def main():
    parser = ArgumentParser(description="Build a human-readable README for an ablation suite.")
    parser.add_argument("--ablation_root", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    root = Path(args.ablation_root)
    output_path = Path(args.output) if args.output else root / "README.md"

    suite_config_path = root / "suite_config.json"
    suite_config = _load_json(suite_config_path) if suite_config_path.exists() else {}

    status_dir = root / "status"
    done_records = _load_jsonl(status_dir / "done.jsonl")
    failed_records = _load_jsonl(status_dir / "failed.jsonl")
    running_records = _load_jsonl(status_dir / "running.jsonl")
    pending_records = _load_jsonl(status_dir / "pending.jsonl")

    lines = []
    title = suite_config.get("title", "SAGA Ablation Report")
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(
        "This README is the generated experiment report for the current ablation round. "
        "It summarizes the scene set, the active defaults, stage-wise winners, OOM fallback events, and the raw artifacts."
    )
    lines.append("")

    lines.append("## Dataset and Setup")
    lines.append("")
    scenes = suite_config.get("scenes", [])
    if scenes:
        lines.append("Scenes:")
        for scene in scenes:
            lines.append(f"- `{scene['name']}`: `{scene['source_path']}`")
        lines.append("")
    defaults = suite_config.get("defaults", {})
    if defaults:
        lines.append("Common defaults:")
        for key in (
            "sam_downsample",
            "num_sampled_rays",
            "viz_interval",
            "graph_laplacian_samples",
            "graph_laplacian_scales",
            "eval_render_downsample",
            "cluster_sample_size",
            "save_viz_views",
            "mask_scale_tolerance",
            "mask_min_area",
            "mask_max_area_ratio",
            "mask_max_iou_overlap",
            "mask_boundary_erode_kernel",
            "min_valid_masks_per_view",
        ):
            if key in defaults:
                lines.append(f"- `{key}`: `{defaults[key]}`")
        lines.append("")

    lines.append("## Experiment Matrix")
    lines.append("")
    for stage in suite_config.get("stages", []):
        enabled = stage.get("enabled", True)
        suffix = "" if enabled else " (disabled)"
        lines.append(f"### {stage.get('display_name', stage.get('name'))}{suffix}")
        if stage.get("description"):
            lines.append(stage["description"])
        variants = stage.get("variants", [])
        if variants:
            lines.append("")
            lines.append("Variants:")
            for variant in variants:
                lines.append(f"- `{variant}`")
        lines.append("")

    lines.append("## Results")
    lines.append("")
    for stage in suite_config.get("stages", []):
        stage_name = stage["name"]
        stage_dir = root / "stages" / stage_name
        aggregate_csv = stage_dir / "aggregate.csv"
        aggregate_md = stage_dir / "aggregate.md"
        winner_json = stage_dir / "winner.json"
        aggregate_rows = _load_csv(aggregate_csv)
        winner_payload = _load_json(winner_json) if winner_json.exists() else None
        scene_summary_rows = _stage_scene_summary_rows(stage_dir)

        lines.append(f"### {stage.get('display_name', stage_name)}")
        lines.append("")
        lines.append(f"Winner: {_winner_line(winner_payload)}")
        lines.append("")
        if aggregate_rows:
            lines.append("Stage aggregate:")
            lines.append("")
            lines.append(
                _markdown_table(
                    aggregate_rows,
                    [
                        "label",
                        "cluster_method",
                        "num_scenes",
                        "mean_connectivity_weighted",
                        "mean_mask_best_iou",
                        "mean_num_clusters",
                        "mean_cluster_confidence",
                        "total_selected_masks",
                    ],
                ).rstrip()
            )
            lines.append("")
        else:
            lines.append("_Stage aggregate is not available yet. Showing scene-level summaries instead._")
            lines.append("")

        if scene_summary_rows:
            stage_aggregate_rows = _aggregate_scene_rows(scene_summary_rows)
            scale_prefixes = _detect_scale_prefixes(stage_aggregate_rows)
            if scale_prefixes:
                lines.append("Per-scale stage aggregate (scene mean):")
                lines.append("")
                lines.append(
                    _markdown_table(
                        stage_aggregate_rows,
                        _per_scale_columns(scale_prefixes),
                    ).rstrip()
                )
                lines.append("")
                lines.append("Per-scale winners:")
                lines.append("")
                for prefix in scale_prefixes:
                    winner = _winner_for_keys(
                        stage_aggregate_rows,
                        f"{prefix}_connectivity_weighted_mean",
                        f"{prefix}_mask_best_iou_mean",
                    )
                    if winner is None:
                        continue
                    lines.append(
                        f"- `scale={_scale_value_from_prefix(prefix)}`: "
                        f"`{winner['label']}` "
                        f"(Conn={float(winner.get(f'{prefix}_connectivity_weighted_mean', 0.0)):.4f}, "
                        f"IoU={float(winner.get(f'{prefix}_mask_best_iou_mean', 0.0)):.4f}, "
                        f"Scenes={int(winner.get('num_scenes', 0) or 0)})"
                    )
                lines.append("")

        if scene_summary_rows:
            lines.append("Per-scene, per-scale comparison:")
            lines.append("")
            for scene_name, rows in sorted(scene_summary_rows.items()):
                sorted_rows = _sort_rows(rows)
                scale_prefixes = _detect_scale_prefixes(sorted_rows)
                lines.append(f"#### {scene_name}")
                lines.append("")
                lines.append(
                    _markdown_table(
                        sorted_rows,
                        _per_scale_columns(scale_prefixes),
                    ).rstrip()
                )
                lines.append("")
        if aggregate_csv.exists() or aggregate_md.exists():
            links = []
            if aggregate_csv.exists():
                links.append(f"[CSV]({_rel(aggregate_csv, root)})")
            if aggregate_md.exists():
                links.append(f"[Markdown]({_rel(aggregate_md, root)})")
            lines.append("Artifacts: " + ", ".join(links))
            lines.append("")
        elif scene_summary_rows:
            artifact_links = []
            for scene_name in sorted(scene_summary_rows.keys()):
                summary_csv = stage_dir / scene_name / "ablation_summary.csv"
                summary_md = stage_dir / scene_name / "ablation_summary.md"
                if summary_csv.exists():
                    artifact_links.append(f"[{scene_name} CSV]({_rel(summary_csv, root)})")
                if summary_md.exists():
                    artifact_links.append(f"[{scene_name} Markdown]({_rel(summary_md, root)})")
            if artifact_links:
                lines.append("Artifacts: " + ", ".join(artifact_links))
                lines.append("")

    oom_rows = _oom_rows(done_records + failed_records)
    lines.append("## OOM and Fallback Log")
    lines.append("")
    if oom_rows:
        lines.append(
            _markdown_table(
                oom_rows,
                [
                    "job_id",
                    "stage",
                    "scene",
                    "status",
                    "oom_fallback_level",
                    "num_sampled_rays",
                    "graph_laplacian_samples",
                    "cluster_sample_size",
                    "render_downsample",
                    "save_viz_views",
                ],
            ).rstrip()
        )
        lines.append("")
    else:
        lines.append("No OOM fallbacks were recorded.")
        lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    manifest_path = root / "jobs_manifest.csv"
    if manifest_path.exists():
        lines.append(f"- [jobs_manifest.csv]({_rel(manifest_path, root)})")
    for status_name in ("pending.jsonl", "running.jsonl", "done.jsonl", "failed.jsonl"):
        path = status_dir / status_name
        if path.exists():
            lines.append(f"- [status/{status_name}]({_rel(path, root)})")
    for stage in suite_config.get("stages", []):
        stage_dir = root / "stages" / stage["name"]
        if stage_dir.exists():
            lines.append(f"- [stage dir: {stage['name']}]({_rel(stage_dir, root)})")
    lines.append("")

    viz_paths = sorted(root.glob("stages/*/*/ablation_eval/viz/*.png"))[:8]
    if viz_paths:
        lines.append("Representative visualizations:")
        for path in viz_paths:
            lines.append(f"- [{path.name}]({_rel(path, root)})")
        lines.append("")

    lines.append("## Next Step")
    lines.append("")
    lines.append(
        suite_config.get(
            "next_step",
            "After this round, the next step is to add mesh-aware training-time losses and repeat the study with MILo and model-family comparisons.",
        )
    )
    lines.append("")

    lines.append("## Status Snapshot")
    lines.append("")
    lines.append(f"- Pending records: `{len(pending_records)}`")
    lines.append(f"- Running records: `{len(running_records)}`")
    lines.append(f"- Completed records: `{len(done_records)}`")
    lines.append(f"- Failed records: `{len(failed_records)}`")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
