#!/usr/bin/env python3
import csv
import json
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path


SUMMARY_KEYS = [
    "mean_num_clusters",
    "mean_cluster_confidence",
    "mean_connectivity_weighted",
    "mean_mask_best_iou",
    "mean_mask_best_precision",
    "mean_mask_best_recall",
    "mean_pixel_confidence",
    "total_selected_masks",
    "total_evaluated_views",
]


def _mean(values):
    numeric = [float(v) for v in values if v is not None]
    if not numeric:
        return 0.0
    return float(sum(numeric) / len(numeric))


def _load_result(path):
    with open(path, "r") as handle:
        return json.load(handle)


def _write_csv(path, rows):
    fieldnames = sorted({key for row in rows for key in row.keys()})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown(path, rows):
    columns = [
        "label",
        "cluster_method",
        "num_scenes",
        "mean_connectivity_weighted",
        "mean_mask_best_iou",
        "mean_num_clusters",
        "mean_cluster_confidence",
        "total_selected_masks",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for key in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")


def _aggregate_rows(data_by_label):
    rows = []
    for label, records in sorted(data_by_label.items()):
        cluster_methods = sorted({record.get("cluster_method", "") for record in records})
        source_scenes = sorted({Path(record.get("source_path", "")).name for record in records})
        row = {
            "label": label,
            "cluster_method": ",".join(cluster_methods),
            "num_scenes": len(records),
            "scenes": ",".join(source_scenes),
            "result_jsons": ",".join(record["_result_json"] for record in records),
        }
        summary = [record.get("summary", {}) for record in records]
        for key in SUMMARY_KEYS:
            row[key] = _mean(item.get(key) for item in summary)
        rows.append(row)
    return rows


def _pick_winner(rows, primary_key, secondary_key):
    if not rows:
        return None
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row.get(primary_key, 0.0)),
            float(row.get(secondary_key, 0.0)),
            float(row.get("mean_cluster_confidence", 0.0)),
        ),
        reverse=True,
    )
    return ordered[0]


def main():
    parser = ArgumentParser(description="Aggregate per-scene ablation JSON results into a stage-level summary.")
    parser.add_argument("json_files", nargs="+", help="Evaluation JSON files produced by eval_ablation.py")
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--output_md", type=str, required=True)
    parser.add_argument("--winner_json", type=str, required=True)
    parser.add_argument("--primary_key", type=str, default="mean_connectivity_weighted")
    parser.add_argument("--secondary_key", type=str, default="mean_mask_best_iou")
    args = parser.parse_args()

    grouped = defaultdict(list)
    for json_path in args.json_files:
        data = _load_result(json_path)
        data["_result_json"] = str(Path(json_path))
        grouped[str(data.get("label"))].append(data)

    rows = _aggregate_rows(grouped)
    rows.sort(
        key=lambda row: (
            float(row.get(args.primary_key, 0.0)),
            float(row.get(args.secondary_key, 0.0)),
            float(row.get("mean_cluster_confidence", 0.0)),
        ),
        reverse=True,
    )

    _write_csv(args.output_csv, rows)
    _write_markdown(args.output_md, rows)

    winner = _pick_winner(rows, args.primary_key, args.secondary_key)
    winner_payload = {
        "primary_key": args.primary_key,
        "secondary_key": args.secondary_key,
        "winner": winner,
    }
    Path(args.winner_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.winner_json).write_text(json.dumps(winner_payload, indent=2) + "\n")

    for row in rows:
        print(
            f"{row['label']} [{row['cluster_method']}] "
            f"Conn={row.get('mean_connectivity_weighted', 0.0):.4f} "
            f"IoU={row.get('mean_mask_best_iou', 0.0):.4f} "
            f"Scenes={row.get('num_scenes', 0)}"
        )


if __name__ == "__main__":
    main()
