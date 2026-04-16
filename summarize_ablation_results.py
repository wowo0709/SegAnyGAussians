#!/usr/bin/env python3
import csv
import json
from argparse import ArgumentParser
from pathlib import Path


def _scale_prefix(scale):
    return f"scale_{scale:.2f}".replace('-', 'm').replace('.', 'p')


def _load_result(path):
    with open(path, 'r') as handle:
        return json.load(handle)


def _flatten_result(data, path):
    row = {
        'label': data.get('label'),
        'model_path': data.get('model_path'),
        'cluster_method': data.get('cluster_method'),
        'camera_split_used': data.get('camera_split_used'),
        'feature_iteration': data.get('feature_iteration'),
        'num_views': data.get('num_views'),
        'mean_num_clusters': data.get('summary', {}).get('mean_num_clusters'),
        'mean_cluster_confidence': data.get('summary', {}).get('mean_cluster_confidence'),
        'mean_connectivity_weighted': data.get('summary', {}).get('mean_connectivity_weighted'),
        'mean_mask_best_iou': data.get('summary', {}).get('mean_mask_best_iou'),
        'mean_mask_best_precision': data.get('summary', {}).get('mean_mask_best_precision'),
        'mean_mask_best_recall': data.get('summary', {}).get('mean_mask_best_recall'),
        'mean_pixel_confidence': data.get('summary', {}).get('mean_pixel_confidence'),
        'total_selected_masks': data.get('summary', {}).get('total_selected_masks'),
        'total_evaluated_views': data.get('summary', {}).get('total_evaluated_views'),
        'result_json': str(path),
    }
    for scale_metric in data.get('scales', []):
        prefix = _scale_prefix(float(scale_metric['scale']))
        for key in (
            'num_clusters',
            'cluster_confidence_mean',
            'cluster_size_min',
            'cluster_size_median',
            'cluster_size_max',
            'connectivity_mean',
            'connectivity_weighted_mean',
            'selected_masks',
            'mask_best_iou_mean',
            'mask_best_precision_mean',
            'mask_best_recall_mean',
            'pixel_confidence_mean',
            'evaluated_views',
        ):
            row[f'{prefix}_{key}'] = scale_metric.get(key)
    return row


def _write_csv(path, rows):
    fieldnames = sorted({key for row in rows for key in row.keys()})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown(path, rows):
    summary_columns = [
        'label',
        'cluster_method',
        'mean_num_clusters',
        'mean_connectivity_weighted',
        'mean_mask_best_iou',
        'mean_mask_best_precision',
        'mean_mask_best_recall',
        'total_selected_masks',
    ]
    lines = []
    lines.append('| ' + ' | '.join(summary_columns) + ' |')
    lines.append('| ' + ' | '.join(['---'] * len(summary_columns)) + ' |')
    for row in rows:
        values = []
        for key in summary_columns:
            value = row.get(key, '')
            if isinstance(value, float):
                values.append(f'{value:.4f}')
            else:
                values.append(str(value))
        lines.append('| ' + ' | '.join(values) + ' |')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    parser = ArgumentParser(description='Summarize multiple SAGA ablation evaluation JSON files into CSV/Markdown tables.')
    parser.add_argument('json_files', nargs='+', help='Evaluation JSON files produced by eval_ablation.py')
    parser.add_argument('--output_csv', type=str, default=None)
    parser.add_argument('--output_md', type=str, default=None)
    parser.add_argument('--sort_key', type=str, default='mean_mask_best_iou')
    parser.add_argument('--descending', action='store_true')
    args = parser.parse_args()

    rows = []
    for json_path in args.json_files:
        path = Path(json_path)
        rows.append(_flatten_result(_load_result(path), path))

    sort_key = args.sort_key
    rows.sort(key=lambda row: (row.get(sort_key) is None, row.get(sort_key, 0.0)), reverse=args.descending)

    if args.output_csv:
        _write_csv(args.output_csv, rows)
    if args.output_md:
        _write_markdown(args.output_md, rows)

    for row in rows:
        print(
            f"{row.get('label')} [{row.get('cluster_method')}] "
            f"IoU={row.get('mean_mask_best_iou', 0.0):.4f} "
            f"Conn={row.get('mean_connectivity_weighted', 0.0):.4f} "
            f"Clusters={row.get('mean_num_clusters', 0.0):.2f}"
        )


if __name__ == '__main__':
    main()
