#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 SOURCE_PATH BASE_SCENE_MODEL ABLATION_ROOT" >&2
  echo "Example: $0 /root/node1/data2/diyscene/mipnerf360/360_v2/garden /root/node1/data2/diyscene/3dgs/fitted_gs/mipnerf360/360_v2/garden /root/node1/data2/diyscene/ablations/garden" >&2
  exit 1
fi

SOURCE_PATH="$1"
BASE_MODEL="$2"
ABLATION_ROOT="$3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
TRAIN_ITERATIONS="${TRAIN_ITERATIONS:-10000}"
FEATURE_ITERATION_OVERRIDE="${FEATURE_ITERATION_OVERRIDE:-$TRAIN_ITERATIONS}"
NUM_SAMPLED_RAYS="${NUM_SAMPLED_RAYS:-768}"
VIZ_INTERVAL="${VIZ_INTERVAL:-0}"
VIZ_VIEW_IDX="${VIZ_VIEW_IDX:-0}"
VIZ_MODE="${VIZ_MODE:-embedding}"
VIZ_MAX_SAMPLES="${VIZ_MAX_SAMPLES:-5000}"
EVAL_SCALES="${EVAL_SCALES:-0.35 0.50 0.60}"
EVAL_CLUSTER_METHODS="${EVAL_CLUSTER_METHODS:-NormalizedCut}"
EVAL_MAX_VIEWS="${EVAL_MAX_VIEWS:-16}"
EVAL_VIEW_STRIDE="${EVAL_VIEW_STRIDE:-1}"
EVAL_RENDER_DOWNSAMPLE="${EVAL_RENDER_DOWNSAMPLE:-8}"
SUPERVISION_MODE="${SUPERVISION_MODE:-multiscale}"
MASK_SCALE_TARGET="${MASK_SCALE_TARGET:-0.5}"
MASK_SCALE_TOLERANCE="${MASK_SCALE_TOLERANCE:-0.10}"
MASK_MIN_AREA="${MASK_MIN_AREA:-64}"
MASK_MAX_AREA_RATIO="${MASK_MAX_AREA_RATIO:-0.60}"
MASK_MAX_IOU_OVERLAP="${MASK_MAX_IOU_OVERLAP:-0.90}"
MASK_BOUNDARY_ERODE_KERNEL="${MASK_BOUNDARY_ERODE_KERNEL:-3}"
MIN_VALID_MASKS_PER_VIEW="${MIN_VALID_MASKS_PER_VIEW:-2}"
SAVE_VIZ_VIEWS="${SAVE_VIZ_VIEWS:-1}"
GRAPH_LAPLACIAN_WEIGHT="${GRAPH_LAPLACIAN_WEIGHT:-0.05}"
GRAPH_LAPLACIAN_SAMPLES="${GRAPH_LAPLACIAN_SAMPLES:-4096}"
GRAPH_LAPLACIAN_SCALES="${GRAPH_LAPLACIAN_SCALES:-2}"
GRAPH_LAPLACIAN_K="${GRAPH_LAPLACIAN_K:-16}"
GRAPH_SPATIAL_WEIGHT="${GRAPH_SPATIAL_WEIGHT:-1.0}"
GRAPH_SH0_COLOR_WEIGHT="${GRAPH_SH0_COLOR_WEIGHT:-0.0}"
GRAPH_SH0_COLOR_SIGMA="${GRAPH_SH0_COLOR_SIGMA:-0.25}"
CLUSTER_SAMPLE_SIZE="${CLUSTER_SAMPLE_SIZE:-12000}"
CLUSTER_GRAPH_K="${CLUSTER_GRAPH_K:-16}"
CLUSTER_MAX_CLUSTERS="${CLUSTER_MAX_CLUSTERS:-24}"
CLUSTER_MIN_CLUSTER_SIZE="${CLUSTER_MIN_CLUSTER_SIZE:-128}"
CLUSTER_CUT_THRESHOLD="${CLUSTER_CUT_THRESHOLD:-0.12}"
CLUSTER_SPATIAL_WEIGHT="${CLUSTER_SPATIAL_WEIGHT:-1.0}"
CLUSTER_SH0_COLOR_WEIGHT="${CLUSTER_SH0_COLOR_WEIGHT:-0.0}"
CLUSTER_SH0_COLOR_SIGMA="${CLUSTER_SH0_COLOR_SIGMA:-0.25}"
CLUSTER_MESH_WEIGHT="${CLUSTER_MESH_WEIGHT:-0.0}"
CLUSTER_MESH_PATH="${CLUSTER_MESH_PATH:-}"
HDBSCAN_MIN_CLUSTER_SIZE="${HDBSCAN_MIN_CLUSTER_SIZE:-10}"
HDBSCAN_EPSILON="${HDBSCAN_EPSILON:-0.01}"
CONNECTIVITY_K="${CONNECTIVITY_K:-12}"
BOUNDARY_NEGATIVE_WEIGHT="${BOUNDARY_NEGATIVE_WEIGHT:-0.10}"
BOUNDARY_NEGATIVE_MARGIN="${BOUNDARY_NEGATIVE_MARGIN:-0.15}"
BOUNDARY_BAND_KERNEL="${BOUNDARY_BAND_KERNEL:-5}"
SUMMARY_SORT_KEY="${SUMMARY_SORT_KEY:-mean_connectivity_weighted}"
VARIANT_LIST="${VARIANT_LIST:-baseline,laplacian,laplacian_boundary}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
RUN_CONTEXT_JSON="${RUN_CONTEXT_JSON:-$ABLATION_ROOT/run_context.json}"

export TRAIN_ITERATIONS FEATURE_ITERATION_OVERRIDE NUM_SAMPLED_RAYS VIZ_INTERVAL VIZ_VIEW_IDX VIZ_MODE VIZ_MAX_SAMPLES
export EVAL_SCALES EVAL_CLUSTER_METHODS EVAL_MAX_VIEWS EVAL_VIEW_STRIDE EVAL_RENDER_DOWNSAMPLE
export SUPERVISION_MODE MASK_SCALE_TARGET MASK_SCALE_TOLERANCE MASK_MIN_AREA MASK_MAX_AREA_RATIO MASK_MAX_IOU_OVERLAP
export MASK_BOUNDARY_ERODE_KERNEL MIN_VALID_MASKS_PER_VIEW SAVE_VIZ_VIEWS
export GRAPH_LAPLACIAN_WEIGHT GRAPH_LAPLACIAN_SAMPLES GRAPH_LAPLACIAN_SCALES GRAPH_LAPLACIAN_K GRAPH_SPATIAL_WEIGHT
export GRAPH_SH0_COLOR_WEIGHT GRAPH_SH0_COLOR_SIGMA
export CLUSTER_SAMPLE_SIZE CLUSTER_GRAPH_K CLUSTER_MAX_CLUSTERS CLUSTER_MIN_CLUSTER_SIZE CLUSTER_CUT_THRESHOLD
export CLUSTER_SPATIAL_WEIGHT CLUSTER_SH0_COLOR_WEIGHT CLUSTER_SH0_COLOR_SIGMA CLUSTER_MESH_WEIGHT CLUSTER_MESH_PATH
export HDBSCAN_MIN_CLUSTER_SIZE HDBSCAN_EPSILON CONNECTIVITY_K
export BOUNDARY_NEGATIVE_WEIGHT BOUNDARY_NEGATIVE_MARGIN BOUNDARY_BAND_KERNEL
export SUMMARY_SORT_KEY VARIANT_LIST SKIP_TRAIN SKIP_EVAL

mkdir -p "$ABLATION_ROOT"

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -f "$src" ]]; then
    cp -f "$src" "$dst"
  fi
}

prepare_variant_root() {
  local variant_name="$1"
  local variant_dir="$ABLATION_ROOT/$variant_name"
  mkdir -p "$variant_dir/point_cloud"

  copy_if_exists "$BASE_MODEL/cfg_args" "$variant_dir/cfg_args"
  copy_if_exists "$BASE_MODEL/cameras.json" "$variant_dir/cameras.json"
  copy_if_exists "$BASE_MODEL/feature_cfg_args" "$variant_dir/feature_cfg_args"

  if [[ "$SKIP_TRAIN" == "1" ]]; then
    copy_if_exists "$BASE_MODEL/feature_model_path.txt" "$variant_dir/feature_model_path.txt"
    if [[ -d "$BASE_MODEL/feature_model" || -L "$BASE_MODEL/feature_model" ]]; then
      if [[ -e "$variant_dir/feature_model" && ! -L "$variant_dir/feature_model" ]]; then
        rm -rf "$variant_dir/feature_model"
      fi
      ln -sfn "$BASE_MODEL/feature_model" "$variant_dir/feature_model"
    fi
  else
    rm -f "$variant_dir/feature_model_path.txt"
    if [[ -L "$variant_dir/feature_model" ]]; then
      rm -f "$variant_dir/feature_model"
    fi
  fi

  if [[ -f "$BASE_MODEL/input.ply" && ! -e "$variant_dir/input.ply" ]]; then
    ln -s "$BASE_MODEL/input.ply" "$variant_dir/input.ply"
  fi

  local linked_any=0
  for iter_dir in "$BASE_MODEL"/point_cloud/iteration_*; do
    [[ -d "$iter_dir" ]] || continue
    if [[ -f "$iter_dir/scene_point_cloud.ply" || -f "$iter_dir/point_cloud.ply" || -f "$iter_dir/contrastive_feature_point_cloud.ply" ]]; then
      local iter_name
      iter_name="$(basename "$iter_dir")"
      if [[ "$iter_name" == "iteration_${TRAIN_ITERATIONS}" && "$SKIP_TRAIN" != "1" ]]; then
        echo "Skipping base scene checkpoint $iter_name so contrastive features can be saved there." >&2
        continue
      fi
      ln -sfn "$iter_dir" "$variant_dir/point_cloud/$iter_name"
      linked_any=1
    fi
  done

  if [[ $linked_any -eq 0 ]]; then
    echo "No checkpoints were found under $BASE_MODEL/point_cloud" >&2
    exit 1
  fi
}

write_run_context() {
  python - <<'PY' "$RUN_CONTEXT_JSON" "$SOURCE_PATH" "$BASE_MODEL" "$ABLATION_ROOT" "$VARIANT_LIST"
import json
import os
import sys

path, source_path, base_model, ablation_root, variant_list = sys.argv[1:6]
data = {
    "source_path": source_path,
    "base_model": base_model,
    "ablation_root": ablation_root,
    "variant_list": [item for item in variant_list.split(",") if item],
    "train_iterations": int(os.environ["TRAIN_ITERATIONS"]),
    "feature_iteration_override": int(os.environ["FEATURE_ITERATION_OVERRIDE"]),
    "num_sampled_rays": int(os.environ["NUM_SAMPLED_RAYS"]),
    "viz_interval": int(os.environ["VIZ_INTERVAL"]),
    "viz_view_idx": os.environ["VIZ_VIEW_IDX"],
    "viz_mode": os.environ["VIZ_MODE"],
    "viz_max_samples": int(os.environ["VIZ_MAX_SAMPLES"]),
    "eval_scales": [float(x) for x in os.environ["EVAL_SCALES"].split()],
    "eval_cluster_methods": os.environ["EVAL_CLUSTER_METHODS"].split(),
    "eval_max_views": int(os.environ["EVAL_MAX_VIEWS"]),
    "eval_view_stride": int(os.environ["EVAL_VIEW_STRIDE"]),
    "eval_render_downsample": int(os.environ["EVAL_RENDER_DOWNSAMPLE"]),
    "supervision_mode": os.environ["SUPERVISION_MODE"],
    "mask_scale_target": float(os.environ["MASK_SCALE_TARGET"]),
    "mask_scale_tolerance": float(os.environ["MASK_SCALE_TOLERANCE"]),
    "mask_min_area": int(os.environ["MASK_MIN_AREA"]),
    "mask_max_area_ratio": float(os.environ["MASK_MAX_AREA_RATIO"]),
    "mask_max_iou_overlap": float(os.environ["MASK_MAX_IOU_OVERLAP"]),
    "mask_boundary_erode_kernel": int(os.environ["MASK_BOUNDARY_ERODE_KERNEL"]),
    "min_valid_masks_per_view": int(os.environ["MIN_VALID_MASKS_PER_VIEW"]),
    "graph_laplacian_weight": float(os.environ["GRAPH_LAPLACIAN_WEIGHT"]),
    "graph_laplacian_samples": int(os.environ["GRAPH_LAPLACIAN_SAMPLES"]),
    "graph_laplacian_scales": int(os.environ["GRAPH_LAPLACIAN_SCALES"]),
    "graph_laplacian_k": int(os.environ["GRAPH_LAPLACIAN_K"]),
    "graph_spatial_weight": float(os.environ["GRAPH_SPATIAL_WEIGHT"]),
    "graph_sh0_color_weight": float(os.environ["GRAPH_SH0_COLOR_WEIGHT"]),
    "graph_sh0_color_sigma": float(os.environ["GRAPH_SH0_COLOR_SIGMA"]),
    "cluster_sample_size": int(os.environ["CLUSTER_SAMPLE_SIZE"]),
    "cluster_graph_k": int(os.environ["CLUSTER_GRAPH_K"]),
    "cluster_max_clusters": int(os.environ["CLUSTER_MAX_CLUSTERS"]),
    "cluster_min_cluster_size": int(os.environ["CLUSTER_MIN_CLUSTER_SIZE"]),
    "cluster_cut_threshold": float(os.environ["CLUSTER_CUT_THRESHOLD"]),
    "cluster_spatial_weight": float(os.environ["CLUSTER_SPATIAL_WEIGHT"]),
    "cluster_sh0_color_weight": float(os.environ["CLUSTER_SH0_COLOR_WEIGHT"]),
    "cluster_sh0_color_sigma": float(os.environ["CLUSTER_SH0_COLOR_SIGMA"]),
    "cluster_mesh_weight": float(os.environ["CLUSTER_MESH_WEIGHT"]),
    "cluster_mesh_path": os.environ["CLUSTER_MESH_PATH"] or None,
    "hdbscan_min_cluster_size": int(os.environ["HDBSCAN_MIN_CLUSTER_SIZE"]),
    "hdbscan_epsilon": float(os.environ["HDBSCAN_EPSILON"]),
    "connectivity_k": int(os.environ["CONNECTIVITY_K"]),
    "save_viz_views": int(os.environ["SAVE_VIZ_VIEWS"]),
    "boundary_negative_weight": float(os.environ["BOUNDARY_NEGATIVE_WEIGHT"]),
    "boundary_negative_margin": float(os.environ["BOUNDARY_NEGATIVE_MARGIN"]),
    "boundary_band_kernel": int(os.environ["BOUNDARY_BAND_KERNEL"]),
    "summary_sort_key": os.environ["SUMMARY_SORT_KEY"],
    "skip_train": os.environ["SKIP_TRAIN"] == "1",
    "skip_eval": os.environ["SKIP_EVAL"] == "1",
}
with open(path, "w") as handle:
    json.dump(data, handle, indent=2)
PY
}

declare -a TRAIN_ARGS=()
declare -a EVAL_ARGS=()
VARIANT_HAS_METHOD=0

resolve_variant_args() {
  local variant_name="$1"
  TRAIN_ARGS=()
  EVAL_ARGS=()
  VARIANT_HAS_METHOD=0
  case "$variant_name" in
    baseline)
      TRAIN_ARGS+=(--graph_laplacian_weight 0.0 --boundary_negative_weight 0.0 --graph_sh0_color_weight 0.0)
      ;;
    laplacian)
      TRAIN_ARGS+=(
        --graph_laplacian_weight "$GRAPH_LAPLACIAN_WEIGHT"
        --graph_laplacian_samples "$GRAPH_LAPLACIAN_SAMPLES"
        --graph_laplacian_scales "$GRAPH_LAPLACIAN_SCALES"
        --graph_laplacian_k "$GRAPH_LAPLACIAN_K"
        --graph_spatial_weight "$GRAPH_SPATIAL_WEIGHT"
        --boundary_negative_weight 0.0
        --graph_sh0_color_weight 0.0
      )
      ;;
    laplacian_boundary)
      TRAIN_ARGS+=(
        --graph_laplacian_weight "$GRAPH_LAPLACIAN_WEIGHT"
        --graph_laplacian_samples "$GRAPH_LAPLACIAN_SAMPLES"
        --graph_laplacian_scales "$GRAPH_LAPLACIAN_SCALES"
        --graph_laplacian_k "$GRAPH_LAPLACIAN_K"
        --graph_spatial_weight "$GRAPH_SPATIAL_WEIGHT"
        --boundary_negative_weight "$BOUNDARY_NEGATIVE_WEIGHT"
        --boundary_negative_margin "$BOUNDARY_NEGATIVE_MARGIN"
        --boundary_band_kernel "$BOUNDARY_BAND_KERNEL"
        --graph_sh0_color_weight 0.0
      )
      ;;
    laplacian_boundary_color)
      TRAIN_ARGS+=(
        --graph_laplacian_weight "$GRAPH_LAPLACIAN_WEIGHT"
        --graph_laplacian_samples "$GRAPH_LAPLACIAN_SAMPLES"
        --graph_laplacian_scales "$GRAPH_LAPLACIAN_SCALES"
        --graph_laplacian_k "$GRAPH_LAPLACIAN_K"
        --graph_spatial_weight "$GRAPH_SPATIAL_WEIGHT"
        --boundary_negative_weight "$BOUNDARY_NEGATIVE_WEIGHT"
        --boundary_negative_margin "$BOUNDARY_NEGATIVE_MARGIN"
        --boundary_band_kernel "$BOUNDARY_BAND_KERNEL"
        --graph_sh0_color_weight "$GRAPH_SH0_COLOR_WEIGHT"
        --graph_sh0_color_sigma "$GRAPH_SH0_COLOR_SIGMA"
      )
      ;;
    multiscale)
      TRAIN_ARGS+=(--supervision_mode multiscale)
      EVAL_ARGS+=(--supervision_mode multiscale)
      ;;
    single_scale_band_035)
      TRAIN_ARGS+=(--supervision_mode single_scale_band --mask_scale_target 0.35)
      EVAL_ARGS+=(--supervision_mode single_scale_band --mask_scale_target 0.35)
      ;;
    single_scale_band_050)
      TRAIN_ARGS+=(--supervision_mode single_scale_band --mask_scale_target 0.50)
      EVAL_ARGS+=(--supervision_mode single_scale_band --mask_scale_target 0.50)
      ;;
    single_scale_band_060)
      TRAIN_ARGS+=(--supervision_mode single_scale_band --mask_scale_target 0.60)
      EVAL_ARGS+=(--supervision_mode single_scale_band --mask_scale_target 0.60)
      ;;
    sanity_graphlap_0p02)
      TRAIN_ARGS+=(
        --graph_laplacian_weight 0.02
        --graph_laplacian_samples "$GRAPH_LAPLACIAN_SAMPLES"
        --graph_laplacian_scales "$GRAPH_LAPLACIAN_SCALES"
        --graph_laplacian_k "$GRAPH_LAPLACIAN_K"
        --graph_spatial_weight "$GRAPH_SPATIAL_WEIGHT"
        --boundary_negative_weight 0.0
        --graph_sh0_color_weight 0.0
      )
      ;;
    sanity_graphlap_0p05)
      TRAIN_ARGS+=(
        --graph_laplacian_weight 0.05
        --graph_laplacian_samples "$GRAPH_LAPLACIAN_SAMPLES"
        --graph_laplacian_scales "$GRAPH_LAPLACIAN_SCALES"
        --graph_laplacian_k "$GRAPH_LAPLACIAN_K"
        --graph_spatial_weight "$GRAPH_SPATIAL_WEIGHT"
        --boundary_negative_weight 0.0
        --graph_sh0_color_weight 0.0
      )
      ;;
    sanity_graphlap_0p10)
      TRAIN_ARGS+=(
        --graph_laplacian_weight 0.10
        --graph_laplacian_samples "$GRAPH_LAPLACIAN_SAMPLES"
        --graph_laplacian_scales "$GRAPH_LAPLACIAN_SCALES"
        --graph_laplacian_k "$GRAPH_LAPLACIAN_K"
        --graph_spatial_weight "$GRAPH_SPATIAL_WEIGHT"
        --boundary_negative_weight 0.0
        --graph_sh0_color_weight 0.0
      )
      ;;
    sanity_boundary_0p05)
      TRAIN_ARGS+=(
        --graph_laplacian_weight "$GRAPH_LAPLACIAN_WEIGHT"
        --graph_laplacian_samples "$GRAPH_LAPLACIAN_SAMPLES"
        --graph_laplacian_scales "$GRAPH_LAPLACIAN_SCALES"
        --graph_laplacian_k "$GRAPH_LAPLACIAN_K"
        --graph_spatial_weight "$GRAPH_SPATIAL_WEIGHT"
        --boundary_negative_weight 0.05
        --boundary_negative_margin "$BOUNDARY_NEGATIVE_MARGIN"
        --boundary_band_kernel "$BOUNDARY_BAND_KERNEL"
        --graph_sh0_color_weight 0.0
      )
      ;;
    sanity_boundary_0p10)
      TRAIN_ARGS+=(
        --graph_laplacian_weight "$GRAPH_LAPLACIAN_WEIGHT"
        --graph_laplacian_samples "$GRAPH_LAPLACIAN_SAMPLES"
        --graph_laplacian_scales "$GRAPH_LAPLACIAN_SCALES"
        --graph_laplacian_k "$GRAPH_LAPLACIAN_K"
        --graph_spatial_weight "$GRAPH_SPATIAL_WEIGHT"
        --boundary_negative_weight 0.10
        --boundary_negative_margin "$BOUNDARY_NEGATIVE_MARGIN"
        --boundary_band_kernel "$BOUNDARY_BAND_KERNEL"
        --graph_sh0_color_weight 0.0
      )
      ;;
    sanity_boundary_0p20)
      TRAIN_ARGS+=(
        --graph_laplacian_weight "$GRAPH_LAPLACIAN_WEIGHT"
        --graph_laplacian_samples "$GRAPH_LAPLACIAN_SAMPLES"
        --graph_laplacian_scales "$GRAPH_LAPLACIAN_SCALES"
        --graph_laplacian_k "$GRAPH_LAPLACIAN_K"
        --graph_spatial_weight "$GRAPH_SPATIAL_WEIGHT"
        --boundary_negative_weight 0.20
        --boundary_negative_margin "$BOUNDARY_NEGATIVE_MARGIN"
        --boundary_band_kernel "$BOUNDARY_BAND_KERNEL"
        --graph_sh0_color_weight 0.0
      )
      ;;
    sanity_sh0color_0p5)
      TRAIN_ARGS+=(
        --graph_laplacian_weight "$GRAPH_LAPLACIAN_WEIGHT"
        --graph_laplacian_samples "$GRAPH_LAPLACIAN_SAMPLES"
        --graph_laplacian_scales "$GRAPH_LAPLACIAN_SCALES"
        --graph_laplacian_k "$GRAPH_LAPLACIAN_K"
        --graph_spatial_weight "$GRAPH_SPATIAL_WEIGHT"
        --boundary_negative_weight "$BOUNDARY_NEGATIVE_WEIGHT"
        --boundary_negative_margin "$BOUNDARY_NEGATIVE_MARGIN"
        --boundary_band_kernel "$BOUNDARY_BAND_KERNEL"
        --graph_sh0_color_weight 0.5
        --graph_sh0_color_sigma "$GRAPH_SH0_COLOR_SIGMA"
      )
      ;;
    sanity_sh0color_1p0)
      TRAIN_ARGS+=(
        --graph_laplacian_weight "$GRAPH_LAPLACIAN_WEIGHT"
        --graph_laplacian_samples "$GRAPH_LAPLACIAN_SAMPLES"
        --graph_laplacian_scales "$GRAPH_LAPLACIAN_SCALES"
        --graph_laplacian_k "$GRAPH_LAPLACIAN_K"
        --graph_spatial_weight "$GRAPH_SPATIAL_WEIGHT"
        --boundary_negative_weight "$BOUNDARY_NEGATIVE_WEIGHT"
        --boundary_negative_margin "$BOUNDARY_NEGATIVE_MARGIN"
        --boundary_band_kernel "$BOUNDARY_BAND_KERNEL"
        --graph_sh0_color_weight 1.0
        --graph_sh0_color_sigma "$GRAPH_SH0_COLOR_SIGMA"
      )
      ;;
    sanity_sh0color_2p0)
      TRAIN_ARGS+=(
        --graph_laplacian_weight "$GRAPH_LAPLACIAN_WEIGHT"
        --graph_laplacian_samples "$GRAPH_LAPLACIAN_SAMPLES"
        --graph_laplacian_scales "$GRAPH_LAPLACIAN_SCALES"
        --graph_laplacian_k "$GRAPH_LAPLACIAN_K"
        --graph_spatial_weight "$GRAPH_SPATIAL_WEIGHT"
        --boundary_negative_weight "$BOUNDARY_NEGATIVE_WEIGHT"
        --boundary_negative_margin "$BOUNDARY_NEGATIVE_MARGIN"
        --boundary_band_kernel "$BOUNDARY_BAND_KERNEL"
        --graph_sh0_color_weight 2.0
        --graph_sh0_color_sigma "$GRAPH_SH0_COLOR_SIGMA"
      )
      ;;
    hdbscan)
      VARIANT_HAS_METHOD=1
      EVAL_ARGS+=(--cluster_method HDBSCAN)
      ;;
    hdbscan_refined_xyz)
      VARIANT_HAS_METHOD=1
      EVAL_ARGS+=(--cluster_method HDBSCANRefined --cluster_spatial_weight 1.0 --cluster_sh0_color_weight 0.0)
      ;;
    hdbscan_refined_xyz_sh0)
      VARIANT_HAS_METHOD=1
      EVAL_ARGS+=(--cluster_method HDBSCANRefined --cluster_spatial_weight 1.0 --cluster_sh0_color_weight "$CLUSTER_SH0_COLOR_WEIGHT" --cluster_sh0_color_sigma "$CLUSTER_SH0_COLOR_SIGMA")
      ;;
    ncut_xyz)
      VARIANT_HAS_METHOD=1
      EVAL_ARGS+=(--cluster_method NormalizedCut --cluster_spatial_weight 1.0 --cluster_sh0_color_weight 0.0)
      ;;
    ncut_xyz_sh0)
      VARIANT_HAS_METHOD=1
      EVAL_ARGS+=(--cluster_method NormalizedCut --cluster_spatial_weight 1.0 --cluster_sh0_color_weight "$CLUSTER_SH0_COLOR_WEIGHT" --cluster_sh0_color_sigma "$CLUSTER_SH0_COLOR_SIGMA")
      ;;
    ncut_sh0)
      VARIANT_HAS_METHOD=1
      EVAL_ARGS+=(--cluster_method NormalizedCut --cluster_spatial_weight 0.0 --cluster_sh0_color_weight "$CLUSTER_SH0_COLOR_WEIGHT" --cluster_sh0_color_sigma "$CLUSTER_SH0_COLOR_SIGMA")
      ;;
    ncut_feature)
      VARIANT_HAS_METHOD=1
      EVAL_ARGS+=(--cluster_method NormalizedCut --cluster_spatial_weight 0.0 --cluster_sh0_color_weight 0.0)
      ;;
    *)
      echo "Unknown ablation variant: $variant_name" >&2
      exit 1
      ;;
  esac
}

declare -a EXECUTED_JSONS=()

run_eval_variant() {
  local variant_name="$1"
  local variant_dir="$2"
  local method="$3"
  shift 3
  local method_slug
  method_slug="$(echo "$method" | tr '[:upper:]' '[:lower:]')"
  python eval_ablation.py \
    -s "$SOURCE_PATH" \
    -m "$variant_dir" \
    --label "$variant_name" \
    --iteration "$FEATURE_ITERATION_OVERRIDE" \
    --cluster_method "$method" \
    --cluster_sample_size "$CLUSTER_SAMPLE_SIZE" \
    --cluster_graph_k "$CLUSTER_GRAPH_K" \
    --cluster_max_clusters "$CLUSTER_MAX_CLUSTERS" \
    --cluster_min_cluster_size "$CLUSTER_MIN_CLUSTER_SIZE" \
    --cluster_cut_threshold "$CLUSTER_CUT_THRESHOLD" \
    --cluster_spatial_weight "$CLUSTER_SPATIAL_WEIGHT" \
    --cluster_sh0_color_weight "$CLUSTER_SH0_COLOR_WEIGHT" \
    --cluster_sh0_color_sigma "$CLUSTER_SH0_COLOR_SIGMA" \
    --cluster_mesh_weight "$CLUSTER_MESH_WEIGHT" \
    --hdbscan_min_cluster_size "$HDBSCAN_MIN_CLUSTER_SIZE" \
    --hdbscan_epsilon "$HDBSCAN_EPSILON" \
    --connectivity_k "$CONNECTIVITY_K" \
    --camera_split test \
    --max_views "$EVAL_MAX_VIEWS" \
    --view_stride "$EVAL_VIEW_STRIDE" \
    --eval_scales $EVAL_SCALES \
    --render_downsample "$EVAL_RENDER_DOWNSAMPLE" \
    --supervision_mode "$SUPERVISION_MODE" \
    --mask_scale_target "$MASK_SCALE_TARGET" \
    --mask_scale_tolerance "$MASK_SCALE_TOLERANCE" \
    --mask_min_area "$MASK_MIN_AREA" \
    --mask_max_area_ratio "$MASK_MAX_AREA_RATIO" \
    --mask_max_iou_overlap "$MASK_MAX_IOU_OVERLAP" \
    --mask_boundary_erode_kernel "$MASK_BOUNDARY_ERODE_KERNEL" \
    --min_valid_masks_per_view "$MIN_VALID_MASKS_PER_VIEW" \
    --save_viz_views "$SAVE_VIZ_VIEWS" \
    "$@"${CLUSTER_MESH_PATH:+ \
    --cluster_mesh_path "$CLUSTER_MESH_PATH"}
  EXECUTED_JSONS+=("$variant_dir/ablation_eval/${variant_name}_${method_slug}.json")
}

run_variant() {
  local variant_name="$1"
  local variant_dir="$ABLATION_ROOT/$variant_name"
  prepare_variant_root "$variant_name"
  resolve_variant_args "$variant_name"

  if [[ "$SKIP_TRAIN" != "1" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" python train_contrastive_feature.py \
      -s "$SOURCE_PATH" \
      -m "$variant_dir" \
      --iterations "$TRAIN_ITERATIONS" \
      --num_sampled_rays "$NUM_SAMPLED_RAYS" \
      --viz_interval "$VIZ_INTERVAL" \
      --viz_view_idx "$VIZ_VIEW_IDX" \
      --viz_mode "$VIZ_MODE" \
      --viz_max_samples "$VIZ_MAX_SAMPLES" \
      --supervision_mode "$SUPERVISION_MODE" \
      --mask_scale_target "$MASK_SCALE_TARGET" \
      --mask_scale_tolerance "$MASK_SCALE_TOLERANCE" \
      --mask_min_area "$MASK_MIN_AREA" \
      --mask_max_area_ratio "$MASK_MAX_AREA_RATIO" \
      --mask_max_iou_overlap "$MASK_MAX_IOU_OVERLAP" \
      --mask_boundary_erode_kernel "$MASK_BOUNDARY_ERODE_KERNEL" \
      --min_valid_masks_per_view "$MIN_VALID_MASKS_PER_VIEW" \
      --graph_spatial_weight "$GRAPH_SPATIAL_WEIGHT" \
      --graph_sh0_color_weight "$GRAPH_SH0_COLOR_WEIGHT" \
      --graph_sh0_color_sigma "$GRAPH_SH0_COLOR_SIGMA" \
      "${TRAIN_ARGS[@]}"
  fi

  if [[ "$SKIP_EVAL" == "1" ]]; then
    return
  fi

  if [[ $VARIANT_HAS_METHOD -eq 1 ]]; then
    local explicit_method="NormalizedCut"
    for ((i=0; i<${#EVAL_ARGS[@]}; ++i)); do
      if [[ "${EVAL_ARGS[$i]}" == "--cluster_method" ]]; then
        explicit_method="${EVAL_ARGS[$((i + 1))]}"
        break
      fi
    done
    run_eval_variant "$variant_name" "$variant_dir" "$explicit_method" "${EVAL_ARGS[@]}"
    return
  fi

  for method in $EVAL_CLUSTER_METHODS; do
    run_eval_variant "$variant_name" "$variant_dir" "$method" "${EVAL_ARGS[@]}"
  done
}

write_run_context

IFS=',' read -r -a VARIANTS <<< "$VARIANT_LIST"
for variant_name in "${VARIANTS[@]}"; do
  [[ -n "$variant_name" ]] || continue
  run_variant "$variant_name"
done

if [[ "$SKIP_EVAL" != "1" && ${#EXECUTED_JSONS[@]} -gt 0 ]]; then
  python summarize_ablation_results.py \
    "${EXECUTED_JSONS[@]}" \
    --output_csv "$ABLATION_ROOT/ablation_summary.csv" \
    --output_md "$ABLATION_ROOT/ablation_summary.md" \
    --sort_key "$SUMMARY_SORT_KEY" \
    --descending
  echo "Ablation study finished. Summary files:"
  echo "  $ABLATION_ROOT/ablation_summary.csv"
  echo "  $ABLATION_ROOT/ablation_summary.md"
fi
