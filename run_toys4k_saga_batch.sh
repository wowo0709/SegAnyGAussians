#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_toys4k_saga_batch.sh --gpus 0,1,2,3 [options]

Options:
  --gpus IDS                 Comma- or space-separated GPU ids. Required.
  --data-root PATH           Toys4k dataset root.
  --models-dir PATH          Override 3DGS model root.
  --scenes-dir PATH          Override scene metadata root.
  --renders-dir PATH         Override render image root.
  --sam-checkpoint PATH      SAM checkpoint path.
  --sam-points-per-side N    SAM automatic mask grid density. Default: 32
  --sam-pred-iou-thresh F    SAM predicted IoU filter. Default: 0.88
  --sam-stability-score-thresh F
                             SAM stability filter. Default: 0.95
  --sam-crop-n-layers N      SAM crop pyramid depth. Default: 0
  --sam-crop-n-points-downscale-factor N
                             Crop-layer point grid downscale factor. Default: 1
  --sam-min-mask-region-area N
                             Minimum area for SAM small-region cleanup. Default: 100
  --instances IDS            Comma-separated subset of instance ids to run.
  --instance-regex REGEX     Only run instances whose id matches REGEX.
  --downsample N             Downsample passed to extract script. Default: 1
  --downsample-type TYPE     image or mask. Default: image
  --iterations N             Contrastive feature iterations. Default: 10000
  --num-sampled-rays N       Contrastive feature ray count. Default: 1000
  --scale-definition NAME    Scale definition for get_scale.py. Default: legacy_3d_std
  --mask-scales-dir-name NAME
                             Directory under each render root for saved scales. Default: mask_scales
  --supervision-mode MODE    Contrastive supervision mode. Default: multiscale
  --mask-scale-target F      Target scale band center. Default: 0.5
  --mask-scale-tolerance F   Target scale band tolerance. Default: 0.1
  --boundary-negative-weight F
                             Boundary negative loss weight. Default: 0.0
  --boundary-negative-margin F
                             Boundary negative margin. Default: 0.15
  --boundary-band-kernel N   Boundary band kernel size. Default: 5
  --min-valid-masks-per-view N
                             Minimum selected masks for a view. Default: 2
  --feature-subdir NAME      Feature output subdir under each model. Default: contrastive_feature
  --viz-interval N           train_contrastive_feature.py --viz_interval. Default: 0
  --viz-view-idx "0 1 2"     train_contrastive_feature.py --viz_view_idx. Default: 0
  --viz-mode MODE            embedding or cluster. Default: embedding
  --log-root PATH            Root directory for per-instance logs.
  --force                    Re-run stages even if outputs already exist.
  --skip-extract             Skip extract_segment_everything_masks.py
  --skip-scale               Skip get_scale.py
  --skip-train               Skip train_contrastive_feature.py
  --help                     Show this message.

Environment overrides:
  PYTHON_BIN                Default: python

Example:
  ./run_toys4k_saga_batch.sh \
    --gpus 0,1,2,3 \
    --downsample 1 \
    --iterations 10000 \
    --num-sampled-rays 1000
EOF
}

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

join_by() {
  local delimiter="$1"
  shift
  local first=1
  for item in "$@"; do
    if [[ $first -eq 1 ]]; then
      printf '%s' "$item"
      first=0
    else
      printf '%s%s' "$delimiter" "$item"
    fi
  done
}

parse_list() {
  local raw="$1"
  printf '%s\n' "$raw" | tr ', ' '\n\n' | sed '/^$/d'
}

count_pt_files() {
  local dir="$1"
  if [[ ! -d "$dir" ]]; then
    echo 0
    return 0
  fi
  find "$dir" -maxdepth 1 -type f -name '*.pt' | wc -l | tr -d ' '
}

count_png_files() {
  local dir="$1"
  if [[ ! -d "$dir" ]]; then
    echo 0
    return 0
  fi
  find "$dir" -maxdepth 1 -type f -name '*.png' | wc -l | tr -d ' '
}

extract_done() {
  local render_root="$1"
  local pt_count png_count
  pt_count="$(count_pt_files "$render_root/sam_masks")"
  png_count="$(count_png_files "$render_root/sam_masks_preview")"
  [[ "$pt_count" -gt 0 && "$png_count" -ge "$pt_count" ]]
}

scale_done() {
  local render_root="$1"
  local mask_scales_dir_name="$2"
  [[ "$(count_pt_files "$render_root/$mask_scales_dir_name")" -gt 0 ]]
}

train_done() {
  local feature_root="$1"
  local iterations="$2"
  [[ -f "$feature_root/point_cloud/iteration_${iterations}/contrastive_feature_point_cloud.ply" ]]
}

run_stage() {
  local stage_name="$1"
  local stage_log="$2"
  shift 2

  echo "[$(timestamp)] [$stage_name] Starting"
  echo "[$(timestamp)] [$stage_name] Log: $stage_log"
  echo "[$(timestamp)] [$stage_name] Command: $*"

  if "$@" >"$stage_log" 2>&1; then
    echo "[$(timestamp)] [$stage_name] Completed"
    return 0
  fi

  local rc=$?
  echo "[$(timestamp)] [$stage_name] Failed with exit code $rc"
  echo "[$(timestamp)] [$stage_name] Tail of $stage_log:"
  tail -n 40 "$stage_log" || true
  return "$rc"
}

run_instance_pipeline() {
  local gpu_id="$1"
  local instance_id="$2"

  local render_root="$RENDERS_DIR/$instance_id"
  local scene_root="$SCENES_DIR/$instance_id"
  local model_root="$MODELS_DIR/$instance_id"
  local feature_root="$model_root/$FEATURE_SUBDIR"
  local instance_log_dir="$LOG_ROOT/$instance_id"
  local worker_log="$instance_log_dir/worker.log"

  mkdir -p "$instance_log_dir"
  : >"$worker_log"
  exec >>"$worker_log" 2>&1

  echo "[$(timestamp)] Instance: $instance_id"
  echo "[$(timestamp)] GPU: $gpu_id"
  echo "[$(timestamp)] Render root: $render_root"
  echo "[$(timestamp)] Scene root: $scene_root"
  echo "[$(timestamp)] Model root: $model_root"
  echo "[$(timestamp)] Feature root: $feature_root"
  echo "[$(timestamp)] Scale definition: $SCALE_DEFINITION"
  echo "[$(timestamp)] Mask scales dir: $MASK_SCALES_DIR_NAME"

  if [[ "$SKIP_EXTRACT" != "1" ]]; then
    if [[ "$FORCE" != "1" ]] && extract_done "$render_root"; then
      echo "[$(timestamp)] [extract] Skipping because sam_masks already exist."
    else
      run_stage \
        "extract" \
        "$instance_log_dir/extract.log" \
        env CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" extract_segment_everything_masks.py \
          --image_root "$render_root" \
          --sam_checkpoint_path "$SAM_CHECKPOINT" \
          --downsample "$DOWNSAMPLE" \
          --downsample_type "$DOWNSAMPLE_TYPE" \
          --points-per-side "$SAM_POINTS_PER_SIDE" \
          --pred-iou-thresh "$SAM_PRED_IOU_THRESH" \
          --stability-score-thresh "$SAM_STABILITY_SCORE_THRESH" \
          --crop-n-layers "$SAM_CROP_N_LAYERS" \
          --crop-n-points-downscale-factor "$SAM_CROP_N_POINTS_DOWNSCALE_FACTOR" \
          --min-mask-region-area "$SAM_MIN_MASK_REGION_AREA"
    fi
  else
    echo "[$(timestamp)] [extract] Skipped by user."
  fi

  if [[ "$SKIP_SCALE" != "1" ]]; then
    if [[ "$FORCE" != "1" ]] && scale_done "$render_root" "$MASK_SCALES_DIR_NAME"; then
      echo "[$(timestamp)] [scale] Skipping because $MASK_SCALES_DIR_NAME already exists."
    else
      run_stage \
        "scale" \
        "$instance_log_dir/scale.log" \
        env CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" get_scale.py \
          --image_root "$render_root" \
          --source_path "$scene_root" \
          --images "$render_root" \
          --model_path "$model_root" \
          --scale-definition "$SCALE_DEFINITION" \
          --mask-scales-dir-name "$MASK_SCALES_DIR_NAME"
    fi
  else
    echo "[$(timestamp)] [scale] Skipped by user."
  fi

  if [[ "$SKIP_TRAIN" != "1" ]]; then
    if [[ "$FORCE" != "1" ]] && train_done "$feature_root" "$ITERATIONS"; then
      echo "[$(timestamp)] [train] Skipping because contrastive feature checkpoint already exists."
    else
      local -a train_cmd
      local -a viz_view_idx_args=()
      if [[ -n "$VIZ_VIEW_IDX" ]]; then
        # shellcheck disable=SC2206
        local tmp_viz_args=( $VIZ_VIEW_IDX )
        if [[ ${#tmp_viz_args[@]} -gt 0 ]]; then
          viz_view_idx_args=(--viz_view_idx "${tmp_viz_args[@]}")
        fi
      fi

      train_cmd=(
        env CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" train_contrastive_feature.py
        --source_path "$scene_root"
        --images "$render_root"
        --model_path "$model_root"
        --feature_model_path "$feature_root"
        --iterations "$ITERATIONS"
        --num_sampled_rays "$NUM_SAMPLED_RAYS"
        --mask_scales_dir_name "$MASK_SCALES_DIR_NAME"
        --supervision_mode "$SUPERVISION_MODE"
        --mask_scale_target "$MASK_SCALE_TARGET"
        --mask_scale_tolerance "$MASK_SCALE_TOLERANCE"
        --boundary_negative_weight "$BOUNDARY_NEGATIVE_WEIGHT"
        --boundary_negative_margin "$BOUNDARY_NEGATIVE_MARGIN"
        --boundary_band_kernel "$BOUNDARY_BAND_KERNEL"
        --min_valid_masks_per_view "$MIN_VALID_MASKS_PER_VIEW"
        --viz_interval "$VIZ_INTERVAL"
        --viz_mode "$VIZ_MODE"
      )
      if [[ ${#viz_view_idx_args[@]} -gt 0 ]]; then
        train_cmd+=("${viz_view_idx_args[@]}")
      fi

      run_stage \
        "train" \
        "$instance_log_dir/train.log" \
        "${train_cmd[@]}"
    fi
  else
    echo "[$(timestamp)] [train] Skipped by user."
  fi

  echo "[$(timestamp)] Instance completed successfully."
}

reap_finished_slots() {
  local idx
  for idx in "${!SLOT_PIDS[@]}"; do
    local pid="${SLOT_PIDS[$idx]:-}"
    [[ -n "$pid" ]] || continue
    if kill -0 "$pid" 2>/dev/null; then
      continue
    fi

    local instance_id="${SLOT_INSTANCES[$idx]}"
    local gpu_id="${SLOT_GPUS[$idx]}"
    local worker_log="${SLOT_LOGS[$idx]}"

    if wait "$pid"; then
      echo "[$(timestamp)] Finished instance $instance_id on GPU $gpu_id"
      SUCCEEDED_INSTANCES+=("$instance_id")
    else
      local rc=$?
      echo "[$(timestamp)] FAILED instance $instance_id on GPU $gpu_id with exit code $rc"
      echo "[$(timestamp)] See log: $worker_log"
      FAILED_INSTANCES+=("$instance_id|$gpu_id|$rc|$worker_log")
    fi

    SLOT_PIDS[$idx]=""
    SLOT_INSTANCES[$idx]=""
    SLOT_LOGS[$idx]=""
  done
}

find_free_slot() {
  while true; do
    reap_finished_slots
    local idx
    for idx in "${!SLOT_PIDS[@]}"; do
      if [[ -z "${SLOT_PIDS[$idx]:-}" ]]; then
        echo "$idx"
        return 0
      fi
    done
    wait -n || true
  done
}

cleanup() {
  local idx
  for idx in "${!SLOT_PIDS[@]}"; do
    local pid="${SLOT_PIDS[$idx]:-}"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_DATA_ROOT="/root/node1/data2/diyscene/dataset/toys4k"
DATA_ROOT="${DATA_ROOT:-$DEFAULT_DATA_ROOT}"
MODELS_DIR="${MODELS_DIR:-}"
SCENES_DIR="${SCENES_DIR:-}"
RENDERS_DIR="${RENDERS_DIR:-}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-}"
SAM_POINTS_PER_SIDE="${SAM_POINTS_PER_SIDE:-32}"
SAM_PRED_IOU_THRESH="${SAM_PRED_IOU_THRESH:-0.88}"
SAM_STABILITY_SCORE_THRESH="${SAM_STABILITY_SCORE_THRESH:-0.95}"
SAM_CROP_N_LAYERS="${SAM_CROP_N_LAYERS:-0}"
SAM_CROP_N_POINTS_DOWNSCALE_FACTOR="${SAM_CROP_N_POINTS_DOWNSCALE_FACTOR:-1}"
SAM_MIN_MASK_REGION_AREA="${SAM_MIN_MASK_REGION_AREA:-100}"
GPU_LIST="${GPU_LIST:-}"
INSTANCE_IDS_RAW="${INSTANCE_IDS:-}"
INSTANCE_REGEX="${INSTANCE_REGEX:-}"
DOWNSAMPLE="${DOWNSAMPLE:-1}"
DOWNSAMPLE_TYPE="${DOWNSAMPLE_TYPE:-image}"
ITERATIONS="${ITERATIONS:-10000}"
NUM_SAMPLED_RAYS="${NUM_SAMPLED_RAYS:-1000}"
SCALE_DEFINITION="${SCALE_DEFINITION:-legacy_3d_std}"
MASK_SCALES_DIR_NAME="${MASK_SCALES_DIR_NAME:-mask_scales}"
SUPERVISION_MODE="${SUPERVISION_MODE:-multiscale}"
MASK_SCALE_TARGET="${MASK_SCALE_TARGET:-0.5}"
MASK_SCALE_TOLERANCE="${MASK_SCALE_TOLERANCE:-0.1}"
BOUNDARY_NEGATIVE_WEIGHT="${BOUNDARY_NEGATIVE_WEIGHT:-0.0}"
BOUNDARY_NEGATIVE_MARGIN="${BOUNDARY_NEGATIVE_MARGIN:-0.15}"
BOUNDARY_BAND_KERNEL="${BOUNDARY_BAND_KERNEL:-5}"
MIN_VALID_MASKS_PER_VIEW="${MIN_VALID_MASKS_PER_VIEW:-2}"
FEATURE_SUBDIR="${FEATURE_SUBDIR:-contrastive_feature}"
VIZ_INTERVAL="${VIZ_INTERVAL:-0}"
VIZ_VIEW_IDX="${VIZ_VIEW_IDX:-0}"
VIZ_MODE="${VIZ_MODE:-embedding}"
LOG_ROOT="${LOG_ROOT:-}"
FORCE="${FORCE:-0}"
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"
SKIP_SCALE="${SKIP_SCALE:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      GPU_LIST="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --models-dir)
      MODELS_DIR="$2"
      shift 2
      ;;
    --scenes-dir)
      SCENES_DIR="$2"
      shift 2
      ;;
    --renders-dir)
      RENDERS_DIR="$2"
      shift 2
      ;;
    --sam-checkpoint)
      SAM_CHECKPOINT="$2"
      shift 2
      ;;
    --sam-points-per-side)
      SAM_POINTS_PER_SIDE="$2"
      shift 2
      ;;
    --sam-pred-iou-thresh)
      SAM_PRED_IOU_THRESH="$2"
      shift 2
      ;;
    --sam-stability-score-thresh)
      SAM_STABILITY_SCORE_THRESH="$2"
      shift 2
      ;;
    --sam-crop-n-layers)
      SAM_CROP_N_LAYERS="$2"
      shift 2
      ;;
    --sam-crop-n-points-downscale-factor)
      SAM_CROP_N_POINTS_DOWNSCALE_FACTOR="$2"
      shift 2
      ;;
    --sam-min-mask-region-area)
      SAM_MIN_MASK_REGION_AREA="$2"
      shift 2
      ;;
    --instances)
      INSTANCE_IDS_RAW="$2"
      shift 2
      ;;
    --instance-regex)
      INSTANCE_REGEX="$2"
      shift 2
      ;;
    --downsample)
      DOWNSAMPLE="$2"
      shift 2
      ;;
    --downsample-type)
      DOWNSAMPLE_TYPE="$2"
      shift 2
      ;;
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --num-sampled-rays)
      NUM_SAMPLED_RAYS="$2"
      shift 2
      ;;
    --scale-definition)
      SCALE_DEFINITION="$2"
      shift 2
      ;;
    --mask-scales-dir-name)
      MASK_SCALES_DIR_NAME="$2"
      shift 2
      ;;
    --supervision-mode)
      SUPERVISION_MODE="$2"
      shift 2
      ;;
    --mask-scale-target)
      MASK_SCALE_TARGET="$2"
      shift 2
      ;;
    --mask-scale-tolerance)
      MASK_SCALE_TOLERANCE="$2"
      shift 2
      ;;
    --boundary-negative-weight)
      BOUNDARY_NEGATIVE_WEIGHT="$2"
      shift 2
      ;;
    --boundary-negative-margin)
      BOUNDARY_NEGATIVE_MARGIN="$2"
      shift 2
      ;;
    --boundary-band-kernel)
      BOUNDARY_BAND_KERNEL="$2"
      shift 2
      ;;
    --min-valid-masks-per-view)
      MIN_VALID_MASKS_PER_VIEW="$2"
      shift 2
      ;;
    --feature-subdir)
      FEATURE_SUBDIR="$2"
      shift 2
      ;;
    --viz-interval)
      VIZ_INTERVAL="$2"
      shift 2
      ;;
    --viz-view-idx)
      VIZ_VIEW_IDX="$2"
      shift 2
      ;;
    --viz-mode)
      VIZ_MODE="$2"
      shift 2
      ;;
    --log-root)
      LOG_ROOT="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --skip-extract)
      SKIP_EXTRACT=1
      shift
      ;;
    --skip-scale)
      SKIP_SCALE=1
      shift
      ;;
    --skip-train)
      SKIP_TRAIN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

MODELS_DIR="${MODELS_DIR:-$DATA_ROOT/3dgs/fitted_gs/models}"
SCENES_DIR="${SCENES_DIR:-$DATA_ROOT/3dgs/fitted_gs/scenes}"
RENDERS_DIR="${RENDERS_DIR:-$DATA_ROOT/renders}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-$SCRIPT_DIR/third_party/segment-anything/sam_ckpt/sam_vit_h_4b8939.pth}"
LOG_ROOT="${LOG_ROOT:-$DATA_ROOT/logs/saga_batch_$(date -u +%Y%m%d_%H%M%S)}"
cd "$SCRIPT_DIR"

if [[ -z "$GPU_LIST" ]]; then
  echo "--gpus is required." >&2
  usage >&2
  exit 1
fi

if [[ ! -d "$MODELS_DIR" ]]; then
  echo "Models directory not found: $MODELS_DIR" >&2
  exit 1
fi

if [[ ! -d "$SCENES_DIR" ]]; then
  echo "Scenes directory not found: $SCENES_DIR" >&2
  exit 1
fi

if [[ ! -d "$RENDERS_DIR" ]]; then
  echo "Renders directory not found: $RENDERS_DIR" >&2
  exit 1
fi

if [[ "$SKIP_EXTRACT" != "1" ]] && [[ ! -f "$SAM_CHECKPOINT" ]]; then
  echo "SAM checkpoint not found: $SAM_CHECKPOINT" >&2
  exit 1
fi

mapfile -t GPU_IDS < <(parse_list "$GPU_LIST")
if [[ ${#GPU_IDS[@]} -eq 0 ]]; then
  echo "No valid GPU ids parsed from: $GPU_LIST" >&2
  exit 1
fi

declare -a INSTANCE_FILTER_SET=()
if [[ -n "$INSTANCE_IDS_RAW" ]]; then
  mapfile -t INSTANCE_FILTER_SET < <(parse_list "$INSTANCE_IDS_RAW")
fi

declare -a SELECTED_INSTANCES=()
declare -a ALL_MODEL_INSTANCES=()
mapfile -t ALL_MODEL_INSTANCES < <(find "$MODELS_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)

declare -A INSTANCE_FILTER_MAP=()
if [[ ${#INSTANCE_FILTER_SET[@]} -gt 0 ]]; then
  for instance_id in "${INSTANCE_FILTER_SET[@]}"; do
    INSTANCE_FILTER_MAP["$instance_id"]=1
  done
fi

for instance_id in "${ALL_MODEL_INSTANCES[@]}"; do
  if [[ ${#INSTANCE_FILTER_SET[@]} -gt 0 ]] && [[ -z "${INSTANCE_FILTER_MAP[$instance_id]:-}" ]]; then
    continue
  fi
  if [[ -n "$INSTANCE_REGEX" ]] && [[ ! "$instance_id" =~ $INSTANCE_REGEX ]]; then
    continue
  fi
  if [[ ! -d "$SCENES_DIR/$instance_id" ]]; then
    echo "Skipping $instance_id because scene metadata is missing." >&2
    continue
  fi
  if [[ ! -d "$RENDERS_DIR/$instance_id" ]]; then
    echo "Skipping $instance_id because render images are missing." >&2
    continue
  fi
  SELECTED_INSTANCES+=("$instance_id")
done

if [[ ${#SELECTED_INSTANCES[@]} -eq 0 ]]; then
  echo "No matching instances found." >&2
  exit 1
fi

mkdir -p "$LOG_ROOT"

echo "[$(timestamp)] Repo root: $SCRIPT_DIR"
echo "[$(timestamp)] Data root: $DATA_ROOT"
echo "[$(timestamp)] Models dir: $MODELS_DIR"
echo "[$(timestamp)] Scenes dir: $SCENES_DIR"
echo "[$(timestamp)] Renders dir: $RENDERS_DIR"
echo "[$(timestamp)] SAM checkpoint: $SAM_CHECKPOINT"
echo "[$(timestamp)] GPUs: $(join_by ',' "${GPU_IDS[@]}")"
echo "[$(timestamp)] Instances: $(join_by ' ' "${SELECTED_INSTANCES[@]}")"
echo "[$(timestamp)] Logs: $LOG_ROOT"
echo "[$(timestamp)] Extract settings: downsample=$DOWNSAMPLE, downsample_type=$DOWNSAMPLE_TYPE"
echo "[$(timestamp)] Scale settings: definition=$SCALE_DEFINITION, dir=$MASK_SCALES_DIR_NAME"
echo "[$(timestamp)] Train settings: iterations=$ITERATIONS, num_sampled_rays=$NUM_SAMPLED_RAYS, feature_subdir=$FEATURE_SUBDIR"

declare -a SLOT_PIDS=()
declare -a SLOT_GPUS=()
declare -a SLOT_INSTANCES=()
declare -a SLOT_LOGS=()
declare -a SUCCEEDED_INSTANCES=()
declare -a FAILED_INSTANCES=()

for idx in "${!GPU_IDS[@]}"; do
  SLOT_PIDS[$idx]=""
  SLOT_GPUS[$idx]="${GPU_IDS[$idx]}"
  SLOT_INSTANCES[$idx]=""
  SLOT_LOGS[$idx]=""
done

trap cleanup EXIT INT TERM

for instance_id in "${SELECTED_INSTANCES[@]}"; do
  slot_idx="$(find_free_slot)"
  gpu_id="${GPU_IDS[$slot_idx]}"
  instance_log_dir="$LOG_ROOT/$instance_id"
  worker_log="$instance_log_dir/worker.log"
  mkdir -p "$instance_log_dir"

  echo "[$(timestamp)] Launching instance $instance_id on GPU $gpu_id"
  (
    set -euo pipefail
    run_instance_pipeline "$gpu_id" "$instance_id"
  ) &

  SLOT_PIDS[$slot_idx]=$!
  SLOT_INSTANCES[$slot_idx]="$instance_id"
  SLOT_LOGS[$slot_idx]="$worker_log"
done

while true; do
  reap_finished_slots
  all_done=1
  for idx in "${!SLOT_PIDS[@]}"; do
    if [[ -n "${SLOT_PIDS[$idx]:-}" ]]; then
      all_done=0
      break
    fi
  done
  if [[ $all_done -eq 1 ]]; then
    break
  fi
  wait -n || true
done

echo "[$(timestamp)] Batch run complete."
echo "[$(timestamp)] Succeeded: ${#SUCCEEDED_INSTANCES[@]}"
if [[ ${#SUCCEEDED_INSTANCES[@]} -gt 0 ]]; then
  printf '  %s\n' "${SUCCEEDED_INSTANCES[@]}"
fi

echo "[$(timestamp)] Failed: ${#FAILED_INSTANCES[@]}"
if [[ ${#FAILED_INSTANCES[@]} -gt 0 ]]; then
  for failure in "${FAILED_INSTANCES[@]}"; do
    IFS='|' read -r instance_id gpu_id rc log_path <<<"$failure"
    echo "  instance=$instance_id gpu=$gpu_id exit_code=$rc log=$log_path"
  done
  exit 1
fi
