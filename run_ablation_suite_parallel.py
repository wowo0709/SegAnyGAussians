#!/usr/bin/env python3
import argparse
import csv
import json
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path


SCENES = [
    {
        "name": "bonsai",
        "group": "mipnerf360",
        "source_path": "/root/node1/data2/diyscene/mipnerf360/360_v2/bonsai",
        "base_model": "/root/node1/data2/diyscene/3dgs/fitted_gs/mipnerf360/360_v2/bonsai",
    },
    {
        "name": "garden",
        "group": "mipnerf360",
        "source_path": "/root/node1/data2/diyscene/mipnerf360/360_v2/garden",
        "base_model": "/root/node1/data2/diyscene/3dgs/fitted_gs/mipnerf360/360_v2/garden",
    },
    {
        "name": "bench",
        "group": "3d-ovs",
        "source_path": "/root/node1/data2/diyscene/3d-ovs/bench",
        "base_model": "/root/node1/data2/diyscene/3dgs/fitted_gs/3d-ovs/bench",
    },
    {
        "name": "covered_desk",
        "group": "3d-ovs",
        "source_path": "/root/node1/data2/diyscene/3d-ovs/covered_desk",
        "base_model": "/root/node1/data2/diyscene/3dgs/fitted_gs/3d-ovs/covered_desk",
    },
    {
        "name": "room",
        "group": "3d-ovs",
        "source_path": "/root/node1/data2/diyscene/3d-ovs/room",
        "base_model": "/root/node1/data2/diyscene/3dgs/fitted_gs/3d-ovs/room",
    },
    {
        "name": "sofa",
        "group": "3d-ovs",
        "source_path": "/root/node1/data2/diyscene/3d-ovs/sofa",
        "base_model": "/root/node1/data2/diyscene/3dgs/fitted_gs/3d-ovs/sofa",
    },
]


OOM_PATTERNS = (
    "cuda out of memory",
    "outofmemoryerror",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
    "runtimeerror: cublas",
    "\nkilled\n",
)


def log(message):
    timestamp = time.strftime("%H:%M:%S", time.gmtime())
    print(f"[{timestamp}] {message}", flush=True)


def env_int(name, default):
    return int(os.environ.get(name, default))


def env_float(name, default):
    return float(os.environ.get(name, default))


def env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).lower() in {"1", "true", "yes", "on"}


def env_list(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return list(default)
    return [item for item in raw.replace(",", " ").split() if item]


def default_settings():
    return {
        "sam_downsample": 2,
        "train_iterations": env_int("TRAIN_ITERATIONS", 10000),
        "num_sampled_rays": env_int("NUM_SAMPLED_RAYS", 768),
        "viz_interval": env_int("VIZ_INTERVAL", 0),
        "graph_laplacian_samples": env_int("GRAPH_LAPLACIAN_SAMPLES", 4096),
        "graph_laplacian_scales": env_int("GRAPH_LAPLACIAN_SCALES", 2),
        "graph_laplacian_k": env_int("GRAPH_LAPLACIAN_K", 16),
        "eval_render_downsample": env_int("EVAL_RENDER_DOWNSAMPLE", 8),
        "cluster_sample_size": env_int("CLUSTER_SAMPLE_SIZE", 12000),
        "save_viz_views": env_int("SAVE_VIZ_VIEWS", 1),
        "mask_scale_tolerance": env_float("MASK_SCALE_TOLERANCE", 0.10),
        "mask_min_area": env_int("MASK_MIN_AREA", 64),
        "mask_max_area_ratio": env_float("MASK_MAX_AREA_RATIO", 0.60),
        "mask_max_iou_overlap": env_float("MASK_MAX_IOU_OVERLAP", 0.90),
        "mask_boundary_erode_kernel": env_int("MASK_BOUNDARY_ERODE_KERNEL", 3),
        "min_valid_masks_per_view": env_int("MIN_VALID_MASKS_PER_VIEW", 2),
        "eval_scales": [float(x) for x in env_list("EVAL_SCALES", ["0.35", "0.50", "0.60"])],
        "graph_laplacian_weight": env_float("GRAPH_LAPLACIAN_WEIGHT", 0.05),
        "graph_spatial_weight": env_float("GRAPH_SPATIAL_WEIGHT", 1.0),
        "graph_sh0_color_weight": env_float("GRAPH_SH0_COLOR_WEIGHT", 1.0),
        "graph_sh0_color_sigma": env_float("GRAPH_SH0_COLOR_SIGMA", 0.25),
        "boundary_negative_weight": env_float("BOUNDARY_NEGATIVE_WEIGHT", 0.10),
        "boundary_negative_margin": env_float("BOUNDARY_NEGATIVE_MARGIN", 0.15),
        "boundary_band_kernel": env_int("BOUNDARY_BAND_KERNEL", 5),
        "cluster_spatial_weight": env_float("CLUSTER_SPATIAL_WEIGHT", 1.0),
        "cluster_sh0_color_weight": env_float("CLUSTER_SH0_COLOR_WEIGHT", 0.0),
        "cluster_sh0_color_sigma": env_float("CLUSTER_SH0_COLOR_SIGMA", 0.25),
        "cluster_graph_k": env_int("CLUSTER_GRAPH_K", 16),
        "cluster_max_clusters": env_int("CLUSTER_MAX_CLUSTERS", 24),
        "cluster_min_cluster_size": env_int("CLUSTER_MIN_CLUSTER_SIZE", 128),
        "cluster_cut_threshold": env_float("CLUSTER_CUT_THRESHOLD", 0.12),
        "hdbscan_min_cluster_size": env_int("HDBSCAN_MIN_CLUSTER_SIZE", 10),
        "hdbscan_epsilon": env_float("HDBSCAN_EPSILON", 0.01),
        "connectivity_k": env_int("CONNECTIVITY_K", 12),
        "eval_max_views": env_int("EVAL_MAX_VIEWS", 16),
        "eval_view_stride": env_int("EVAL_VIEW_STRIDE", 1),
        "summary_sort_key": os.environ.get("SUMMARY_SORT_KEY", "mean_connectivity_weighted"),
        "log_heartbeat_sec": env_int("LOG_HEARTBEAT_SEC", 0),
    }


def read_last_progress_snippet(path: Path, max_bytes: int = 16384, max_chars: int = 220):
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(size - max_bytes, 0))
            chunk = handle.read().decode("utf-8", errors="ignore")
    except FileNotFoundError:
        return None
    tokens = []
    for piece in chunk.replace("\r", "\n").split("\n"):
        text = piece.strip()
        if text:
            tokens.append(text)
    if not tokens:
        return None
    snippet = tokens[-1]
    if len(snippet) > max_chars:
        snippet = "..." + snippet[-(max_chars - 3):]
    return snippet


def pick_smoke_scenes(scenes):
    if not scenes:
        raise RuntimeError("No scenes remain after filtering.")
    preferred = ["garden", "sofa"]
    chosen = [name for name in preferred if any(scene["name"] == name for scene in scenes)]
    target_count = min(2, len(scenes))
    if len(chosen) >= target_count:
        return chosen[:target_count]
    for scene in scenes:
        if scene["name"] not in chosen:
            chosen.append(scene["name"])
        if len(chosen) >= target_count:
            break
    return chosen


def build_stage_templates(scenes):
    smoke_scene_names = pick_smoke_scenes(scenes)
    return [
        {
            "name": "stage_0_smoke",
            "display_name": "Stage 0: scheduler smoke",
            "description": "Scheduler, logging, fallback, and README smoke test on two scenes with lightweight settings.",
            "variants": ["multiscale", "single_scale_band_035", "single_scale_band_050", "single_scale_band_060"],
            "scene_names": smoke_scene_names,
            "enabled": True,
        },
        {
            "name": "stage_a",
            "display_name": "Stage A: supervision ablation",
            "description": "Compare multiscale supervision against single-scale part-band supervision.",
            "variants": ["multiscale", "single_scale_band_035", "single_scale_band_050", "single_scale_band_060"],
            "scene_names": [scene["name"] for scene in scenes],
            "enabled": True,
        },
        {
            "name": "stage_s",
            "display_name": "Stage S: small weight sanity check",
            "description": "Limited 3-point weight checks on two scenes to verify the defaults are not pathological.",
            "variants": [
                "sanity_graphlap_0p02",
                "sanity_graphlap_0p05",
                "sanity_graphlap_0p10",
                "sanity_boundary_0p05",
                "sanity_boundary_0p10",
                "sanity_boundary_0p20",
                "sanity_sh0color_0p5",
                "sanity_sh0color_1p0",
                "sanity_sh0color_2p0",
            ],
            "scene_names": smoke_scene_names,
            "enabled": True,
        },
        {
            "name": "stage_b",
            "display_name": "Stage B: training objective ablation",
            "description": "Compare which training losses materially improve clean part clusters.",
            "variants": ["baseline", "laplacian", "laplacian_boundary", "laplacian_boundary_color"],
            "scene_names": [scene["name"] for scene in scenes],
            "enabled": True,
        },
        {
            "name": "stage_c",
            "display_name": "Stage C: clustering cue ablation",
            "description": "Eval-only comparison of HDBSCAN, HDBSCANRefined, and NormalizedCut with xyz and SH0 cue combinations.",
            "variants": ["hdbscan", "hdbscan_refined_xyz", "hdbscan_refined_xyz_sh0", "ncut_xyz", "ncut_xyz_sh0", "ncut_sh0", "ncut_feature"],
            "scene_names": [scene["name"] for scene in scenes],
            "enabled": True,
        },
    ]


class AblationSuiteRunner:
    def __init__(self, ablation_root: Path, repo_root: Path, gpu_ids, scenes=None, heartbeat_sec=None):
        self.ablation_root = ablation_root
        self.repo_root = repo_root
        self.gpu_ids = list(gpu_ids)
        self.scenes = list(scenes if scenes is not None else SCENES)
        self.defaults = default_settings()
        if heartbeat_sec is not None:
            self.defaults["log_heartbeat_sec"] = max(0, int(heartbeat_sec))
        self.status_dir = self.ablation_root / "status"
        self.logs_dir = self.ablation_root / "logs"
        self.jobs_dir = self.ablation_root / "jobs"
        self.stages_dir = self.ablation_root / "stages"
        self.status_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.stages_dir.mkdir(parents=True, exist_ok=True)
        self.status_lock = threading.Lock()
        self.manifest_lock = threading.Lock()
        self.jobs_manifest_path = self.ablation_root / "jobs_manifest.csv"
        self.manifest_fields = [
            "job_id",
            "stage",
            "scene",
            "source_path",
            "base_model",
            "job_root",
            "variant_list",
            "skip_train",
            "skip_eval",
        ]
        if not self.jobs_manifest_path.exists():
            with open(self.jobs_manifest_path, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.manifest_fields)
                writer.writeheader()

    def write_suite_config(self, stage_templates):
        config = {
            "title": "SAGA 3DGS Part-Cluster Ablation (No Mesh / No MILo)",
            "sam_downsample": self.defaults["sam_downsample"],
            "scenes": self.scenes,
            "defaults": self.defaults,
            "stages": stage_templates,
            "next_step": "After this round, fit MILo scenes, add mesh-aware training-time losses, and rerun the study with mesh and model-family comparisons.",
        }
        path = self.ablation_root / "suite_config.json"
        path.write_text(json.dumps(config, indent=2) + "\n")

    def append_status(self, name, record):
        record = dict(record)
        record.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        target = self.status_dir / f"{name}.jsonl"
        with self.status_lock:
            with open(target, "a") as handle:
                handle.write(json.dumps(record) + "\n")

    def append_manifest_rows(self, rows):
        with self.manifest_lock:
            with open(self.jobs_manifest_path, "a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.manifest_fields)
                for row in rows:
                    writer.writerow(row)

    def build_common_env(self):
        env = os.environ.copy()
        env.update(
            {
                "TRAIN_ITERATIONS": str(self.defaults["train_iterations"]),
                "FEATURE_ITERATION_OVERRIDE": str(self.defaults["train_iterations"]),
                "NUM_SAMPLED_RAYS": str(self.defaults["num_sampled_rays"]),
                "VIZ_INTERVAL": str(self.defaults["viz_interval"]),
                "VIZ_VIEW_IDX": os.environ.get("VIZ_VIEW_IDX", "0"),
                "VIZ_MODE": os.environ.get("VIZ_MODE", "embedding"),
                "VIZ_MAX_SAMPLES": os.environ.get("VIZ_MAX_SAMPLES", "5000"),
                "EVAL_SCALES": " ".join(f"{scale:.2f}" for scale in self.defaults["eval_scales"]),
                "EVAL_CLUSTER_METHODS": os.environ.get("EVAL_CLUSTER_METHODS", "NormalizedCut"),
                "EVAL_MAX_VIEWS": str(self.defaults["eval_max_views"]),
                "EVAL_VIEW_STRIDE": str(self.defaults["eval_view_stride"]),
                "EVAL_RENDER_DOWNSAMPLE": str(self.defaults["eval_render_downsample"]),
                "SUPERVISION_MODE": os.environ.get("SUPERVISION_MODE", "multiscale"),
                "MASK_SCALE_TARGET": os.environ.get("MASK_SCALE_TARGET", "0.50"),
                "MASK_SCALE_TOLERANCE": str(self.defaults["mask_scale_tolerance"]),
                "MASK_MIN_AREA": str(self.defaults["mask_min_area"]),
                "MASK_MAX_AREA_RATIO": str(self.defaults["mask_max_area_ratio"]),
                "MASK_MAX_IOU_OVERLAP": str(self.defaults["mask_max_iou_overlap"]),
                "MASK_BOUNDARY_ERODE_KERNEL": str(self.defaults["mask_boundary_erode_kernel"]),
                "MIN_VALID_MASKS_PER_VIEW": str(self.defaults["min_valid_masks_per_view"]),
                "GRAPH_LAPLACIAN_WEIGHT": str(self.defaults["graph_laplacian_weight"]),
                "GRAPH_LAPLACIAN_SAMPLES": str(self.defaults["graph_laplacian_samples"]),
                "GRAPH_LAPLACIAN_SCALES": str(self.defaults["graph_laplacian_scales"]),
                "GRAPH_LAPLACIAN_K": str(self.defaults["graph_laplacian_k"]),
                "GRAPH_SPATIAL_WEIGHT": str(self.defaults["graph_spatial_weight"]),
                "GRAPH_SH0_COLOR_WEIGHT": str(self.defaults["graph_sh0_color_weight"]),
                "GRAPH_SH0_COLOR_SIGMA": str(self.defaults["graph_sh0_color_sigma"]),
                "BOUNDARY_NEGATIVE_WEIGHT": str(self.defaults["boundary_negative_weight"]),
                "BOUNDARY_NEGATIVE_MARGIN": str(self.defaults["boundary_negative_margin"]),
                "BOUNDARY_BAND_KERNEL": str(self.defaults["boundary_band_kernel"]),
                "CLUSTER_SAMPLE_SIZE": str(self.defaults["cluster_sample_size"]),
                "CLUSTER_GRAPH_K": str(self.defaults["cluster_graph_k"]),
                "CLUSTER_MAX_CLUSTERS": str(self.defaults["cluster_max_clusters"]),
                "CLUSTER_MIN_CLUSTER_SIZE": str(self.defaults["cluster_min_cluster_size"]),
                "CLUSTER_CUT_THRESHOLD": str(self.defaults["cluster_cut_threshold"]),
                "CLUSTER_SPATIAL_WEIGHT": str(self.defaults["cluster_spatial_weight"]),
                "CLUSTER_SH0_COLOR_WEIGHT": str(self.defaults["cluster_sh0_color_weight"]),
                "CLUSTER_SH0_COLOR_SIGMA": str(self.defaults["cluster_sh0_color_sigma"]),
                "HDBSCAN_MIN_CLUSTER_SIZE": str(self.defaults["hdbscan_min_cluster_size"]),
                "HDBSCAN_EPSILON": str(self.defaults["hdbscan_epsilon"]),
                "CONNECTIVITY_K": str(self.defaults["connectivity_k"]),
                "SAVE_VIZ_VIEWS": str(self.defaults["save_viz_views"]),
                "SUMMARY_SORT_KEY": self.defaults["summary_sort_key"],
            }
        )
        return env

    def query_gpu_status(self):
        command = [
            "nvidia-smi",
            "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        output = subprocess.check_output(command, text=True)
        status = {}
        for line in output.strip().splitlines():
            index_str, free_mem_str, util_str = [part.strip() for part in line.split(",")]
            status[int(index_str)] = {
                "memory_free_mb": int(free_mem_str),
                "utilization": int(util_str),
            }
        return status

    def wait_for_gpu(self, gpu_id, min_free_mb, max_util):
        logged_wait = False
        while True:
            try:
                status = self.query_gpu_status()[gpu_id]
            except Exception:
                if not logged_wait:
                    log(f"GPU {gpu_id}: failed to query nvidia-smi status, retrying...")
                    logged_wait = True
                time.sleep(10)
                continue
            if status["memory_free_mb"] >= min_free_mb and status["utilization"] <= max_util:
                if logged_wait:
                    log(
                        f"GPU {gpu_id}: ready with {status['memory_free_mb']} MiB free, "
                        f"util {status['utilization']}%"
                    )
                return status
            if not logged_wait:
                log(
                    f"GPU {gpu_id}: waiting for >= {min_free_mb} MiB free and <= {max_util}% util "
                    f"(currently {status['memory_free_mb']} MiB free, {status['utilization']}% util)"
                )
                logged_wait = True
            time.sleep(20)

    def fallback_overrides(self, level):
        overrides = {}
        if level >= 1:
            overrides.update({"NUM_SAMPLED_RAYS": "512", "GRAPH_LAPLACIAN_SAMPLES": "2048"})
        if level >= 2:
            overrides.update({"CLUSTER_SAMPLE_SIZE": "8000", "EVAL_RENDER_DOWNSAMPLE": "12"})
        if level >= 3:
            overrides.update({"SAVE_VIZ_VIEWS": "0"})
        return overrides

    def is_oom_log(self, log_path: Path):
        try:
            text = log_path.read_text(errors="ignore").lower()
        except FileNotFoundError:
            return False
        return any(pattern in text for pattern in OOM_PATTERNS)

    def write_job_script(self, job):
        script_path = self.jobs_dir / f"{job['job_id']}.sh"
        exports = []
        for key, value in sorted(job["env"].items()):
            escaped = str(value).replace('"', '\\"')
            exports.append(f'export {key}="{escaped}"')
        content = "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f'cd "{self.repo_root}"',
                *exports,
                f'bash "{self.repo_root / "run_ablation_study.sh"}" "{job["source_path"]}" "{job["base_model"]}" "{job["job_root"]}"',
                "",
            ]
        )
        script_path.write_text(content)
        script_path.chmod(0o755)
        job["script_path"] = str(script_path)

    def build_job(self, stage_name, scene, stage_dir, variant_list, base_model, env_overrides=None, skip_train=False, skip_eval=False):
        env = self.build_common_env()
        if env_overrides:
            env.update({key: str(value) for key, value in env_overrides.items()})
        env["VARIANT_LIST"] = ",".join(variant_list)
        env["SKIP_TRAIN"] = "1" if skip_train else "0"
        env["SKIP_EVAL"] = "1" if skip_eval else "0"
        job_root = stage_dir / scene["name"]
        env["RUN_CONTEXT_JSON"] = str(job_root / "run_context.json")
        job_id = f"{stage_name}__{scene['name']}"
        job = {
            "job_id": job_id,
            "stage": stage_name,
            "scene": scene["name"],
            "source_path": scene["source_path"],
            "base_model": base_model,
            "job_root": str(job_root),
            "variant_list": list(variant_list),
            "skip_train": skip_train,
            "skip_eval": skip_eval,
            "env": env,
        }
        self.write_job_script(job)
        return job

    def run_stage_jobs(self, stage_name, display_name, jobs, gpu_ids):
        stage_dir = self.stages_dir / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        log(
            f"{display_name} started: {len(jobs)} job(s), "
            f"GPUs {', '.join(str(gpu) for gpu in gpu_ids)}"
        )
        stage_manifest_path = stage_dir / "jobs_manifest.csv"
        rows = []
        for job in jobs:
            rows.append(
                {
                    "job_id": job["job_id"],
                    "stage": stage_name,
                    "scene": job["scene"],
                    "source_path": job["source_path"],
                    "base_model": job["base_model"],
                    "job_root": job["job_root"],
                    "variant_list": ",".join(job["variant_list"]),
                    "skip_train": int(job["skip_train"]),
                    "skip_eval": int(job["skip_eval"]),
                }
            )
            self.append_status("pending", rows[-1])

        self.append_manifest_rows(rows)
        with open(stage_manifest_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.manifest_fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        job_queue = queue.Queue()
        for job in jobs:
            job_queue.put(job)

        failures = []
        failure_lock = threading.Lock()

        def worker(gpu_id):
            log(f"{display_name}: worker attached to GPU {gpu_id}")
            while True:
                try:
                    job = job_queue.get_nowait()
                except queue.Empty:
                    log(f"{display_name}: GPU {gpu_id} worker finished all assigned jobs")
                    return
                status = self.wait_for_gpu(gpu_id, min_free_mb=18432, max_util=30)
                success = self.run_job(job, gpu_id, status)
                if not success:
                    with failure_lock:
                        failures.append(job["job_id"])
                job_queue.task_done()

        threads = []
        for gpu_id in gpu_ids:
            thread = threading.Thread(target=worker, args=(gpu_id,), daemon=True)
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

        aggregate_csv = stage_dir / "aggregate.csv"
        aggregate_md = stage_dir / "aggregate.md"
        winner_json = stage_dir / "winner.json"
        json_files = sorted(stage_dir.glob("*/ablation_eval/*.json"))
        if json_files:
            command = [
                "python",
                str(self.repo_root / "aggregate_ablation_stage.py"),
                *[str(path) for path in json_files],
                "--output_csv",
                str(aggregate_csv),
                "--output_md",
                str(aggregate_md),
                "--winner_json",
                str(winner_json),
            ]
            subprocess.run(command, check=True, cwd=self.repo_root)
            log(f"{display_name}: aggregated {len(json_files)} eval JSON file(s)")

        subprocess.run(
            ["python", str(self.repo_root / "build_ablation_readme.py"), "--ablation_root", str(self.ablation_root)],
            check=True,
            cwd=self.repo_root,
        )
        log(f"{display_name}: README updated at {self.ablation_root / 'README.md'}")

        if failures:
            raise RuntimeError(f"{display_name} failed for jobs: {', '.join(sorted(failures))}")
        log(f"{display_name} completed successfully")

    def run_job(self, job, gpu_id, gpu_status):
        base_env = job["env"].copy()
        log_dir = self.logs_dir / job["stage"] / job["scene"]
        log_dir.mkdir(parents=True, exist_ok=True)
        job_root = Path(job["job_root"])

        for fallback_level in range(4):
            env = base_env.copy()
            env.update(self.fallback_overrides(fallback_level))
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            log_path = log_dir / f"{job['job_id']}__attempt{fallback_level}.log"
            log(
                f"{job['job_id']}: launching on GPU {gpu_id} "
                f"(fallback {fallback_level}, log: {log_path})"
            )
            shutil.rmtree(job_root, ignore_errors=True)
            actual_config = {
                key: env.get(key)
                for key in (
                    "TRAIN_ITERATIONS",
                    "FEATURE_ITERATION_OVERRIDE",
                    "NUM_SAMPLED_RAYS",
                    "GRAPH_LAPLACIAN_SAMPLES",
                    "CLUSTER_SAMPLE_SIZE",
                    "EVAL_RENDER_DOWNSAMPLE",
                    "SAVE_VIZ_VIEWS",
                    "SUPERVISION_MODE",
                    "MASK_SCALE_TARGET",
                    "VARIANT_LIST",
                )
            }
            self.append_status(
                "running",
                {
                    "job_id": job["job_id"],
                    "stage": job["stage"],
                    "scene": job["scene"],
                    "gpu_id": gpu_id,
                    "gpu_memory_free_mb": gpu_status["memory_free_mb"],
                    "gpu_utilization": gpu_status["utilization"],
                    "fallback_level": fallback_level,
                    "log_path": str(log_path),
                    "actual_config": actual_config,
                },
            )

            start_time = time.time()
            with open(log_path, "w") as handle:
                process = subprocess.Popen(
                    ["bash", str(self.repo_root / "run_ablation_study.sh"), job["source_path"], job["base_model"], job["job_root"]],
                    cwd=self.repo_root,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
                last_report_time = start_time
                last_reported_snippet = None
                heartbeat_sec = max(5, int(self.defaults["log_heartbeat_sec"])) if int(self.defaults["log_heartbeat_sec"]) > 0 else 0
                while True:
                    result_code = process.poll()
                    now = time.time()
                    if result_code is not None:
                        break
                    if heartbeat_sec > 0 and now - last_report_time >= heartbeat_sec:
                        snippet = read_last_progress_snippet(log_path)
                        elapsed_min = (now - start_time) / 60.0
                        if snippet and snippet != last_reported_snippet:
                            log(f"{job['job_id']}: {snippet} [elapsed {elapsed_min:.1f} min]")
                            last_reported_snippet = snippet
                        else:
                            log(f"{job['job_id']}: still running on GPU {gpu_id} [elapsed {elapsed_min:.1f} min]")
                        last_report_time = now
                    time.sleep(2)
                result = subprocess.CompletedProcess(process.args, result_code)
            duration_sec = time.time() - start_time

            if result.returncode == 0:
                log(
                    f"{job['job_id']}: finished successfully in {duration_sec / 60.0:.1f} min "
                    f"(fallback {fallback_level})"
                )
                self.append_status(
                    "done",
                    {
                        "job_id": job["job_id"],
                        "stage": job["stage"],
                        "scene": job["scene"],
                        "status": "done",
                        "gpu_id": gpu_id,
                        "oom_fallback_level": fallback_level,
                        "detected_oom": fallback_level > 0,
                        "log_path": str(log_path),
                        "duration_sec": duration_sec,
                        "actual_config": actual_config,
                        "summary_csv": str(job_root / "ablation_summary.csv"),
                        "summary_md": str(job_root / "ablation_summary.md"),
                    },
                )
                return True

            detected_oom = self.is_oom_log(log_path)
            retry_planned = detected_oom and fallback_level < 3
            if retry_planned:
                log(
                    f"{job['job_id']}: detected OOM on fallback {fallback_level}, "
                    f"retrying with fallback {fallback_level + 1}"
                )
            else:
                log(
                    f"{job['job_id']}: failed in {duration_sec / 60.0:.1f} min "
                    f"(fallback {fallback_level}, oom={detected_oom})"
                )
            self.append_status(
                "failed",
                {
                    "job_id": job["job_id"],
                    "stage": job["stage"],
                    "scene": job["scene"],
                    "status": "retrying" if retry_planned else "failed",
                    "gpu_id": gpu_id,
                    "oom_fallback_level": fallback_level,
                    "detected_oom": detected_oom,
                    "retry_planned": retry_planned,
                    "return_code": result.returncode,
                    "log_path": str(log_path),
                    "duration_sec": duration_sec,
                    "actual_config": actual_config,
                },
            )
            if not retry_planned:
                return False
        return False

    def scene_lookup(self, scene_name):
        for scene in self.scenes:
            if scene["name"] == scene_name:
                return scene
        raise KeyError(scene_name)

    def aggregate_winner(self, stage_name):
        winner_path = self.stages_dir / stage_name / "winner.json"
        if not winner_path.exists():
            return None
        payload = json.loads(winner_path.read_text())
        winner = payload.get("winner")
        return None if not winner else winner.get("label")

    def stage_env_from_supervision(self, label):
        if label is None or label == "multiscale":
            return {"SUPERVISION_MODE": "multiscale", "MASK_SCALE_TARGET": "0.50"}
        if label == "single_scale_band_035":
            return {"SUPERVISION_MODE": "single_scale_band", "MASK_SCALE_TARGET": "0.35"}
        if label == "single_scale_band_050":
            return {"SUPERVISION_MODE": "single_scale_band", "MASK_SCALE_TARGET": "0.50"}
        if label == "single_scale_band_060":
            return {"SUPERVISION_MODE": "single_scale_band", "MASK_SCALE_TARGET": "0.60"}
        raise RuntimeError(f"Unsupported supervision label: {label}")

    def run(self, run_stage0_smoke=True, selected_stages=None):
        selected_stages = None if selected_stages is None else set(selected_stages)
        stage_templates = build_stage_templates(self.scenes)
        for stage in stage_templates:
            enabled = stage["enabled"]
            if stage["name"] == "stage_0_smoke" and not run_stage0_smoke:
                enabled = False
            if selected_stages is not None and stage["name"] not in selected_stages:
                enabled = False
            stage["enabled"] = enabled
        self.write_suite_config(stage_templates)

        stage_0_template = next((stage for stage in stage_templates if stage["name"] == "stage_0_smoke"), None)
        if stage_0_template is not None and stage_0_template["enabled"]:
            stage_dir = self.stages_dir / "stage_0_smoke"
            jobs = []
            for scene_name in stage_0_template["scene_names"]:
                scene = self.scene_lookup(scene_name)
                jobs.append(
                    self.build_job(
                        "stage_0_smoke",
                        scene,
                        stage_dir,
                        ["multiscale", "single_scale_band_035", "single_scale_band_050", "single_scale_band_060"],
                        base_model=scene["base_model"],
                    env_overrides={
                        "TRAIN_ITERATIONS": "1",
                        "FEATURE_ITERATION_OVERRIDE": "1",
                        "EVAL_MAX_VIEWS": "1",
                        "SAVE_VIZ_VIEWS": "0",
                        "EVAL_CLUSTER_METHODS": "NormalizedCut",
                        "CLUSTER_SPATIAL_WEIGHT": "1.0",
                        "CLUSTER_SH0_COLOR_WEIGHT": "0.0",
                    },
                )
            )
            self.run_stage_jobs("stage_0_smoke", "Stage 0", jobs, self.gpu_ids[:2])

        stage_a_winner = None
        if any(stage["name"] == "stage_a" and stage["enabled"] for stage in stage_templates):
            stage_a_dir = self.stages_dir / "stage_a"
            stage_a_jobs = []
            for scene in self.scenes:
                stage_a_jobs.append(
                    self.build_job(
                        "stage_a",
                        scene,
                        stage_a_dir,
                        ["multiscale", "single_scale_band_035", "single_scale_band_050", "single_scale_band_060"],
                        base_model=scene["base_model"],
                        env_overrides={
                            "GRAPH_LAPLACIAN_WEIGHT": "0.0",
                            "BOUNDARY_NEGATIVE_WEIGHT": "0.0",
                            "GRAPH_SH0_COLOR_WEIGHT": "0.0",
                            "CLUSTER_SPATIAL_WEIGHT": "1.0",
                            "CLUSTER_SH0_COLOR_WEIGHT": "0.0",
                            "EVAL_CLUSTER_METHODS": "NormalizedCut",
                        },
                    )
                )
            self.run_stage_jobs("stage_a", "Stage A", stage_a_jobs, self.gpu_ids)
            stage_a_winner = self.aggregate_winner("stage_a")
        if stage_a_winner is None:
            stage_a_winner = os.environ.get("STAGE_A_WINNER_LABEL")
        stage_a_env = self.stage_env_from_supervision(stage_a_winner)

        stage_s_template = next((stage for stage in stage_templates if stage["name"] == "stage_s"), None)
        if stage_s_template is not None and stage_s_template["enabled"]:
            stage_s_dir = self.stages_dir / "stage_s"
            stage_s_jobs = []
            for scene_name in stage_s_template["scene_names"]:
                scene = self.scene_lookup(scene_name)
                env_overrides = dict(stage_a_env)
                env_overrides.update(
                    {
                        "EVAL_CLUSTER_METHODS": "NormalizedCut",
                        "CLUSTER_SPATIAL_WEIGHT": "1.0",
                        "CLUSTER_SH0_COLOR_WEIGHT": "0.0",
                    }
                )
                stage_s_jobs.append(
                    self.build_job(
                        "stage_s",
                        scene,
                        stage_s_dir,
                        [
                            "sanity_graphlap_0p02",
                            "sanity_graphlap_0p05",
                            "sanity_graphlap_0p10",
                            "sanity_boundary_0p05",
                            "sanity_boundary_0p10",
                            "sanity_boundary_0p20",
                            "sanity_sh0color_0p5",
                            "sanity_sh0color_1p0",
                            "sanity_sh0color_2p0",
                        ],
                        base_model=scene["base_model"],
                        env_overrides=env_overrides,
                    )
                )
            self.run_stage_jobs("stage_s", "Stage S", stage_s_jobs, self.gpu_ids[:2])

        stage_b_winner = None
        if any(stage["name"] == "stage_b" and stage["enabled"] for stage in stage_templates):
            stage_b_dir = self.stages_dir / "stage_b"
            stage_b_jobs = []
            for scene in self.scenes:
                env_overrides = dict(stage_a_env)
                env_overrides.update(
                    {
                        "EVAL_CLUSTER_METHODS": "NormalizedCut",
                        "CLUSTER_SPATIAL_WEIGHT": "1.0",
                        "CLUSTER_SH0_COLOR_WEIGHT": "0.0",
                    }
                )
                stage_b_jobs.append(
                    self.build_job(
                        "stage_b",
                        scene,
                        stage_b_dir,
                        ["baseline", "laplacian", "laplacian_boundary", "laplacian_boundary_color"],
                        base_model=scene["base_model"],
                        env_overrides=env_overrides,
                    )
                )
            self.run_stage_jobs("stage_b", "Stage B", stage_b_jobs, self.gpu_ids)
            stage_b_winner = self.aggregate_winner("stage_b")
        stage_c_enabled = any(stage["name"] == "stage_c" and stage["enabled"] for stage in stage_templates)
        if stage_b_winner is None:
            stage_b_winner = os.environ.get("STAGE_B_WINNER_LABEL")
        if stage_c_enabled and stage_b_winner is None:
            raise RuntimeError("Stage B did not produce a winner.")

        if stage_c_enabled:
            stage_b_dir = self.stages_dir / "stage_b"
            stage_c_dir = self.stages_dir / "stage_c"
            stage_c_jobs = []
            for scene in self.scenes:
                env_overrides = dict(stage_a_env)
                env_overrides.update(
                    {
                        "EVAL_CLUSTER_METHODS": "NormalizedCut",
                        "CLUSTER_SH0_COLOR_WEIGHT": "1.0",
                        "CLUSTER_SH0_COLOR_SIGMA": str(self.defaults["cluster_sh0_color_sigma"]),
                    }
                )
                base_model = str(stage_b_dir / scene["name"] / stage_b_winner)
                stage_c_jobs.append(
                    self.build_job(
                        "stage_c",
                        scene,
                        stage_c_dir,
                        ["hdbscan", "hdbscan_refined_xyz", "hdbscan_refined_xyz_sh0", "ncut_xyz", "ncut_xyz_sh0", "ncut_sh0", "ncut_feature"],
                        base_model=base_model,
                        env_overrides=env_overrides,
                        skip_train=True,
                        skip_eval=False,
                    )
                )
            self.run_stage_jobs("stage_c", "Stage C", stage_c_jobs, self.gpu_ids)

        subprocess.run(
            ["python", str(self.repo_root / "build_ablation_readme.py"), "--ablation_root", str(self.ablation_root)],
            check=True,
            cwd=self.repo_root,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Run the SAGA 3DGS ablation suite in parallel across multiple GPUs.")
    parser.add_argument("ablation_root", type=str, help="Output root for the entire ablation suite.")
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=env_list("GPU_IDS", [str(i) for i in range(8)]))
    parser.add_argument("--repo_root", type=str, default=str(Path(__file__).resolve().parent))
    parser.add_argument("--heartbeat_sec", type=int, default=None, help="Optional console heartbeat interval in seconds for per-job progress. Disabled by default.")
    parser.add_argument("--skip_stage0_smoke", action="store_true")
    parser.add_argument("--stages", nargs="+", default=None, help="Optional subset of stages to run, e.g. stage_0_smoke stage_a")
    parser.add_argument("--scene_names", nargs="+", default=None, help="Optional explicit scene names, e.g. bench covered_desk room sofa")
    parser.add_argument("--scene_groups", nargs="+", default=None, help="Optional scene groups, e.g. 3d-ovs mipnerf360")
    return parser.parse_args()


def resolve_scenes(scene_names=None, scene_groups=None):
    scenes = list(SCENES)
    if scene_groups:
        wanted_groups = {group.strip() for group in scene_groups if group.strip()}
        scenes = [scene for scene in scenes if scene["group"] in wanted_groups]
    if scene_names:
        wanted_names = {name.strip() for name in scene_names if name.strip()}
        scenes = [scene for scene in scenes if scene["name"] in wanted_names]
    if not scenes:
        raise RuntimeError("Scene filtering removed every scene. Check --scene_names/--scene_groups.")
    return scenes


def main():
    args = parse_args()
    scenes = resolve_scenes(scene_names=args.scene_names, scene_groups=args.scene_groups)
    log(f"Ablation root: {args.ablation_root}")
    log(f"Repo root: {args.repo_root}")
    log(f"Selected GPUs: {', '.join(str(gpu) for gpu in args.gpu_ids)}")
    log(f"Selected scenes: {', '.join(scene['name'] for scene in scenes)}")
    if args.scene_groups:
        log(f"Scene groups filter: {', '.join(args.scene_groups)}")
    if args.stages:
        log(f"Selected stages: {', '.join(args.stages)}")
    if args.skip_stage0_smoke:
        log("Stage 0 smoke is disabled")
    if args.heartbeat_sec is not None and args.heartbeat_sec > 0:
        log(f"Per-job heartbeat enabled every {args.heartbeat_sec} sec")
    runner = AblationSuiteRunner(
        Path(args.ablation_root),
        Path(args.repo_root),
        [int(gpu) for gpu in args.gpu_ids],
        scenes=scenes,
        heartbeat_sec=args.heartbeat_sec,
    )
    runner.run(run_stage0_smoke=not args.skip_stage0_smoke, selected_stages=args.stages)


if __name__ == "__main__":
    main()
