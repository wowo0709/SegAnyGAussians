#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter


DEFAULT_MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
DEFAULT_PROMPT = (
    "clean realistic studio product photo of a single toy object, "
    "natural detailed texture, sharp focus, high quality materials, "
    "white background"
)
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, floating fragments, disconnected parts, duplicate parts, "
    "deformed geometry, extra objects, cluttered background, noisy texture, "
    "oversaturated, watermark, text, logo"
)
RESAMPLE_LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
RESAMPLE_NEAREST = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST


@dataclass(frozen=True)
class ViewSpec:
    view_id: str
    rgb_path: Path
    mask_path: Optional[Path]
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refine SegAnyGaussians render_job views with a vanilla SDXL inpaint model. "
            "Each view is processed independently."
        )
    )
    parser.add_argument("--job-json", type=Path, required=True, help="Path to render_job_150views.json.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <job_dir>/sdxl_<num_views>views after view selection.",
    )
    parser.add_argument(
        "--view-ids",
        nargs="*",
        default=None,
        help="Optional subset of view ids such as 000 040 149. Defaults to all views in the job.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of selected views, applied after --view-ids filtering.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model id or local path for the SDXL inpaint pipeline.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not attempt to download the model. Only use locally cached weights.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Execution device. Defaults to cuda when available.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "float32"),
        default="auto",
        help="Pipeline dtype. Defaults to float16 on cuda and float32 on cpu.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Positive prompt used for all views.",
    )
    parser.add_argument(
        "--negative-prompt",
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt used for all views.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.30,
        help="Inpaint denoising strength. Lower values preserve more of the original render.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=30,
        help="Number of denoising steps per view.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=1024,
        help="Square working resolution passed to SDXL.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed used directly or offset per view depending on --seed-mode.",
    )
    parser.add_argument(
        "--seed-mode",
        choices=("per_view", "shared"),
        default="per_view",
        help="Use a different seed per view or reuse the same seed for every view.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=int,
        default=127,
        help="Threshold used to binarize the mask image.",
    )
    parser.add_argument(
        "--mask-polarity",
        choices=("object", "background"),
        default="object",
        help="Interpret white pixels as the object to refine or as background to preserve.",
    )
    parser.add_argument(
        "--mask-dilate",
        type=int,
        default=8,
        help="Dilate the binary mask by this many pixels before inpainting.",
    )
    parser.add_argument(
        "--mask-blur",
        type=float,
        default=1.5,
        help="Gaussian blur radius applied to the composite alpha mask.",
    )
    parser.add_argument(
        "--crop-padding",
        type=float,
        default=0.20,
        help="Extra padding ratio added around the object mask when --crop-to-mask is enabled.",
    )
    parser.add_argument(
        "--min-crop-size",
        type=int,
        default=256,
        help="Minimum crop side length in source pixels.",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Ignore per-view masks and inpaint the full image.",
    )
    parser.add_argument(
        "--no-crop-to-mask",
        action="store_true",
        help="Disable object-centric square cropping before sending the view to SDXL.",
    )
    parser.add_argument(
        "--no-preserve-background",
        action="store_true",
        help="Use the full SDXL output instead of compositing only the masked region back onto the original image.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip views whose output png already exists in the output directory.",
    )
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save per-view processed masks and crop inputs under <output_dir>/debug.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate inputs and write metadata without loading the SDXL pipeline.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def dump_json(path: Path, payload: Dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2))


def parse_view_specs(job_json: Path) -> List[ViewSpec]:
    payload = load_json(job_json)
    views = payload.get("views")
    if not isinstance(views, list):
        raise TypeError("Expected render job JSON to contain a 'views' list.")

    parsed: List[ViewSpec] = []
    for entry in views:
        view_id = str(entry["view_id"])
        rgb_path = Path(entry["output_rgb"])
        mask_path = Path(entry["output_mask"]) if "output_mask" in entry else None
        width = int(entry["width"])
        height = int(entry["height"])
        if not rgb_path.exists():
            raise FileNotFoundError("Input RGB render was not found: %s" % rgb_path)
        if mask_path is not None and not mask_path.exists():
            mask_path = None
        parsed.append(
            ViewSpec(
                view_id=view_id,
                rgb_path=rgb_path,
                mask_path=mask_path,
                width=width,
                height=height,
            )
        )
    return parsed


def select_views(views: Sequence[ViewSpec], requested_ids: Optional[Sequence[str]], limit: Optional[int]) -> List[ViewSpec]:
    by_id = {view.view_id: view for view in views}
    if requested_ids:
        missing = [view_id for view_id in requested_ids if view_id not in by_id]
        if missing:
            raise KeyError("Requested view ids were not found in the render job: %s" % ", ".join(missing))
        selected = [by_id[view_id] for view_id in requested_ids]
    else:
        selected = list(views)

    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive when provided.")
        selected = selected[:limit]
    return selected


def resolve_device(device_flag: str) -> str:
    import torch

    if device_flag == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_flag == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but torch.cuda.is_available() is false.")
    return device_flag


def resolve_dtype(dtype_flag: str, device: str):
    import torch

    if dtype_flag == "float16":
        return torch.float16
    if dtype_flag == "float32":
        return torch.float32
    if device == "cuda":
        return torch.float16
    return torch.float32


def load_pipeline(args: argparse.Namespace, device: str):
    import torch
    from diffusers import AutoPipelineForInpainting

    dtype = resolve_dtype(args.dtype, device)
    load_kwargs = {
        "torch_dtype": dtype,
        "local_files_only": args.local_files_only,
    }
    if args.cache_dir is not None:
        load_kwargs["cache_dir"] = str(args.cache_dir)

    if dtype == torch.float16:
        try:
            pipe = AutoPipelineForInpainting.from_pretrained(args.model_id, variant="fp16", **load_kwargs)
        except Exception:
            pipe = AutoPipelineForInpainting.from_pretrained(args.model_id, **load_kwargs)
    else:
        pipe = AutoPipelineForInpainting.from_pretrained(args.model_id, **load_kwargs)

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    return pipe


def load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def load_mask_image(path: Optional[Path], size: Tuple[int, int], args: argparse.Namespace) -> Image.Image:
    if args.no_mask or path is None:
        return Image.new("L", size, 255)

    with Image.open(path) as image:
        mask = image.convert("L")

    if mask.size != size:
        mask = mask.resize(size, resample=RESAMPLE_NEAREST)

    mask_array = np.asarray(mask, dtype=np.uint8)
    if args.mask_polarity == "background":
        mask_array = 255 - mask_array
    binary = np.where(mask_array > args.mask_threshold, 255, 0).astype(np.uint8)
    processed = Image.fromarray(binary, mode="L")

    if args.mask_dilate > 0:
        kernel_size = max(3, 2 * int(args.mask_dilate) + 1)
        processed = processed.filter(ImageFilter.MaxFilter(size=kernel_size))
    return processed


def compute_square_bbox(mask: Image.Image, padding_ratio: float, min_size: int) -> Tuple[int, int, int, int]:
    mask_array = np.asarray(mask, dtype=np.uint8) > 0
    height, width = mask_array.shape
    ys, xs = np.where(mask_array)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, width, height)

    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1

    box_width = x1 - x0
    box_height = y1 - y0
    side = int(round(max(box_width, box_height) * (1.0 + float(padding_ratio) * 2.0)))
    side = max(side, int(min_size))
    side = min(side, width, height)

    center_x = 0.5 * (x0 + x1)
    center_y = 0.5 * (y0 + y1)
    left = int(round(center_x - side / 2.0))
    top = int(round(center_y - side / 2.0))
    left = max(0, min(left, width - side))
    top = max(0, min(top, height - side))
    return (left, top, left + side, top + side)


def build_blend_mask(mask: Image.Image, blur_radius: float) -> Image.Image:
    blended = mask
    if blur_radius > 0:
        blended = blended.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
    return blended


def composite_result(
    original_rgb: Image.Image,
    refined_rgb: Image.Image,
    alpha_mask: Optional[Image.Image],
    bbox: Tuple[int, int, int, int],
) -> Image.Image:
    left, top, right, bottom = bbox
    base = original_rgb.copy()
    refined_patch = refined_rgb.resize((right - left, bottom - top), resample=RESAMPLE_LANCZOS)
    if alpha_mask is not None:
        alpha = alpha_mask.resize((right - left, bottom - top), resample=RESAMPLE_LANCZOS)
        original_patch = base.crop(bbox)
        refined_patch = Image.composite(refined_patch, original_patch, alpha)
    base.paste(refined_patch, (left, top))
    return base


def save_debug_artifacts(
    debug_dir: Path,
    view_id: str,
    original_rgb: Image.Image,
    original_mask: Image.Image,
    work_rgb: Image.Image,
    work_mask: Image.Image,
) -> None:
    ensure_dir(debug_dir)
    original_rgb.save(debug_dir / ("%s_input.png" % view_id))
    original_mask.save(debug_dir / ("%s_mask.png" % view_id))
    work_rgb.save(debug_dir / ("%s_crop_input.png" % view_id))
    work_mask.save(debug_dir / ("%s_crop_mask.png" % view_id))


def prepare_work_images(
    original_rgb: Image.Image,
    mask: Image.Image,
    args: argparse.Namespace,
) -> Tuple[Image.Image, Image.Image, Tuple[int, int, int, int], Optional[Image.Image]]:
    width, height = original_rgb.size
    bbox = (0, 0, width, height)
    if not args.no_crop_to_mask:
        bbox = compute_square_bbox(mask, padding_ratio=args.crop_padding, min_size=args.min_crop_size)

    crop_rgb = original_rgb.crop(bbox)
    crop_mask = mask.crop(bbox)
    work_rgb = crop_rgb.resize((args.target_size, args.target_size), resample=RESAMPLE_LANCZOS)
    work_mask = crop_mask.resize((args.target_size, args.target_size), resample=RESAMPLE_NEAREST)
    alpha_mask = build_blend_mask(crop_mask, args.mask_blur)
    return work_rgb, work_mask, bbox, alpha_mask


def run_pipeline_on_view(
    pipe,
    view: ViewSpec,
    output_dir: Path,
    args: argparse.Namespace,
    device: str,
    process_index: int,
    debug_dir: Optional[Path],
) -> Path:
    import torch

    output_path = output_dir / ("%s.png" % view.view_id)
    if args.skip_existing and output_path.exists():
        print("Skipping existing view %s" % view.view_id)
        return output_path

    original_rgb = load_rgb_image(view.rgb_path)
    original_mask = load_mask_image(view.mask_path, original_rgb.size, args)
    work_rgb, work_mask, bbox, alpha_mask = prepare_work_images(original_rgb, original_mask, args)

    if debug_dir is not None:
        save_debug_artifacts(debug_dir, view.view_id, original_rgb, original_mask, work_rgb, work_mask)

    generator_device = "cpu" if device == "cpu" else device
    if args.seed_mode == "shared":
        effective_seed = int(args.seed)
    else:
        effective_seed = int(args.seed) + int(process_index)
    generator = torch.Generator(device=generator_device).manual_seed(effective_seed)
    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=work_rgb,
        mask_image=work_mask,
        strength=float(args.strength),
        guidance_scale=float(args.guidance_scale),
        num_inference_steps=int(args.num_inference_steps),
        generator=generator,
        height=args.target_size,
        width=args.target_size,
    ).images[0]
    result = result.convert("RGB")

    if args.no_preserve_background:
        final_image = result.resize(original_rgb.size, resample=RESAMPLE_LANCZOS)
    else:
        final_image = composite_result(original_rgb, result, alpha_mask, bbox)

    if final_image.size != original_rgb.size:
        raise RuntimeError(
            "Refined image size mismatch for view %s: expected %s, got %s"
            % (view.view_id, original_rgb.size, final_image.size)
        )
    final_image.save(output_path)
    print("Refined view %s with seed %d -> %s" % (view.view_id, effective_seed, output_path))
    return output_path


def build_metadata(
    job_json: Path,
    output_dir: Path,
    selected_views: Sequence[ViewSpec],
    args: argparse.Namespace,
    device: str,
) -> Dict:
    resolved_dtype = str(resolve_dtype(args.dtype, device)).replace("torch.", "")
    return {
        "job_json": str(job_json),
        "output_dir": str(output_dir),
        "model_id": args.model_id,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "device": device,
        "dtype": args.dtype,
        "resolved_dtype": resolved_dtype,
        "strength": args.strength,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "target_size": args.target_size,
        "output_resolution_mode": "match_input",
        "seed": args.seed,
        "seed_mode": args.seed_mode,
        "use_mask": not args.no_mask,
        "mask_polarity": args.mask_polarity,
        "mask_threshold": args.mask_threshold,
        "mask_dilate": args.mask_dilate,
        "mask_blur": args.mask_blur,
        "crop_to_mask": not args.no_crop_to_mask,
        "crop_padding": args.crop_padding,
        "preserve_background": not args.no_preserve_background,
        "selected_view_ids": [view.view_id for view in selected_views],
    }


def main() -> None:
    args = parse_args()
    job_json = args.job_json.resolve()
    if not job_json.exists():
        raise FileNotFoundError("Render job JSON was not found: %s" % job_json)
    if args.target_size <= 0 or args.target_size % 8 != 0:
        raise ValueError("--target-size must be a positive multiple of 8 for SDXL.")
    if not 0.0 < float(args.strength) <= 1.0:
        raise ValueError("--strength must be in the interval (0, 1].")
    if int(args.num_inference_steps) <= 0:
        raise ValueError("--num-inference-steps must be positive.")
    if float(args.guidance_scale) < 0.0:
        raise ValueError("--guidance-scale must be non-negative.")

    all_views = parse_view_specs(job_json)
    selected_views = select_views(all_views, args.view_ids, args.limit)
    if not selected_views:
        raise ValueError("No views were selected for refinement.")

    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    else:
        default_name = "sdxl_%dviews" % len(selected_views)
        if args.seed_mode == "shared":
            default_name += "_sharedseed"
        output_dir = (job_json.parent / default_name).resolve()
    ensure_dir(output_dir)
    debug_dir = ensure_dir(output_dir / "debug") if args.save_debug else None
    device = resolve_device(args.device)

    metadata = build_metadata(job_json, output_dir, selected_views, args, device)
    dump_json(output_dir / "metadata.json", metadata)
    print("Prepared %d view(s) from %s" % (len(selected_views), job_json))
    print("Output directory: %s" % output_dir)

    if args.dry_run:
        print("Dry run complete; pipeline load was skipped.")
        return

    pipe = load_pipeline(args, device)
    for process_index, view in enumerate(selected_views):
        run_pipeline_on_view(
            pipe=pipe,
            view=view,
            output_dir=output_dir,
            args=args,
            device=device,
            process_index=process_index,
            debug_dir=debug_dir,
        )

    print("Finished refining %d view(s) into %s" % (len(selected_views), output_dir))


if __name__ == "__main__":
    main()
