import os
from PIL import Image
import cv2
import torch
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
from segment_anything import (SamAutomaticMaskGenerator, SamPredictor,
                              sam_model_registry)

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
MASK_PREVIEW_DIRNAME = 'sam_masks_preview'
MASK_PREVIEW_MAX_MASKS = 24
MASK_PREVIEW_PALETTE = np.array([
    [230, 25, 75],
    [60, 180, 75],
    [255, 225, 25],
    [0, 130, 200],
    [245, 130, 48],
    [145, 30, 180],
    [70, 240, 240],
    [240, 50, 230],
    [210, 245, 60],
    [250, 190, 190],
    [0, 128, 128],
    [230, 190, 255],
    [170, 110, 40],
    [255, 250, 200],
    [128, 0, 0],
    [170, 255, 195],
], dtype=np.uint8)

def has_images(path):
    if not os.path.isdir(path):
        return False
    for entry in os.listdir(path):
        if os.path.splitext(entry)[1].lower() in IMAGE_EXTENSIONS:
            return True
    return False

def resolve_image_dir(image_root, downsample, downsample_type):
    if downsample_type == 'mask' or downsample == 1:
        candidates = [
            os.path.join(image_root, 'images'),
            image_root,
        ]
        for candidate in candidates:
            if has_images(candidate):
                return candidate, False
    else:
        candidates = [
            os.path.join(image_root, 'images_' + str(downsample)),
            os.path.join(image_root, 'images'),
            image_root,
        ]
        for index, candidate in enumerate(candidates):
            if has_images(candidate):
                return candidate, index > 0
    raise FileNotFoundError(f"Could not find readable images under image_root: {image_root}")

def save_mask_preview(image_bgr, masks, output_path):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    overlay = image_rgb.astype(np.float32).copy()

    if masks.numel() > 0 and masks.shape[0] > 0:
        mask_areas = masks.reshape(masks.shape[0], -1).sum(dim=1)
        sorted_indices = torch.argsort(mask_areas, descending=True).tolist()

        for rank, mask_idx in enumerate(sorted_indices[:MASK_PREVIEW_MAX_MASKS]):
            mask = masks[mask_idx].cpu().numpy().astype(np.uint8)
            color = MASK_PREVIEW_PALETTE[rank % len(MASK_PREVIEW_PALETTE)].astype(np.float32)
            overlay[mask == 1] = overlay[mask == 1] * 0.55 + color * 0.45

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color.tolist(), 1)

    preview = np.concatenate([image_rgb, np.clip(overlay, 0, 255).astype(np.uint8)], axis=1)
    cv2.imwrite(output_path, cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    
    parser = ArgumentParser(description="SAM segment everything masks extracting params")
    
    parser.add_argument("--image_root", default='/datasets/nerf_data/360_v2/garden/', type=str)
    parser.add_argument("--sam_checkpoint_path", default='./third_party/segment-anything/sam_ckpt/sam_vit_h_4b8939.pth', type=str)
    parser.add_argument("--sam_arch", default="vit_h", type=str)
    parser.add_argument("--downsample", default=1, type=int)
    parser.add_argument("--downsample_type", default='image', type=str, choices=['image', 'mask'], help="Downsample then segment, or segment then downsample.")
    parser.add_argument("--points-per-side", dest="points_per_side", default=32, type=int)
    parser.add_argument("--pred-iou-thresh", dest="pred_iou_thresh", default=0.88, type=float)
    parser.add_argument("--stability-score-thresh", dest="stability_score_thresh", default=0.95, type=float)
    parser.add_argument("--crop-n-layers", dest="crop_n_layers", default=0, type=int)
    parser.add_argument("--crop-n-points-downscale-factor", dest="crop_n_points_downscale_factor", default=1, type=int)
    parser.add_argument("--min-mask-region-area", dest="min_mask_region_area", default=100, type=int)

    args = parser.parse_args()
    
    print("Initializing SAM...")
    model_type = args.sam_arch
    sam = sam_model_registry[model_type](checkpoint=args.sam_checkpoint_path).to('cuda')
    predictor = SamPredictor(sam)

    print(
        "SAM automatic mask generator params:",
        {
            "points_per_side": args.points_per_side,
            "pred_iou_thresh": args.pred_iou_thresh,
            "stability_score_thresh": args.stability_score_thresh,
            "crop_n_layers": args.crop_n_layers,
            "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
            "min_mask_region_area": args.min_mask_region_area,
        },
    )
    
    # custom
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        box_nms_thresh=0.7,
        stability_score_thresh=args.stability_score_thresh,
        crop_n_layers=args.crop_n_layers,
        crop_n_points_downscale_factor=args.crop_n_points_downscale_factor,
        min_mask_region_area=args.min_mask_region_area,
    )

    IMAGE_DIR, downsample_manually = resolve_image_dir(args.image_root, args.downsample, args.downsample_type)
    if downsample_manually:
        print("No downsampled images, do it manually.")
    OUTPUT_DIR = os.path.join(args.image_root, 'sam_masks')
    PREVIEW_DIR = os.path.join(args.image_root, MASK_PREVIEW_DIRNAME)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    
    print("Extracting SAM segment everything masks...")
    
    image_files = [
        path for path in sorted(os.listdir(IMAGE_DIR))
        if os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS
    ]

    for path in tqdm(image_files):
        name = path.split('.')[0]
        img = cv2.imread(os.path.join(IMAGE_DIR, path))
        if img is None:
            print(f"Skipping unreadable image: {os.path.join(IMAGE_DIR, path)}")
            continue
        if downsample_manually:
            img = cv2.resize(img,dsize=(img.shape[1] // args.downsample, img.shape[0] // args.downsample),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
        masks = mask_generator.generate(img)
        # print(len(masks))
        mask_list = []
        for m in masks:
            m_score = torch.from_numpy(m['segmentation']).float().to('cuda')

            if args.downsample_type == 'mask':
                m_score = torch.nn.functional.interpolate(m_score.unsqueeze(0).unsqueeze(0), size=(img.shape[0] // args.downsample, img.shape[1] // args.downsample) , mode='bilinear', align_corners=False).squeeze()
                m_score[m_score >= 0.5] = 1
                m_score[m_score != 1] = 0
                m_score = m_score.bool()

            if len(m_score.unique()) < 2:
                continue
            else:
                mask_list.append(m_score.bool())
        if mask_list:
            masks = torch.stack(mask_list, dim=0)
        else:
            masks = torch.zeros((0, img.shape[0], img.shape[1]), dtype=torch.bool, device='cuda')

        masks_cpu = masks.cpu()
        torch.save(masks_cpu, os.path.join(OUTPUT_DIR, name+'.pt'))
        save_mask_preview(img, masks_cpu, os.path.join(PREVIEW_DIR, name + '.png'))



"""
[Download SAM checkpoint]
mkdir -p /root/dev/SegAnyGAussians/third_party/segment-anything/sam_ckpt
cd /root/dev/SegAnyGAussians/third_party/segment-anything/sam_ckpt
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
"""
