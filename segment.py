import os
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

# Monkeypatch torch.as_tensor to fix MPS float64 bug in MobileSAM
_original_as_tensor = torch.as_tensor
def _mps_as_tensor(data, dtype=None, device=None, **kwargs):
    try:
        return _original_as_tensor(data, dtype=dtype, device=device, **kwargs)
    except TypeError as e:
        if "MPS" in str(e) and "float64" in str(e):
            return _original_as_tensor(data, dtype=torch.float32, device=device, **kwargs)
        raise e
torch.as_tensor = _mps_as_tensor

from utils import load_config, download_model, get_device

# Keep downloaded rembg weights inside the project so the app works in
# sandboxed environments and does not depend on a writable user home folder.
U2NET_HOME = Path(__file__).resolve().parent / "models" / "u2net"
U2NET_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("U2NET_HOME", str(U2NET_HOME))

def setup_mobilesam(config):
    model_path = Path(config['model_paths']['mobilesam'])
    if not model_path.is_absolute():
        model_path = Path(__file__).resolve().parent / model_path
    # Default URL for MobileSAM weights
    url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
    download_model(url, model_path)
    
    device = get_device(config)
    print(f"Loading MobileSAM on device: {device}")
    
    model_type = "vit_t"
    sam = sam_model_registry[model_type](checkpoint=str(model_path))
    sam.to(device=device)
    sam.eval()
    
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator

def compute_iou(box1, box2):
    # box is [x, y, w, h]
    x1_1, y1_1, w1, h1 = box1
    x2_1 = x1_1 + w1
    y2_1 = y1_1 + h1

    x1_2, y1_2, w2, h2 = box2
    x2_2 = x1_2 + w2
    y2_2 = y1_2 + h2

    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = w1 * h1
    box2_area = w2 * h2
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def filter_masks(masks, image_shape, config):
    thresholds = config['thresholds']
    min_area = image_shape[0] * image_shape[1] * thresholds['min_mask_area_ratio']
    max_area = image_shape[0] * image_shape[1] * thresholds['max_mask_area_ratio']
    
    filtered = []
    # Sort masks by area descending
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    
    for mask_data in masks:
        area = mask_data['area']
        if area < min_area or area > max_area:
            continue
            
        bbox = mask_data['bbox']
        
        # NMS check
        is_duplicate = False
        for kept_mask in filtered:
            if compute_iou(bbox, kept_mask['bbox']) > thresholds['nms_iou']:
                is_duplicate = True
                break
                
        if not is_duplicate:
            filtered.append(mask_data)
            
    return filtered


def run_fast_segmentation(image_path, output_dir):
    """Create one foreground mask quickly using the lightweight rembg model."""
    from rembg import new_session, remove

    print("Using fast foreground segmentation (u2netp)...")
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    analysis_image = image.copy()
    analysis_image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
    session = new_session("u2netp", providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
    small_rgba = remove(analysis_image, session=session, alpha_matting=False).convert("RGBA")
    alpha_image = small_rgba.getchannel("A").resize(original_size, Image.Resampling.BILINEAR)
    alpha = np.asarray(alpha_image, dtype=np.uint8)
    rgba = image.copy()
    rgba.putalpha(alpha_image)

    # Ignore tiny alpha noise when calculating the subject bounds.
    foreground = alpha > 8
    ys, xs = np.where(foreground)
    if len(xs) == 0 or len(ys) == 0:
        return []

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    mask_id = "subject_000"
    mask_filename = f"{mask_id}_mask.png"
    thumb_filename = f"{mask_id}_thumb.png"

    os.makedirs(output_dir, exist_ok=True)
    Image.fromarray(alpha, mode="L").save(os.path.join(output_dir, mask_filename))
    rgba.crop((x1, y1, x2, y2)).save(os.path.join(output_dir, thumb_filename))

    return [{
        "id": mask_id,
        "bbox": [x1, y1, x2 - x1, y2 - y1],
        "mask_file": mask_filename,
        "thumb_file": thumb_filename,
        "area": int(foreground.sum()),
        "segmentation_mode": "fast",
        "skip_refine": True,
    }]


def run_segmentation(image_path, output_dir, mode=None):
    config = load_config()
    mode = (mode or os.environ.get("SUBJECT_EXTRACTOR_MODE") or
            config.get("segmentation", {}).get("mode", "fast")).lower()
    if mode not in {"fast", "quality"}:
        raise ValueError(f"Unsupported segmentation mode: {mode}")

    if mode == "fast":
        return run_fast_segmentation(image_path, output_dir)

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load model and generate masks
    mask_generator = setup_mobilesam(config)
    print("Generating masks...")
    masks = mask_generator.generate(image_rgb)
    
    # Filter masks
    print(f"Total raw masks: {len(masks)}")
    filtered_masks = filter_masks(masks, image.shape, config)
    print(f"Filtered masks: {len(filtered_masks)}")
    
    # Save mask data and visualizations
    os.makedirs(output_dir, exist_ok=True)
    
    masks_data = []
    
    for i, mask_data in enumerate(filtered_masks):
        mask = mask_data['segmentation']
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        mask_id = f"subject_{i:03d}"
        mask_filename = os.path.join(output_dir, f"{mask_id}_mask.png")
        cv2.imwrite(mask_filename, mask_uint8)
        
        # Create thumbnail (overlay mask on image, crop to bbox)
        x, y, w, h = [int(v) for v in mask_data['bbox']]
        
        cropped_img = image[y:y+h, x:x+w]
        cropped_mask = mask_uint8[y:y+h, x:x+w]
        
        # Apply alpha to thumbnail for visualization
        b, g, r = cv2.split(cropped_img)
        thumb = cv2.merge((b, g, r, cropped_mask))
        thumb_filename = os.path.join(output_dir, f"{mask_id}_thumb.png")
        cv2.imwrite(thumb_filename, thumb)
        
        masks_data.append({
            "id": mask_id,
            "bbox": [x, y, w, h],
            "mask_file": f"{mask_id}_mask.png",
            "thumb_file": f"{mask_id}_thumb.png",
            "area": mask_data['area']
        })
        
    import gc
    del mask_generator
    gc.collect()

    # Empty torch cache if we used cuda or mps
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return masks_data
