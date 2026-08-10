import os
import cv2
import numpy as np

def extract_subjects(image_path, selected_ids, masks_data, output_dir):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
        
    extracted_data = []
    
    for mask_info in masks_data:
        mask_id = mask_info['id']
        if mask_id not in selected_ids:
            continue
            
        mask_file = os.path.join(output_dir, mask_info['mask_file'])
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        
        # Bounding box
        x, y, w, h = [int(v) for v in mask_info['bbox']]
        
        # Add padding
        padding = int(max(w, h) * 0.1) # 10% padding
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        cropped_img = image[y1:y2, x1:x2]
        cropped_mask = mask[y1:y2, x1:x2]
        
        # Feather the mask slightly to soften edges
        blurred_mask = cv2.GaussianBlur(cropped_mask, (5, 5), 0)
        
        # Add alpha channel
        b, g, r = cv2.split(cropped_img)
        rgba = cv2.merge((b, g, r, blurred_mask))
        
        extracted_file = os.path.join(output_dir, f"{mask_id}_extracted.png")
        cv2.imwrite(extracted_file, rgba)
        
        extracted_data.append({
            "id": mask_id,
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "extracted_file": f"{mask_id}_extracted.png",
            "aspect_ratio": (y2 - y1) / float(max(x2 - x1, 1))
        })
        
    return extracted_data
