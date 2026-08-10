import os
import json
import argparse
import uuid
from pathlib import Path

from segment import run_segmentation
from extract import extract_subjects
from refine import refine_edges
from label import label_subjects
from contact_sheet import create_contact_sheet

def auto_process(image_path, output_base="outputs", mode=None, label_mode="ai",
                 padding_ratio=0.1, feather_kernel=5):
    run_id = f"auto_{uuid.uuid4().hex[:12]}"
    output_dir = os.path.join(output_base, run_id)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"--- Subject Extractor (Auto Mode) ---")
    print(f"Input: {image_path}")
    print(f"Output: {output_dir}")
    
    print("\n--- STAGE 1 & 2: SEGMENT & FILTER ---")
    masks_data = run_segmentation(image_path, output_dir, mode=mode)
    
    if not masks_data:
        print("No masks found. Exiting.")
        return None
        
    print(f"\n--- STAGE 3: AUTO-SELECT ({len(masks_data)} masks) ---")
    selected_ids = [m['id'] for m in masks_data]
    
    print("\n--- STAGE 4: EXTRACT ---")
    extracted_data = extract_subjects(
        image_path, selected_ids, masks_data, output_dir,
        padding_ratio=padding_ratio, feather_kernel=feather_kernel,
    )
    
    print("\n--- STAGE 5: REFINE ---")
    refined_data = refine_edges(extracted_data, output_dir)
    
    print("\n--- STAGE 6: LABEL ---")
    final_data = label_subjects(refined_data, output_dir, label_mode=label_mode)
    
    print("\n--- FINALIZING ---")
    manifest = {
        "source_image": os.path.abspath(image_path),
        "subjects": final_data
    }
    
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        
    create_contact_sheet(final_data, output_dir, "contact_sheet.png")
    print(f"\nDone! Extracted {len(final_data)} subjects to {output_dir}")
    return os.path.join(output_dir, "contact_sheet.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process one image or a folder of images")
    parser.add_argument("input", help="Image path or folder")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--mode", choices=["fast", "quality"], default=None)
    parser.add_argument("--label-mode", choices=["ai", "ocr", "basic"], default="ai")
    args = parser.parse_args()
    input_path = Path(args.input)
    images = sorted(
        p for p in (input_path.iterdir() if input_path.is_dir() else [input_path])
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    )
    if not images:
        raise SystemExit("No supported images found")
    for image in images:
        auto_process(str(image), args.output, args.mode, args.label_mode)
