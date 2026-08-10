import os
import time
import json
import argparse

from segment import run_segmentation
from extract import extract_subjects
from refine import refine_edges
from label import label_subjects
from main import create_contact_sheet

def auto_process(image_path, output_base="outputs"):
    run_id = time.strftime("auto_%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base, run_id)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"--- Subject Extractor (Auto Mode) ---")
    print(f"Input: {image_path}")
    print(f"Output: {output_dir}")
    
    print("\n--- STAGE 1 & 2: SEGMENT & FILTER ---")
    masks_data = run_segmentation(image_path, output_dir)
    
    if not masks_data:
        print("No masks found. Exiting.")
        return None
        
    print(f"\n--- STAGE 3: AUTO-SELECT ({len(masks_data)} masks) ---")
    selected_ids = [m['id'] for m in masks_data]
    
    print("\n--- STAGE 4: EXTRACT ---")
    extracted_data = extract_subjects(image_path, selected_ids, masks_data, output_dir)
    
    print("\n--- STAGE 5: REFINE ---")
    refined_data = refine_edges(extracted_data, output_dir)
    
    print("\n--- STAGE 6: LABEL ---")
    final_data = label_subjects(refined_data, output_dir)
    
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
    auto_process("stickers.jpg")
