import os
import time
import json
import argparse
from pathlib import Path
from PIL import Image

from segment import run_segmentation
from curate_server import start_server
from extract import extract_subjects
from refine import refine_edges
from label import label_subjects

def create_contact_sheet(final_data, output_dir, output_filename="contact_sheet.png"):
    if not final_data:
        return
        
    print("Generating contact sheet...")
    # Calculate grid size
    n = len(final_data)
    cols = int(n ** 0.5)
    rows = (n + cols - 1) // cols
    
    thumb_size = 200
    padding = 20
    
    sheet_w = cols * thumb_size + (cols + 1) * padding
    sheet_h = rows * thumb_size + (rows + 1) * padding
    
    sheet = Image.new("RGBA", (sheet_w, sheet_h), (30, 30, 30, 255))
    
    for i, item in enumerate(final_data):
        row = i // cols
        col = i % cols
        
        x = padding + col * (thumb_size + padding)
        y = padding + row * (thumb_size + padding)
        
        img_path = os.path.join(output_dir, item['filename'])
        try:
            img = Image.open(img_path).convert("RGBA")
            img.thumbnail((thumb_size, thumb_size))
            
            # Center in the grid slot
            offset_x = x + (thumb_size - img.width) // 2
            offset_y = y + (thumb_size - img.height) // 2
            
            # Create a checkerboard background for transparency visibility
            bg = Image.new("RGBA", (img.width, img.height), (100, 100, 100, 255))
            sheet.paste(bg, (offset_x, offset_y))
            sheet.paste(img, (offset_x, offset_y), mask=img)
        except Exception as e:
            print(f"Error adding {item['filename']} to contact sheet: {e}")
            
    sheet.save(os.path.join(output_dir, output_filename))

def main():
    parser = argparse.ArgumentParser(description="Subject Extractor")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--output", default="outputs", help="Base output directory")
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image {args.image} not found.")
        return
        
    # Create unique run directory
    run_id = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output, run_id)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"--- Subject Extractor ---")
    print(f"Input: {args.image}")
    print(f"Output Directory: {output_dir}")
    
    # Stage 1 & 2: Segment and Filter
    print("\n--- STAGE 1 & 2: SEGMENT & FILTER ---")
    masks_data = run_segmentation(args.image, output_dir)
    
    if not masks_data:
        print("No masks found. Exiting.")
        return
        
    # Stage 3: Curate UI
    print("\n--- STAGE 3: CURATE ---")
    import webbrowser
    webbrowser.open("http://127.0.0.1:8000")
    selected_ids, api_key = start_server(output_dir, masks_data, port=8000)
    
    if not selected_ids:
        print("No subjects selected. Exiting.")
        return
        
    # Stage 4: Extract
    print("\n--- STAGE 4: EXTRACT ---")
    extracted_data = extract_subjects(args.image, selected_ids, masks_data, output_dir)
    
    # Stage 5: Refine (rembg)
    print("\n--- STAGE 5: REFINE ---")
    refined_data = refine_edges(extracted_data, output_dir)
    
    # Stage 6: Label (moondream + DeepSeek API)
    print("\n--- STAGE 6: LABEL ---")
    final_data = label_subjects(refined_data, output_dir, api_key=api_key)
    
    # Finalize: Write manifest and contact sheet
    print("\n--- FINALIZING ---")
    manifest = {
        "source_image": os.path.abspath(args.image),
        "subjects": final_data
    }
    
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        
    create_contact_sheet(final_data, output_dir)
    
    print(f"\nDone! Extracted {len(final_data)} subjects to {output_dir}")
    
if __name__ == "__main__":
    main()
