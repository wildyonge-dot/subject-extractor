import os
from PIL import Image
from rembg import remove, new_session
import torch

def refine_edges(extracted_data, output_dir):
    print("Refining edges with rembg...")
    
    # We cache sessions here so we only load a model once per run if multiple subjects need it.
    sessions = {}
    
    refined_data = []
    
    for item in extracted_data:
        mask_id = item['id']
        aspect_ratio = item['aspect_ratio']
        input_file = os.path.join(output_dir, item['extracted_file'])
        output_file = os.path.join(output_dir, f"{mask_id}_refined.png")
        
        # Heuristic: if height is 1.5x width, likely a standing person
        model_name = "u2net_human_seg" if aspect_ratio > 1.5 else "u2netp"
        print(f"[{mask_id}] Using model: {model_name} (aspect ratio: {aspect_ratio:.2f})")
        
        if model_name not in sessions:
            # We must set providers appropriately for the hardware
            # rembg uses ONNX runtime under the hood. CoreML or CPU are typical for Mac.
            # Using CPU by default since it's fast enough for small crops, or CoreMLExecutionProvider if available.
            providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
            sessions[model_name] = new_session(model_name, providers=providers)
        
        session = sessions[model_name]
        
        input_image = Image.open(input_file)
        
        # Process with rembg
        output_image = remove(input_image, session=session, alpha_matting=True)
        output_image.save(output_file)
        
        item['refined_file'] = f"{mask_id}_refined.png"
        item['refine_model'] = model_name
        refined_data.append(item)
        
    # Attempt to clear rembg memory
    sessions.clear()
    
    return refined_data
