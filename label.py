import os
import re
from PIL import Image
import pytesseract
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
from dotenv import load_dotenv

from utils import load_config, get_device

load_dotenv()

def get_clean_filename(caption, ocr_text, config, api_key=None):
    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Warning: DEEPSEEK_API_KEY not set. Using basic slug generator.")
        # Basic fallback slug
        text = ocr_text if ocr_text else caption
        slug = re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')
        return slug[:30]
        
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    
    prompt = f"""
You are a file naming assistant. I will provide a short image description and optional OCR text.
Return ONLY a short, clean filename slug (no extension, lowercase, words separated by hyphens).
Max 5 words. Do not include quotes or any other text.

Description: {caption}
OCR Text: {ocr_text if ocr_text else 'None'}
"""
    try:
        response = client.chat.completions.create(
            model=config['api'].get('deepseek_model', 'deepseek-chat'),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.3
        )
        slug = response.choices[0].message.content.strip()
        slug = re.sub(r'[^a-z0-9\-]+', '', slug.lower()).strip('-')
        return slug
    except Exception as e:
        print(f"DeepSeek API error: {e}")
        # Fallback
        text = ocr_text if ocr_text else caption
        slug = re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')
        return slug[:30]


def label_subjects(refined_data, output_dir, api_key=None, label_mode="ai"):
    config = load_config()
    device = get_device(config)
    
    label_mode = (label_mode or "ai").lower()
    if label_mode not in {"ai", "ocr", "basic"}:
        raise ValueError(f"Unsupported label mode: {label_mode}")

    print(f"Label mode: {label_mode}")
    model_id = "vikhyatk/moondream2"
    revision = "2024-08-26"
    
    model = None
    tokenizer = None
    if label_mode == "ai":
        print(f"Loading moondream2 on {device}...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, revision=revision
            ).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        except Exception as e:
            print(f"Failed to load moondream2: {e}")
    
    labeled_data = []
    
    for item in refined_data:
        mask_id = item['id']
        image_path = os.path.join(output_dir, item['refined_file'])
        rgba = Image.open(image_path).convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[3])
        image = bg
        
        caption = ""
        if model is not None:
            enc_image = model.encode_image(image)
            caption = model.answer_question(enc_image, "Describe the main subject concisely in a few words.", tokenizer)
            print(f"[{mask_id}] Caption: {caption}")
            
        ocr_text = pytesseract.image_to_string(image).strip() if label_mode in {"ai", "ocr"} else ""
        if ocr_text:
            # Clean up OCR slightly
            ocr_text = " ".join(ocr_text.split())[:50]
            print(f"[{mask_id}] OCR: {ocr_text}")
            
        # Generate clean filename
        if label_mode == "basic":
            slug = "subject"
        else:
            slug = get_clean_filename(caption, ocr_text, config, api_key)
        if not slug:
            slug = "unknown-subject"
            
        final_filename = f"{mask_id}_{slug}.png"
        
        # Rename file
        os.rename(image_path, os.path.join(output_dir, final_filename))
        print(f"[{mask_id}] Final filename: {final_filename}")
        
        item['caption'] = caption
        item['ocr_text'] = ocr_text
        item['filename'] = final_filename
        
        labeled_data.append(item)
        
    # Free memory
    if model is not None:
        del model
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    return labeled_data
