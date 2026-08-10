import os
import urllib.request
import yaml
from pathlib import Path

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def download_model(url, save_path):
    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"Downloading model from {url} to {save_path}...")
        urllib.request.urlretrieve(url, save_path)
        print("Download complete.")
    return save_path

def get_device(config=None):
    device_override = "auto"
    env_device = os.environ.get('SUBJECT_EXTRACTOR_DEVICE')
    if env_device:
        device_override = env_device
    elif config and 'hardware' in config and 'device' in config['hardware']:
        device_override = config['hardware']['device'].lower()
    
    if device_override == "cpu":
        return "cpu"
    
    import torch
    
    if device_override in ["mps", "auto"]:
        if torch.backends.mps.is_available():
            try:
                # Test MPS initialization
                torch.zeros(1).to("mps")
                return "mps"
            except Exception as e:
                print(f"MPS initialization failed: {e}. Falling back to CPU.")
    
    if device_override in ["cuda", "auto"]:
        if torch.cuda.is_available():
            try:
                # Test CUDA initialization
                torch.zeros(1).to("cuda")
                return "cuda"
            except Exception as e:
                print(f"CUDA initialization failed: {e}. Falling back to CPU.")
                
    return "cpu"
