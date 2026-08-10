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

def get_device():
    # Temporarily force CPU to avoid MPS deadlock/hang in MobileSAM
    return "cpu"
