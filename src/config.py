# src/config.py
from pathlib import Path
import torch

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
RUNS_DIR = PROJECT_ROOT / 'runs'
MODEL_PATH = RUNS_DIR / 'detect' / 'train7' / 'best.pt'
PRETRAINED_MODEL = 'yolov8n.pt'  # Used if no trained model exists

# Brand configuration
BRAND_CLASSES = ['nike', 'adidas', 'puma']
NUM_CLASSES = len(BRAND_CLASSES) + 1  # +1 for background class

# Detection configuration
CONFIDENCE_THRESHOLD = 0.25
SAVE_DIR = PROJECT_ROOT / 'detected_images'
DB_PATH = PROJECT_ROOT / 'detections.db'

# Training configuration
DATA_PATH = PROJECT_ROOT / 'data'
YAML_PATH = DATA_PATH / 'dataset.yaml'

def get_model_path():
    """Get the appropriate model path based on what's available"""
    if MODEL_PATH.exists():
        return MODEL_PATH
    return PRETRAINED_MODEL

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
HALF_PRECISION = True if DEVICE == 'cuda' else False  # Use FP16 for faster GPU inference