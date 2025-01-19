# src/config.py
from pathlib import Path
import torch

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'

# Data subdirectories
MODELS_DIR = DATA_DIR / 'models'
DETECTIONS_DIR = DATA_DIR / 'detections'
DATASET_DIR = DATA_DIR  # Contains test, train, valid folders

# Model paths
MODEL_PATH = MODELS_DIR / 'best.pt'
PRETRAINED_MODEL = 'yolov8n.pt' # Used if no trained model exists

# Brand configuration
BRAND_CLASSES = ['nike', 'adidas', 'puma']
NUM_CLASSES = len(BRAND_CLASSES) + 1  # +1 for background class

# Detection configuration
CONFIDENCE_THRESHOLD = 0.25
DB_PATH = PROJECT_ROOT / 'detections.db'

# Training configuration
YAML_PATH = DATASET_DIR / 'dataset.yaml'

def get_model_path():
    """Get the appropriate model path based on what's available"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
    if MODEL_PATH.exists():
        return MODEL_PATH
    return PRETRAINED_MODEL

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
HALF_PRECISION = True if DEVICE == 'cuda' else False  # Use FP16 for faster GPU inference