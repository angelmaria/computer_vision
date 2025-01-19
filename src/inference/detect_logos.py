# src/inference/detect_logos.py
from ultralytics import YOLO
import torch
import logging
from config import get_model_path, PRETRAINED_MODEL
import streamlit as st
from pathlib import Path

logger = logging.getLogger(__name__)

@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)
    if DEVICE == 'cuda':
        model.half() if HALF_PRECISION else model.float()
    return model

class LogoDetector:
    def __init__(self, custom_model_path=None):
        """
        Initialize YOLO model
        Args:
            custom_model_path: Optional path to custom trained model
        """
        try:
            if custom_model_path:
                logger.info(f"Attempting to load custom model from: {custom_model_path}")
                if not Path(custom_model_path).exists():
                    logger.error(f"Custom model path does not exist: {custom_model_path}")
                    raise FileNotFoundError(f"Custom model not found at {custom_model_path}")
                self.model = YOLO(custom_model_path)
                logger.info(f"Successfully loaded custom model from: {custom_model_path}")
            else:
                model_path = get_model_path()
                logger.info(f"Attempting to load default model from: {model_path}")
                if not Path(model_path).exists():
                    logger.error(f"Default model path does not exist: {model_path}")
                    raise FileNotFoundError(f"Default model not found at {model_path}")
                self.model = YOLO(model_path)
                logger.info(f"Successfully loaded default model from: {model_path}")
                
        except Exception as e:
            logger.error(f"Error loading custom/default model: {str(e)}")
            logger.warning("Falling back to pretrained model")
            self.model = YOLO(PRETRAINED_MODEL)
        
        # Set device
        self.device = 'cpu' if not torch.cuda.is_available() else 'cuda'
        self.model.to(self.device)
        
    def detect(self, image, conf_threshold=0.25):
        """Detect logos in image"""
        results = self.model(image, conf=conf_threshold, device=self.device)
        
        boxes = []
        scores = []
        labels = []
        
        for r in results:
            boxes.extend(r.boxes.xyxy.cpu().numpy())
            scores.extend(r.boxes.conf.cpu().numpy())
            labels.extend(r.boxes.cls.cpu().numpy())
            
        return boxes, scores, labels
    
    def detect_batch(self, images, conf_threshold=0.25):
        """Detect logos in a batch of images"""
        results = self.model(images, conf=conf_threshold, device=self.device)
        return results