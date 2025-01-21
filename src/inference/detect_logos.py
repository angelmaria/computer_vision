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
            model_path = custom_model_path or get_model_path()
            logger.info(f"Attempting to load model from: {model_path}")
            
            if Path(model_path).exists():
                self.model = YOLO(model_path)
                logger.info(f"Successfully loaded model from: {model_path}")
            else:
                logger.warning(f"Model not found at {model_path}, downloading pretrained model")
                self.model = YOLO('yolov8n.pt')
            
            # Set device
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {self.device}")
            self.model.to(self.device)
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.warning("Falling back to pretrained model")
            self.model = YOLO('yolov8n.pt')
            self.device = 'cpu'
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