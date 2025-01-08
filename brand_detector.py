import torch
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import time
from datetime import datetime
from typing import Dict, List, Tuple
import logging

class BrandDetector:
    def __init__(self, model_path: str = None):
        """
        Initialize the brand detector with a trained YOLO model
        
        Args:
            model_path: Path to trained YOLO model weights
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Initialize model
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
        else:
            self.logger.warning("No model path provided or file not found. Using pretrained model.")
            self.model = YOLO('yolov8n.pt')
        
        self.class_names = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def process_image(self, image: np.ndarray, conf_threshold: float = 0.25) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process a single image and return detections
        
        Args:
            image: Input image as numpy array
            conf_threshold: Confidence threshold for detections
            
        Returns:
            Tuple containing:
            - Annotated image
            - List of detections with coordinates and confidence
        """
        results = self.model(image, conf=conf_threshold)[0]
        annotated_img = results.plot()
        
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            class_name = self.class_names[class_id]
            
            detection = {
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': conf,
                'class_id': class_id,
                'class_name': class_name,
                'timestamp': datetime.now().isoformat()
            }
            detections.append(detection)
            
        return annotated_img, detections
    
    def process_video(self, video_path: str, output_path: str = None, save_frames: bool = False) -> Dict:
        """
        Process a video file and return detection statistics
        
        Args:
            video_path: Path to input video
            output_path: Path to save processed video (optional)
            save_frames: Whether to save frames with detections
            
        Returns:
            Dictionary containing detection statistics
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (int(cap.get(3)), int(cap.get(4)))
            )
            
        statistics = {
            'video_name': Path(video_path).name,
            'total_frames': total_frames,
            'fps': fps,
            'duration': total_frames / fps,
            'detections_per_class': {},
            'frames_with_detections': 0,
            'detection_frames': []
        }
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            annotated_frame, detections = self.process_image(frame)
            
            if detections:
                statistics['frames_with_detections'] += 1
                statistics['detection_frames'].append(frame_count)
                
                for detection in detections:
                    class_name = detection['class_name']
                    if class_name not in statistics['detections_per_class']:
                        statistics['detections_per_class'][class_name] = 0
                    statistics['detections_per_class'][class_name] += 1
            
            if output_path:
                out.write(annotated_frame)
                
            frame_count += 1
            
        # Calculate percentages
        for class_name in statistics['detections_per_class']:
            statistics['detections_per_class'][class_name] = {
                'total_detections': statistics['detections_per_class'][class_name],
                'percentage_time': (statistics['detections_per_class'][class_name] / total_frames) * 100
            }
            
        cap.release()
        if output_path:
            out.release()
            
        return statistics

    def save_detection_image(self, image: np.ndarray, detection: Dict, output_path: str):
        """
        Save a cropped detection image
        
        Args:
            image: Original image
            detection: Detection dictionary with bbox coordinates
            output_path: Path to save cropped image
        """
        x1, y1, x2, y2 = detection['bbox']
        cropped = image[y1:y2, x1:x2]
        cv2.imwrite(output_path, cropped)