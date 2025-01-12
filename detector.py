# detector.py
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
    def __init__(self, project_dir: str = "Computer_Vision_F5"):
        self.project_dir = Path(project_dir)
        self.model_path = self.project_dir / "data" / "models" / "best.pt"
        
        if self.model_path.exists():
            self.model = YOLO(str(self.model_path))
        else:
            self.model = YOLO('yolov8n.pt')  # Fallback to pretrained
        
        self.class_names = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def process_image(self, image: np.ndarray, conf_threshold: float = 0.25) -> Tuple[np.ndarray, List[Dict]]:
        """Process a single image and return detections"""
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
        """Process a video file and return detection statistics"""
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

    def test_single_image(self, image_path: str, output_path: str = None):
        """Test detection on a single image"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        annotated_image, detections = self.process_image(image)
        
        # Draw additional information
        for detection in detections:
            box = detection['bbox']
            conf = detection['confidence']
            label = f"{detection['class_name']} {conf:.2f}"
            
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_x = box[0]
            text_y = box[3] + text_size[1] + 5
            
            cv2.putText(
                annotated_image,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
        
        if output_path:
            cv2.imwrite(output_path, annotated_image)
            return detections
        else:
            cv2.imshow('Detection Result', annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return detections

if __name__ == "__main__":
    # Example usage
    detector = BrandDetector("data/models/best.pt")
    detector.test_single_image("path/to/test/image.jpg", "output.jpg")