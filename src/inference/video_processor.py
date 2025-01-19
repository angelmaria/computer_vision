# src/inference/video_processor.py
from ultralytics import YOLO
import cv2
import numpy as np
import logging
from pathlib import Path
import tempfile
from storage.detection_storage import DetectionStorage

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, model, storage, save_dir):
        """
        Initialize video processor
        Args:
            model: YOLO model instance
            storage: DetectionStorage instance
            save_dir: Directory to save detected logo images
        """
        self.model = model
        self.storage = storage
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def process_video(self, video_file, confidence_threshold=0.25, display_callback=None, progress_callback=None):
        """
        Process video file for logo detection
        Args:
            video_file: Video file to process
            confidence_threshold: Detection confidence threshold
            display_callback: Optional callback for displaying frames
            progress_callback: Optional callback for progress updates
        Returns:
            Dictionary with detection statistics
        """
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(video_file.read())
                video_path = tmp_file.name

            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")

            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps

            # Calculate frame skip based on video length
            # Process 2 frames per second for videos longer than 10 seconds
            if duration > 10:
                frame_skip = int(fps / 2)
            else:
                frame_skip = 1

            frames_with_logos = 0
            detections = []
            frame_idx = 0

            # Process frames in batches
            batch_size = 4
            frames_batch = []
            timestamps_batch = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Skip frames based on calculated rate
                if frame_idx % frame_skip != 0:
                    frame_idx += 1
                    continue

                # Update progress
                if progress_callback:
                    progress_callback(frame_idx, total_frames)

                frames_batch.append(frame)
                timestamps_batch.append(frame_idx / fps)

                # Process batch when it reaches batch_size or last frames
                if len(frames_batch) == batch_size or not ret:
                    # Batch inference
                    results = self.model(frames_batch, conf=confidence_threshold)
                    
                    # Process batch results
                    for batch_idx, (result, timestamp) in enumerate(zip(results, timestamps_batch)):
                        if len(result.boxes) > 0:
                            frames_with_logos += 1
                            
                            # Store only the detection with highest confidence for each frame
                            best_detection = None
                            best_conf = 0
                            
                            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                                if float(conf) > best_conf:
                                    class_name = result.names[int(cls)]
                                    best_detection = {
                                        'box': box.cpu().numpy(),
                                        'class_name': class_name,
                                        'confidence': float(conf)
                                    }
                                    best_conf = float(conf)
                            
                            if best_detection:
                                # Save only the best detection
                                self.storage.save_detection(
                                    frame=frames_batch[batch_idx],
                                    box=best_detection['box'],
                                    class_name=best_detection['class_name'],
                                    score=best_detection['confidence'],
                                    source_file=video_file.name,
                                    timestamp=timestamp
                                )
                                
                                detections.append({
                                    'timestamp': timestamp,
                                    'class_name': best_detection['class_name'],
                                    'confidence': best_detection['confidence']
                                })
                        
                        # Display frame
                        if display_callback:
                            annotated_frame = result.plot()
                            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                            display_callback(rgb_frame)

                    # Clear batch
                    frames_batch = []
                    timestamps_batch = []

                frame_idx += 1

            cap.release()

            # Calculate statistics
            processed_frames = total_frames / frame_skip
            logo_percentage = (frames_with_logos / processed_frames) * 100
            
            self.storage.save_video_stats(
                video_file.name,
                duration,
                total_frames,
                logo_percentage
            )

            Path(video_path).unlink()

            return {
                'duration': duration,
                'total_frames': total_frames,
                'frames_with_logos': frames_with_logos,
                'logo_percentage': logo_percentage,
                'detections': detections
            }

        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise

    def get_statistics(self, video_file):
        """Get statistics for a processed video"""
        return self.storage.get_video_statistics(video_file.name)