# src/storage/detection_storage.py
import sqlite3
import cv2
import os
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DetectionStorage:
    def __init__(self, db_path, save_dir):
        """
        Initialize storage system
        Args:
            db_path: Path to SQLite database
            save_dir: Directory to save detected logo images
        """
        self.db_path = Path(db_path)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS videos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    duration REAL,
                    total_frames INTEGER,
                    logo_percentage REAL
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id INTEGER,
                    class_name TEXT NOT NULL,
                    confidence REAL,
                    timestamp REAL,
                    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
                    image_path TEXT,
                    FOREIGN KEY(video_id) REFERENCES videos(id)
                )
            ''')
    
    def save_video_stats(self, filename, duration, total_frames, logo_percentage):
        """Save video processing statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                INSERT INTO videos (filename, duration, total_frames, logo_percentage)
                VALUES (?, ?, ?, ?)
            ''', (filename, duration, total_frames, logo_percentage))
            return cursor.lastrowid
    
    def save_detection(self, frame, box, class_name, score, source_file, timestamp):
        """
        Save a single detection
        Args:
            frame: Video frame as numpy array
            box: Bounding box coordinates [x1, y1, x2, y2]
            class_name: Detected class name
            score: Detection confidence score
            source_file: Source video filename
            timestamp: Detection timestamp in seconds
        """
        try:
            # Create unique filename for the detection image
            detection_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{class_name}_{timestamp:.2f}.jpg"
            image_path = self.save_dir / detection_filename

            # Crop and save detection image
            x1, y1, x2, y2 = map(int, box)
            cropped = frame[y1:y2, x1:x2]
            cv2.imwrite(str(image_path), cropped)

            # Save to database
            with sqlite3.connect(self.db_path) as conn:
                # Get or create video entry
                cursor = conn.execute('SELECT id FROM videos WHERE filename = ?', (source_file,))
                video_id = cursor.fetchone()
                
                if video_id is None:
                    cursor = conn.execute(
                        'INSERT INTO videos (filename) VALUES (?)',
                        (source_file,)
                    )
                    video_id = cursor.lastrowid
                else:
                    video_id = video_id[0]

                # Save detection
                conn.execute('''
                    INSERT INTO detections 
                    (video_id, class_name, confidence, timestamp, x1, y1, x2, y2, image_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (video_id, class_name, score, timestamp, x1, y1, x2, y2, str(image_path)))

        except Exception as e:
            logger.error(f"Error saving detection: {e}")
            raise

    def get_video_statistics(self, video_filename):
        """Get statistics for a processed video"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('''
                SELECT v.*, 
                       COUNT(d.id) as total_detections,
                       AVG(d.confidence) as avg_confidence
                FROM videos v
                LEFT JOIN detections d ON v.id = d.video_id
                WHERE v.filename = ?
                GROUP BY v.id
            ''', (video_filename,))
            return dict(cursor.fetchone())