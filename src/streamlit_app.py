import streamlit as st
from pathlib import Path
import logging
from datetime import timedelta
import cv2
import numpy as np
from PIL import Image, ImageDraw
import sqlite3
from inference.video_processor import VideoProcessor
from storage.detection_storage import DetectionStorage
from inference.detect_logos import LogoDetector
from training.train_model import train_model
from config import *
import pandas as pd
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_time(seconds):
    """Format seconds to HH:MM:SS"""
    return str(timedelta(seconds=int(seconds)))

def process_image(image, detector, confidence_threshold):
    """Process a single image and return detections"""
    storage = DetectionStorage(DB_PATH, DETECTIONS_DIR)
    boxes, scores, labels = detector.detect(image, conf_threshold=confidence_threshold)
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    detections = []
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, box)
        class_name = BRAND_CLASSES[int(label)]
        
        box_color = '#39FF14'
        text_color = '#39FF14'
        
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)
        draw.text((x1, y1-10), f'{class_name}: {score:.2f}', fill=text_color)
        
        storage.save_detection(
            frame=cv_image,
            box=[x1, y1, x2, y2],
            class_name=class_name,
            score=float(score),
            source_file="uploaded_image.jpg",
            timestamp=0.0
        )
        
        detections.append({
            'class': class_name,
            'confidence': float(score),
            'box': box.tolist()
        })
    
    return img_draw, detections

def process_detection(detector, processor, confidence_threshold, detection_type):
    """Handle detection processing for both image and video"""
    upload_label = f"Choose a {detection_type.lower()} file"
    allowed_types = ['jpg', 'jpeg', 'png', 'webp'] if detection_type == "Image" else ['mp4', 'avi', 'mov']
    
    st.markdown("""
        <style>
        .uploadedFile {
            border: 2px dashed #4A90E2;
            border-radius: 8px;
            padding: 20px;
            background-color: #F8F9FA;
        }
        </style>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        upload_label,
        type=allowed_types,
        help=f"Drag and drop your {detection_type.lower()} here or click to browse"
    )
    
    if uploaded_file:
        try:
            if detection_type == "Image":
                process_image_detection(uploaded_file, detector, confidence_threshold)
            else:
                process_video_detection(uploaded_file, processor, confidence_threshold)
        except Exception as e:
            st.error(f"Error processing {detection_type.lower()}: {str(e)}")
            logger.error(f"Processing error: {str(e)}", exc_info=True)

def process_image_detection(uploaded_file, detector, confidence_threshold):
    """Handle image detection processing"""
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width =True)
    
    with col2:
        st.subheader("Detected Logos")
        processed_image, detections = process_image(image, detector, confidence_threshold)
        st.image(processed_image, use_column_width =True)
    
    if detections:
        st.markdown("### Detection Results")
        for det in detections:
            st.markdown(f"🎯 Found **{det['class']}** with **{det['confidence']:.2f}** confidence")
    else:
        st.info("No logos detected in the image")

def process_video_detection(uploaded_file, processor, confidence_threshold):
    """Handle video detection processing"""
    progress_bar = st.progress(0)
    video_placeholder = st.empty()
    
    def update_progress(frame_idx, total_frames):
        progress = int((frame_idx / total_frames) * 100)
        progress_bar.progress(progress)
    
    with st.spinner("Processing video..."):
        stats = processor.process_video(
            uploaded_file,
            confidence_threshold=confidence_threshold,
            display_callback=video_placeholder.image,
            progress_callback=update_progress
        )
        
        progress_bar.empty()
        st.success("Video processing complete!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Video Duration", format_time(stats['duration']))
        with col2:
            st.metric("Frames with Logos", f"{stats['frames_with_logos']}/{stats['total_frames']}")
        with col3:
            st.metric("Logo Presence", f"{stats['logo_percentage']:.1f}%")
        
        if stats['detections']:
            show_detection_analytics(stats)

def show_detection_analytics(stats):
    """Display detection analytics"""
    st.markdown("### Detection Analysis")
    df = pd.DataFrame(stats['detections'])
    
    st.subheader("Confidence Over Time")
    fig_conf = {
        'data': [{
            'x': df['timestamp'],
            'y': df['confidence'],
            'mode': 'lines+markers',
            'name': 'Confidence',
            'line': {'color': '#4A90E2'}
        }],
        'layout': {
            'xaxis': {'title': 'Time (seconds)'},
            'yaxis': {'title': 'Confidence Score'},
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'margin': {'t': 30}
        }
    }
    st.plotly_chart(fig_conf, use_column_width=True)
    
    st.subheader("Brand Distribution")
    brand_counts = df['class_name'].value_counts()
    st.bar_chart(brand_counts)

def add_database_management():
    """Handle database management interface"""
    st.subheader("Database Management")
    
    Path(DETECTIONS_DIR).mkdir(parents=True, exist_ok=True)
    
    def init_db():
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS videos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT,
                    processed_date TIMESTAMP,
                    duration REAL,
                    total_frames INTEGER,
                    logo_percentage REAL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id INTEGER,
                    frame_number INTEGER,
                    timestamp REAL,
                    class_name TEXT,
                    confidence REAL,
                    box_x1 REAL,
                    box_y1 REAL,
                    box_x2 REAL,
                    box_y2 REAL,
                    FOREIGN KEY(video_id) REFERENCES videos(id)
                )
            """)
            conn.commit()
    
    init_db()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Delete All Data")
        confirm_all = st.checkbox("I understand this will delete all data", key="confirm_clear_all")
        
        if confirm_all:
            if st.button("Clear All Data", key="clear_all"):
                try:
                    with sqlite3.connect(DB_PATH) as conn:
                        cursor = conn.cursor()
                        cursor.execute("DELETE FROM detections")
                        cursor.execute("DELETE FROM videos")
                        conn.commit()
                        cursor.execute("VACUUM")
                    
                    detection_dir = Path(DETECTIONS_DIR)
                    if detection_dir.exists():
                        for file in detection_dir.glob("*"):
                            try:
                                file.unlink()
                            except Exception as e:
                                logger.error(f"Error deleting file {file}: {e}")
                    
                    st.success("Successfully deleted all data!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing data: {str(e)}")
                    logger.error(f"Data clearing error: {str(e)}", exc_info=True)
    
    with col2:
        st.write("Delete Old Data")
        days_threshold = st.number_input(
            "Delete data older than (days):", 
            min_value=1,
            value=30,
            step=1
        )
        
        if st.button("Clear Old Data", key="clear_old"):
            try:
                with sqlite3.connect(DB_PATH) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT id FROM videos 
                        WHERE julianday('now') - julianday(processed_date) > ?
                    """, (days_threshold,))
                    old_video_ids = [row[0] for row in cursor.fetchall()]
                    
                    cursor.execute("""
                        DELETE FROM detections 
                        WHERE video_id IN (
                            SELECT id FROM videos 
                            WHERE julianday('now') - julianday(processed_date) > ?
                        )
                    """, (days_threshold,))
                    
                    cursor.execute("""
                        DELETE FROM videos 
                        WHERE julianday('now') - julianday(processed_date) > ?
                    """, (days_threshold,))
                    
                    conn.commit()
                    cursor.execute("VACUUM")
                    
                    for video_id in old_video_ids:
                        file_pattern = f"{video_id}_*"
                        for file in Path(DETECTIONS_DIR).glob(file_pattern):
                            try:
                                file.unlink()
                            except Exception as e:
                                logger.error(f"Error deleting file {file}: {e}")
                    
                    st.success(f"Successfully deleted data older than {days_threshold} days!")
                    time.sleep(1)
                    st.rerun()
            except Exception as e:
                st.error(f"Error clearing old data: {str(e)}")
                logger.error(f"Database clearing error: {str(e)}", exc_info=True)
    
    st.subheader("Current Database Contents")
    try:
        with sqlite3.connect(DB_PATH) as conn:
            videos_df = pd.read_sql_query("""
                SELECT filename, processed_date, duration, 
                       total_frames, logo_percentage
                FROM videos
                ORDER BY processed_date DESC
            """, conn)
            
            if not videos_df.empty:
                st.write("Processed Videos:")
                st.dataframe(videos_df)
                
                detections_df = pd.read_sql_query("""
                    SELECT v.filename, 
                           COUNT(*) as detection_count,
                           GROUP_CONCAT(DISTINCT d.class_name) as detected_brands
                    FROM videos v
                    JOIN detections d ON v.id = d.video_id
                    GROUP BY v.filename
                """, conn)
                
                st.write("Detection Summary:")
                st.dataframe(detections_df)
            else:
                st.info("No videos in database")
            
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM detections")
            detection_count = cursor.fetchone()[0]
            st.metric("Total Detections", detection_count)
            
    except Exception as e:
        st.error(f"Error displaying database contents: {str(e)}")
        logger.error(f"Database display error: {str(e)}", exc_info=True)

def handle_training_mode():
    """Handle the training mode interface"""
    st.subheader("Model Training")
    
    col1, col2 = st.columns(2)
    with col1:
        n_epochs = st.number_input("Number of epochs", min_value=1, value=50)
        optimize = st.checkbox("Perform hyperparameter optimization")
    
    with col2:
        n_trials = st.number_input(
            "Number of optimization trials",
            min_value=1,
            value=20,
            disabled=not optimize
        )
    
    if st.button("Start Training", type="primary"):
        try:
            with st.spinner("Training model... This may take a while."):
                model_path = train_model(
                    optimize=optimize,
                    n_trials=n_trials,
                    final_epochs=n_epochs
                )
                st.success(f"Training completed! Model saved at: {model_path}")
                
                if st.button("Use New Model"):
                    st.rerun()
                    
        except Exception as e:
            st.error(f"Error during training: {str(e)}")
            logger.error(f"Training error: {str(e)}", exc_info=True)

@st.cache_resource
def get_active_model_path():
    model_path = get_model_path()
    return str(model_path)

def main():
    st.set_page_config(
        page_title="Logo Detection System",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
        <style>
        .stApp {
            background-color: #FAFAFA;
        }
        .main .block-container {
            padding-top: 2rem;
        }
        h1 {
            color: #1E3D59;
            margin-bottom: 2rem;
        }
        .stAlert {
            background-color: #E3F2FD;
            border: none;
            padding: 1rem;
        }
        .stMetric {
            background-color: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("Logo Detection System")
    
    active_model = get_active_model_path()
    st.info(f"Active Model: {active_model}")
    
    mode = st.sidebar.selectbox("Select Mode", ["Detection", "Training"])
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=CONFIDENCE_THRESHOLD,
        step=0.05,
        help="Adjust the confidence threshold for logo detection"
    )
    
    if mode == "Detection":
        try:
            # Initialize detector and processor
            @st.cache_resource
            def load_detector():
                return LogoDetector(MODEL_PATH if Path(MODEL_PATH).exists() else None)
            
            detector = load_detector()
            storage = DetectionStorage(DB_PATH, DETECTIONS_DIR)
            processor = VideoProcessor(detector.model, storage, DETECTIONS_DIR)
            
            # Detection type selection and file upload
            detection_type = st.radio("Select Detection Type", ["Image", "Video"])
            process_detection(detector, processor, confidence_threshold, detection_type)
            
            # Database management section
            add_database_management()
            
        except Exception as e:
            st.error(f"Error initializing system: {str(e)}")
            logger.error(f"Initialization error: {str(e)}", exc_info=True)
            return
    
    else:  # Training mode
        handle_training_mode()

if __name__ == "__main__":
    main()