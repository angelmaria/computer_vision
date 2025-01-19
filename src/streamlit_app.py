# src/streamlit_app.py
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_time(seconds):
    """Format seconds to HH:MM:SS"""
    return str(timedelta(seconds=int(seconds)))

def process_image(image, detector, confidence_threshold):
    """Process a single image and return detections"""
    boxes, scores, labels = detector.detect(image, conf_threshold=confidence_threshold)
    
    # Convert numpy array to PIL Image if necessary
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Draw boxes on image
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    detections = []
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, box)
        class_name = BRAND_CLASSES[int(label)]
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
        # Draw label
        draw.text((x1, y1-10), f'{class_name}: {score:.2f}', fill='red')
        
        detections.append({
            'class': class_name,
            'confidence': float(score),
            'box': box.tolist()
        })
    
    return img_draw, detections

def add_database_management():
    st.subheader("Database Management")
    
    # Move the database viewing logic outside of the button
    # This way it won't reprocess the video
    with sqlite3.connect(DB_PATH) as conn:
        # Get video statistics
        videos_df = pd.read_sql_query("""
            SELECT filename, processed_date, duration, 
                   total_frames, logo_percentage
            FROM videos
            ORDER BY processed_date DESC
        """, conn)
        
        if not videos_df.empty:
            st.write("Processed Videos:")
            st.dataframe(videos_df)
            
            # Get detection counts
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
            st.info("No data in database")
    
    # Delete options
    with st.expander("Delete Data"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Add a key to the checkbox to ensure proper state management
            if st.button("Clear All Data", key="clear_all"):
                confirm = st.checkbox("I understand this will delete all data", key="confirm_clear_all")
                if confirm:
                    try:
                        with sqlite3.connect(DB_PATH) as conn:
                            # Create a cursor
                            cursor = conn.cursor()
                            # Delete all records from both tables
                            cursor.execute("DELETE FROM detections")
                            cursor.execute("DELETE FROM videos")
                            # Commit the changes
                            conn.commit()
                            # Reclaim disk space
                            cursor.execute("VACUUM")
                        
                        # Delete saved images
                        save_dir = Path(DETECTIONS_DIR)
                        if save_dir.exists():
                            for file in save_dir.glob("*"):
                                try:
                                    file.unlink()
                                except Exception as e:
                                    logger.error(f"Error deleting file {file}: {e}")
                        
                        st.success("Database and saved images cleared!")
                        # Force refresh the page to show updated data
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error clearing database: {e}")
                        logger.error(f"Database clearing error: {e}", exc_info=True)
        
        with col2:
            if st.button("Clear Old Data (>30 days)", key="clear_old"):
                try:
                    with sqlite3.connect(DB_PATH) as conn:
                        cursor = conn.cursor()
                        # First get the IDs of old videos
                        cursor.execute("""
                            SELECT id FROM videos 
                            WHERE julianday('now') - julianday(processed_date) > 30
                        """)
                        old_video_ids = [row[0] for row in cursor.fetchall()]
                        
                        # Delete related detections first (due to foreign key constraint)
                        cursor.execute("""
                            DELETE FROM detections 
                            WHERE video_id IN (
                                SELECT id FROM videos 
                                WHERE julianday('now') - julianday(processed_date) > 30
                            )
                        """)
                        
                        # Then delete old videos
                        cursor.execute("""
                            DELETE FROM videos 
                            WHERE julianday('now') - julianday(processed_date) > 30
                        """)
                        
                        # Commit and reclaim space
                        conn.commit()
                        cursor.execute("VACUUM")
                        
                        # Delete associated files
                        for video_id in old_video_ids:
                            file_pattern = f"{video_id}_*"
                            for file in Path(DETECTIONS_DIR).glob(file_pattern):
                                try:
                                    file.unlink()
                                except Exception as e:
                                    logger.error(f"Error deleting file {file}: {e}")
                        
                        st.success("Old data cleared!")
                        # Force refresh the page to show updated data
                        st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error clearing old data: {e}")
                    logger.error(f"Database clearing error: {e}", exc_info=True)
                    
@st.cache_resource
def get_active_model_path():
    model_path = get_model_path()
    return str(model_path)

def main():
    st.set_page_config(page_title="Logo Detection System", layout="wide")
    
    st.title("Logo Detection System")
    
    # Display active model path
    active_model = get_active_model_path()
    st.info(f"Active Model: {active_model}")
    
    # Sidebar for mode selection and confidence threshold
    mode = st.sidebar.selectbox(
        "Select Mode",
        ["Detection", "Training"]
    )
    
    # Add confidence threshold slider to sidebar
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=CONFIDENCE_THRESHOLD,
        step=0.05,
        help="Adjust the confidence threshold for logo detection"
    )
    
    if mode == "Detection":
        # Initialize components
        storage = DetectionStorage(DB_PATH, DETECTIONS_DIR) 
        
        @st.cache_resource
        def load_detector():
            return LogoDetector(MODEL_PATH if Path(MODEL_PATH).exists() else None)
        
        try:
            detector = load_detector()
            processor = VideoProcessor(detector.model, storage, DETECTIONS_DIR)
        except Exception as e:
            st.error(f"Error initializing system: {str(e)}")
            return
        
        # Detection type selection
        detection_type = st.radio("Select Detection Type", ["Image", "Video"])
        
        if detection_type == "Image":
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['jpg', 'jpeg', 'png']
            )
            
            if uploaded_file:
                try:
                    # Read and process image
                    image = Image.open(uploaded_file)
                    
                    # Create columns for original and processed images
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Original Image")
                        st.image(image)
                    
                    with col2:
                        st.subheader("Detected Logos")
                        processed_image, detections = process_image(
                            image, 
                            detector,
                            confidence_threshold=confidence_threshold
                        )
                        st.image(processed_image)
                    
                    # Show detections
                    if detections:
                        st.subheader("Detection Results")
                        for det in detections:
                            st.write(f"Found {det['class']} with {det['confidence']:.2f} confidence")
                    else:
                        st.info("No logos detected in the image")
                        
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                    logger.error(f"Processing error: {str(e)}", exc_info=True)
        
        else:  # Video processing
            uploaded_file = st.file_uploader(
                "Choose a video file",
                type=['mp4', 'avi', 'mov']
            )
            
            if uploaded_file:
                try:
                    # Create a single progress bar
                    progress_bar = st.progress(0)
                    video_placeholder = st.empty()
                    
                    def update_progress(frame_idx, total_frames):
                        """Callback to update progress bar"""
                        progress = int((frame_idx / total_frames) * 100)
                        progress_bar.progress(progress)
                    
                    with st.spinner("Processing video..."):
                        stats = processor.process_video(
                            uploaded_file,
                            confidence_threshold=confidence_threshold,
                            display_callback=video_placeholder.image,
                            progress_callback=update_progress
                        )
                        
                        # Clear progress bar after completion
                        progress_bar.empty()
                        
                        # Display results
                        st.success("Video processing complete!")
                        
                        # Summary metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Video Duration", format_time(stats['duration']))
                        with col2:
                            st.metric("Frames with Logos", 
                                     f"{stats['frames_with_logos']}/{stats['total_frames']}")
                        with col3:
                            st.metric("Logo Presence", 
                                     f"{stats['logo_percentage']:.1f}%")
                        
                        # Detection visualization
                        if stats['detections']:
                            st.subheader("Detection Analysis")
                            
                            # Create DataFrame for better visualization
                            df = pd.DataFrame(stats['detections'])
                            
                            # Confidence over time plot
                            st.subheader("Confidence Over Time")
                            fig_conf = {
                                'data': [{
                                    'x': df['timestamp'],
                                    'y': df['confidence'],
                                    'mode': 'lines+markers',
                                    'name': 'Confidence'
                                }],
                                'layout': {
                                    'xaxis': {'title': 'Time (seconds)'},
                                    'yaxis': {'title': 'Confidence Score'}
                                }
                            }
                            st.plotly_chart(fig_conf)
                            
                            # Brand distribution
                            st.subheader("Brand Distribution")
                            brand_counts = df['class_name'].value_counts()
                            st.bar_chart(brand_counts)
                    
                    # Add database management section after video processing
                    add_database_management()
                        
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
                    logger.error(f"Processing error: {str(e)}", exc_info=True)
    
    else:  # Training mode
        st.subheader("Model Training")
        
        # Training parameters
        col1, col2 = st.columns(2)
        with col1:
            n_epochs = st.number_input("Number of epochs", min_value=1, value=50)
            optimize = st.checkbox("Perform hyperparameter optimization")
        
        with col2:
            if optimize:
                n_trials = st.number_input("Number of optimization trials", min_value=1, value=20)
            else:
                n_trials = 20
        
        if st.button("Start Training"):
            try:
                with st.spinner("Training model... This may take a while."):
                    model_path = train_model(
                        optimize=optimize,
                        n_trials=n_trials,
                        final_epochs=n_epochs
                    )
                    st.success(f"Training completed! Model saved at: {model_path}")
                    
                    # Provide option to use newly trained model
                    if st.button("Use New Model"):
                        st.experimental_rerun()
                        
            except Exception as e:
                st.error(f"Error during training: {str(e)}")
                logger.error(f"Training error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()