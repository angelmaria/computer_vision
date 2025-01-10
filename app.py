# app.py
import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import torch
from detector import BrandDetector
from utils.image_downloader import ImageCollector
from train import BrandTrainingSetup
import tempfile

class BrandDetectionApp:
    def __init__(self):
        self.project_dir = "Computer_Vision_F5"
        self.setup_page()
        self.initialize_session_state()
        
    def setup_page(self):
        st.set_page_config(page_title="Brand Detection", layout="wide")
        st.title("Brand Logo Detection")
        
    def initialize_session_state(self):
        if 'detector' not in st.session_state:
            model_path = Path(self.project_dir) / "models" / "best.pt"
            if model_path.exists():
                st.session_state.detector = BrandDetector(str(model_path))
            else:
                st.session_state.detector = None

    def download_images_section(self):
        st.header("1. Data Collection")
        col1, col2 = st.columns(2)
        
        with col1:
            search_query = st.text_input("Search Query (e.g., 'coca cola logo')")
            num_images = st.number_input("Number of Images", min_value=1, max_value=100, value=50)
            
        with col2:
            if st.button("Download Images"):
                if search_query:
                    with st.spinner("Downloading images..."):
                        collector = ImageCollector(self.project_dir)
                        collector.download_images(search_query, limit=num_images)
                        st.success(f"Downloaded {num_images} images for '{search_query}'")
                else:
                    st.error("Please enter a search query")

    def dataset_management_section(self):
        st.header("2. Dataset Management")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Setup Dataset Structure"):
                setup = BrandTrainingSetup(self.project_dir)
                setup.create_dataset_structure()
                setup.create_data_yaml(['coca_cola'])
                st.success("Dataset structure created!")
                
        with col2:
            if st.button("Split Dataset"):
                setup = BrandTrainingSetup(self.project_dir)
                setup.split_dataset()
                st.success("Dataset split into train/val/test sets!")

    def model_training_section(self):
        st.header("3. Model Training")
        col1, col2 = st.columns(2)
        
        with col1:
            epochs = st.number_input("Number of Epochs", min_value=1, value=50)
            batch_size = st.number_input("Batch Size", min_value=1, value=16)
            
        with col2:
            if st.button("Train Model"):
                setup = BrandTrainingSetup(self.project_dir)
                with st.spinner("Training model..."):
                    setup.train_model(epochs=epochs, batch_size=batch_size)
                st.success("Model training completed!")

    def inference_section(self):
        st.header("4. Logo Detection")
        
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            if st.session_state.detector is None:
                st.error("No trained model found. Please train a model first.")
                return
                
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Read and process image
            image = cv2.imread(tmp_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Make prediction
            annotated_image, detections = st.session_state.detector.process_image(image)
            
            # Display results
            st.image(annotated_image, caption='Detected Logos', use_column_width=True)
            
            if detections:
                st.subheader("Detections:")
                for det in detections:
                    st.write(f"- {det['class_name']}: {det['confidence']:.2%}")
            else:
                st.write("No logos detected.")
            
            # Clean up
            Path(tmp_path).unlink()

    def run(self):
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to", 
            ["Data Collection", "Dataset Management", "Model Training", "Logo Detection"])
        
        if page == "Data Collection":
            self.download_images_section()
        elif page == "Dataset Management":
            self.dataset_management_section()
        elif page == "Model Training":
            self.model_training_section()
        else:
            self.inference_section()

if __name__ == "__main__":
    app = BrandDetectionApp()
    app.run()