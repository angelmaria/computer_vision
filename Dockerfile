# Use Python 3.10 slim image
FROM python:3.10-slim

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better cache usage
COPY requirements.txt .

# Install Python dependencies with specific CPU-only versions
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir opencv-python-headless==4.8.1.78 \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the application
COPY . .

# Create directories for data persistence
RUN mkdir -p /app/data/detections
RUN mkdir -p /app/data/models

# Expose Streamlit port
EXPOSE 8501

# Set environment variables
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV PYTORCH_CPU_ONLY=1

# Command to run the application
CMD ["streamlit", "run", "src/streamlit_app.py"]