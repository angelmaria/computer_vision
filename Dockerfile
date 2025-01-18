# Use a more complete base image
FROM python:3.10

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install dependencies in specific order
RUN pip install --no-cache-dir opencv-python-headless==4.8.1.78 && \
    pip install --no-cache-dir -r requirements.txt

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

# Command to run the application
CMD ["streamlit", "run", "src/streamlit_app.py"]