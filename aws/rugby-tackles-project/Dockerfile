FROM tensorflow/tensorflow:2.15.0

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

#RUN apt-get update && apt-get install -y \
#    libgl1-mesa-glx \    # OpenGL libraries for OpenCV
#    libglib2.0-0 \       # Core system libraries
#    ffmpeg \             # Video processing (crucial for your project!)
#    git \                # Version control
#    vim \                # Text editor for debugging
#    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements first (for better layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY data/ ./data/

# Set Python path
ENV PYTHONPATH=/app

# Expose ports for Jupyter
EXPOSE 8888

# Default command
CMD ["bash"]