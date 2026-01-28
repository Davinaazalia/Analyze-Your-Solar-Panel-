FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy only production requirements
COPY requirements-prod.txt .

# Install Python packages (optimized for production)
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy only necessary application files
COPY api_server.py .
COPY inference_helper.py .
COPY models/saved_models/best_solar_panel_classifier.pt ./models/saved_models/
COPY Web_Implementation ./Web_Implementation

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health', timeout=5)"

# Run the Flask API
CMD ["python", "api_server.py"]
