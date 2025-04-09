FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements_fixed.txt .
RUN pip install --no-cache-dir -r requirements_fixed.txt

# Copy application code
COPY . .

# Create model directory
RUN mkdir -p model

# Expose port for API
EXPOSE 5000

# Run the application in API mode
CMD ["python", "deploy.py", "--mode", "api", "--host", "0.0.0.0", "--port", "5000"]