FROM python:3.11-slim

WORKDIR /app

# Install libgomp1 for LightGBM
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Run the FastAPI server
CMD uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
