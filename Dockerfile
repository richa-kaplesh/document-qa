# Dockerfile — containerizes the FastAPI backend
# Every line is a layer in the Docker image

# Start from official Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first — Docker caches this layer
# If requirements don't change, Docker skips reinstalling on rebuild
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all backend code
COPY . .

# Expose port 8000
EXPOSE 8000

# Command to run when container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]