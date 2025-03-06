# Use the Triton Inference Server base image
FROM hieupth/tritonserver:24.08

# Set working directory inside the container
WORKDIR /workspace

# Install necessary system dependencies
RUN apt-get update && apt-get install -y python3-pip

# Install required Python packages, including python-multipart
RUN pip3 install --no-cache-dir fastapi uvicorn pillow torchvision numpy tritonclient[http] python-multipart

# Copy FastAPI application file into the container
COPY main.py /workspace/main.py

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI application when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
