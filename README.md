# FastAPI for YOLOv9 with Triton Inference Server

This project provides a **FastAPI** application for serving YOLOv9 object detection using **Triton Inference Server**. The API allows users to upload images and receive predictions with bounding boxes.

## Features
- Accepts image uploads via **FastAPI**
- Preprocesses images for YOLOv9 inference
- Sends requests to **Triton Inference Server**
- Retrieves and returns model predictions
- Supports **ONNX-based YOLOv9 models**

## Requirements

### Install Dependencies
Make sure you have Python installed, then install the required packages:

```sh
pip install fastapi tritonclient[http] numpy pillow torch torchvision uvicorn python-multipart
```

### Triton Inference Server
Ensure **Triton Server** is running and serving the YOLOv9 ONNX model using `hieupth/tritonserver:24.08`.

```sh
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v /path/to/model_repository:/models \
    hieupth/tritonserver:24.08 \
    tritonserver --model-repository=/models
```

## Docker Deployment
You can deploy the FastAPI application inside a Docker container using the provided `Dockerfile`:

### 1. Build the Docker Image
```sh
docker build -t fastapi-yolov9 .
```

### 2. Run the Container
```sh
docker run --rm -p 8000:8000 fastapi-yolov9
```

## Usage

### 1. Start the FastAPI Server
Run the FastAPI application with:

```sh
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 2. Send a Prediction Request
You can use `curl` or Postman to send an image for inference:

```sh
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@test_image.jpg'
```

### 3. Response Format
The response will contain raw inference outputs from Triton Server:

```json
{
  "debug_output": [[...]]
}
```

## API Endpoints

### `POST /predict`
- **Description:** Accepts an image file and returns YOLOv9 predictions.
- **Request:** Multipart form-data with an image file.
- **Response:** JSON containing raw model output.

## Code Overview

### `main.py`
- **FastAPI** app initialization
- **Preprocessing**: Resizes and converts images to tensors
- **Inference**: Sends requests to Triton Server via `tritonclient`
- **Response Handling**: Returns raw model output

## TODO
- Post-process YOLOv9 outputs to return bounding boxes
- Implement confidence threshold filtering
- Draw and return detected bounding boxes


