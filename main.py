from fastapi import FastAPI, UploadFile, File, Response
import tritonclient.http as httpclient
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import io

app = FastAPI()

# Triton Server URL and Model Name
TRITON_SERVER_URL = "localhost:8000"
MODEL_NAME = "yolov9-s_onnx"

# Initialize Triton Client
client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL)

# Image preprocessing
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Adjust based on config.pbtxt
        transforms.ToTensor(),
    ])
    image = transform(image).numpy()  # Shape: (3, 640, 640)
    image = np.expand_dims(image, axis=0)  # Add batch dimension -> (1, 3, 640, 640)
    return image.astype(np.float32)  # Ensure data type is FP32

# Draw bounding boxes
def draw_bounding_boxes(image: Image.Image, predictions):
    draw = ImageDraw.Draw(image)
    
    for pred in predictions:
        cx, cy, w, h, confidence, class_id = pred[:6]

        # Convert YOLO format (cx, cy, w, h) to (x_min, y_min, x_max, y_max)
        x_min = int(cx - w / 2)
        y_min = int(cy - h / 2)
        x_max = int(cx + w / 2)
        y_max = int(cy + h / 2)

        draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width=3)

    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Load and preprocess image
        image = Image.open(file.file).convert("RGB")
        transformed_img = preprocess_image(image)

        # Prepare Triton input
        inputs = httpclient.InferInput("input", transformed_img.shape, datatype="FP32")
        inputs.set_data_from_numpy(transformed_img, binary_data=True)

        # Specify the output layer
        outputs = httpclient.InferRequestedOutput("output", binary_data=True)

        # Query Triton Inference Server
        results = client.infer(model_name=MODEL_NAME, inputs=[inputs], outputs=[outputs])
        inference_output = results.as_numpy("output")

        print("RAW OUTPUT:", inference_output)  # Debugging step

        return {"debug_output": inference_output.tolist()}  # Temporary return to check output format

    except Exception as e:
        return {"error": str(e)}

