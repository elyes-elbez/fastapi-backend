from fastapi import FastAPI, File, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import io
import torch
from torchvision import models, transforms
from dotenv import load_dotenv
import os
import requests
import gc  # for manual garbage collection
import psutil  # optional, for memory tracking

# === Load Environment Variables ===
load_dotenv()
OPENROUTER_API_KEY = os.getenv("HF_API_KEY")
print("HF_API_KEY =", OPENROUTER_API_KEY)

# === OpenRouter Config ===
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
openrouter_headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
}

def load_severity_model(model_path):
    model = models.efficientnet_b0(pretrained=False)
    num_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(num_features, 2)
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

def load_model(model_path="skin_injury_model.pth"):
    model = models.efficientnet_b0(pretrained=False)
    num_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(num_features, 5)
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

def preprocess_image(file) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(file)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, image

severity_classes = ['not severe', 'severe']
class_names = ['ingrown nails', 'abrasion', 'bruises', 'burn', 'cut']

# === FastAPI App Initialization ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model()

# === Predict Endpoint ===
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image_tensor, _ = preprocess_image(contents)

        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_class = class_names[predicted.item()]

        model_filename = f"injury_{predicted_class.replace(' ', '_')}_model.pth"
        if not os.path.exists(model_filename):
            return {"error": f"Model file '{model_filename}' not found for class '{predicted_class}'."}

        severity_model = load_severity_model(model_filename)

        with torch.no_grad():
            severity_output = severity_model(image_tensor)
            _, severity_idx = torch.max(severity_output, 1)
            severity_label = severity_classes[severity_idx.item()]

        # Free memory
        del severity_model, severity_output, severity_idx
        gc.collect()

        return {
            "injury_type": predicted_class,
            "severity": severity_label
        }

    except Exception as e:
        return {"error": "Prediction failed", "details": str(e)}

# === Chatbot Endpoint ===
@app.post("/chat")
async def chat(
    prompt: str = Body(""),
    name: str = Body(...),
    age: int = Body(...),
    diagnosis: str = Body(...),
    allergies: str = Body(...)
):
    try:
        patient_info = f"""
Patient name: {name}
Age: {age}
Diagnosis: {diagnosis}
Allergies: {allergies}
"""

        payload = {
            "model": "mistralai/mistral-7b-instruct:free",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a medical assistant. Based on the provided patient information, generate a 2-line treatment recommendation."
                },
                {
                    "role": "user",
                    "content": patient_info
                }
            ]
        }

        response = requests.post(OPENROUTER_URL, headers=openrouter_headers, json=payload)
        data = response.json()

        if response.status_code != 200 or "choices" not in data:
            return {"error": "API Error", "details": data}

        return {"response": data["choices"][0]["message"]["content"]}

    except Exception as e:
        return {"error": "Internal Server Error", "details": str(e)}

# === Root Test Endpoint ===
@app.get("/")
async def root():
    return {"message": "API is running!"}
