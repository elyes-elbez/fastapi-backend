# main.py

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import requests
import io
import torch
from torchvision import models, transforms
import os

# === Chatbot Config ===
HF_API_KEY = os.getenv("HF_API_KEY")
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"

headers = {
    "Authorization": f"Bearer {HF_API_KEY}"
}

# === Load Classification Model ===
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

# === Image Preprocessing ===
def preprocess_image(file) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(file)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, image

# === Class Names ===
class_names = ['ingrown nails', 'abrasion', 'bruises', 'burn', 'cut']

# === FastAPI Initialization ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # update for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model()

# === Classification Endpoint ===
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image_tensor, _ = preprocess_image(contents)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]

    return {"predicted_class": predicted_class}

# === Chatbot Endpoint ===
class PromptRequest(BaseModel):
    prompt: str

@app.post("/chat")
async def chat(request: PromptRequest):
    data = {
        "inputs": request.prompt,
        "parameters": {
            "max_new_tokens": 256,
            "temperature": 0.7
        }
    }

    response = requests.post(HF_API_URL, headers=headers, json=data)

    if response.status_code != 200:
        return {"error": f"Request failed: {response.status_code}", "details": response.json()}

    try:
        text = response.json()[0]["generated_text"]
    except Exception:
        return {"error": "Unexpected response format", "response": response.json()}

    # Remove the prompt part
    if request.prompt in text:
        text = text.replace(request.prompt, "").strip()

    return {"response": text}
