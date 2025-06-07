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

# === Load Environment Variables ===
load_dotenv()
OPENROUTER_API_KEY = os.getenv("HF_API_KEY")

# === Validate API Key Load ===
if not OPENROUTER_API_KEY:
    raise ValueError("HF_API_KEY not found in environment variables")

# === OpenRouter Configuration ===
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
openrouter_headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
}

# === Load Pretrained Classification Model ===
def load_model(model_path: str = "skin_injury_model.pth"):
    model = models.efficientnet_b0(pretrained=False)
    num_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(num_features, 5)
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

# === Image Preprocessing Function ===
def preprocess_image(file: bytes) -> tuple[torch.Tensor, Image.Image]:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(file)).convert("RGB")
    return transform(image).unsqueeze(0), image

# === Class Names ===
class_names = ['ingrown nails', 'abrasion', 'bruises', 'burn', 'cut']

# === Initialize FastAPI App ===
app = FastAPI()

# === CORS Configuration ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load Model on Startup ===
model = load_model()

# === Prediction Endpoint ===
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image_tensor, _ = preprocess_image(contents)

        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_class = class_names[predicted.item()]

        return {"predicted_class": predicted_class}
    except Exception as e:
        return {"error": "Prediction failed", "details": str(e)}

# === Chatbot (AI Recommendation) Endpoint ===
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

        return {"response": data["choices"][0]["message"]["content"].strip()}

    except Exception as e:
        return {"error": "Chatbot Error", "details": str(e)}

# === Root Test Endpoint ===
@app.get("/")
async def root():
    return {"message": "API is running!"}
