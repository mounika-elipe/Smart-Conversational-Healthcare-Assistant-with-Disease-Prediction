import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import joblib
import librosa
from tensorflow import keras
import tempfile
import pandas as pd

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "skin_model.pth"
META_PATH = "model_meta.json"
CLASS_PATH = "classes.json"
PRECAUTION_FILE = "Diseases_with_Precautions.xlsx"

# ----------------------------------------------------
# APP SETUP
# ----------------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# GLOBAL MODELS
# ----------------------------------------------------
image_model = None
IMAGE_CLASS_NAMES = []

text_model = None
text_vectorizer = None
label_encoder = None

voice_model = None
voice_scaler = None
voice_encoder = None

PRECAUTIONS_MAP = {}

# ----------------------------------------------------
# LOAD PRECAUTIONS FROM EXCEL
# ----------------------------------------------------
def load_precautions(filepath):
    precautions = {}
    try:
        df = pd.read_excel(filepath)
        required_cols = {'Disease', 'Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4'}
        if required_cols.issubset(df.columns):
            for _, row in df.iterrows():
                disease = str(row['Disease']).strip().lower()
                p_list = [
                    str(row['Precaution_1']).strip(),
                    str(row['Precaution_2']).strip(),
                    str(row['Precaution_3']).strip(),
                    str(row['Precaution_4']).strip()
                ]
                precautions[disease] = [p for p in p_list if p and p.lower() != 'nan']
            print(f"✅ Loaded precautions for {len(precautions)} diseases")
        else:
            print("⚠️ Excel missing expected columns.")
    except Exception as e:
        print(f"❌ Error reading precautions: {e}")
    return precautions


# ----------------------------------------------------
# IMAGE PREPROCESSING
# ----------------------------------------------------
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


# ----------------------------------------------------
# STARTUP EVENT - LOAD ALL MODELS ONCE
# ----------------------------------------------------
@app.on_event("startup")
async def load_models():
    global image_model, IMAGE_CLASS_NAMES
    global text_model, text_vectorizer, label_encoder
    global voice_model, voice_scaler, voice_encoder
    global PRECAUTIONS_MAP

    # ---------- IMAGE MODEL ----------
    try:
        print("📸 Loading image model...")
        with open(META_PATH, "r") as f:
            meta = json.load(f)
        arch = meta.get("architecture", "resnet18")
        num_classes = meta.get("num_classes", 0)

        with open(CLASS_PATH, "r") as f:
            IMAGE_CLASS_NAMES = json.load(f)

        if arch == "resnet18":
            model = models.resnet18(weights=None)
        elif arch == "resnet34":
            model = models.resnet34(weights=None)
        elif arch == "resnet50":
            model = models.resnet50(weights=None)
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

        model.fc = nn.Linear(model.fc.in_features, num_classes)
        obj = torch.load(MODEL_PATH, map_location=device)
        if isinstance(obj, dict):
            model.load_state_dict(obj)
        else:
            model = obj

        model.to(device)
        model.eval()
        image_model = model
        print("✅ Image model loaded!")
    except Exception as e:
        print(f"❌ Error loading image model: {e}")

    # ---------- TEXT MODEL ----------
    try:
        text_model = joblib.load("textbased_model.pkl")
        text_vectorizer = joblib.load("textbased_vectorizer.pkl")
        label_encoder = joblib.load("textbased_encoder.pkl")
        print("✅ Text model loaded!")
    except Exception as e:
        print(f"⚠️ Error loading text models: {e}")

    # ---------- VOICE MODEL ----------
    try:
        voice_model = keras.models.load_model("voice_disease_model.h5")
        voice_scaler = joblib.load("voice_scaler.pkl")
        voice_encoder = joblib.load("voice_encoder.pkl")
        print("✅ Voice model loaded!")
    except Exception as e:
        print(f"⚠️ Error loading voice models: {e}")

    # ---------- PRECAUTIONS ----------
    PRECAUTIONS_MAP = load_precautions(PRECAUTION_FILE)


# ----------------------------------------------------
# ROUTES
# ----------------------------------------------------
@app.get("/")
async def root():
    return {"message": "✅ Backend is running successfully!"}


# -------------------- IMAGE PREDICTION --------------------
# -------------------- IMAGE PREDICTION --------------------
@app.post("/predict/image")
async def predict_image(file: UploadFile):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_tensor = preprocess_image(image).to(device)

        with torch.no_grad():
            outputs = image_model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            predicted_idx = probabilities.argmax().item()
            predicted_label = IMAGE_CLASS_NAMES[predicted_idx]
            confidence = probabilities[predicted_idx].item()

        disease_key = predicted_label.strip().lower()
        precautions = PRECAUTIONS_MAP.get(disease_key, ["No specific precautions found."])

        # 🔧 Unified response keys
        return {
            "prediction": predicted_label,
            "confidence": round(confidence, 4),
            "precautions": precautions
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image prediction error: {e}")

# -------------------- TEXT PREDICTION --------------------
@app.post("/predict/text")
async def predict_text(text: str = Form(...)):
    try:
        if not text.strip():
            raise HTTPException(status_code=400, detail="Input text is empty")

        vectorized = text_vectorizer.transform([text])
        prediction = text_model.predict(vectorized)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        disease_key = predicted_label.strip().lower()
        precautions = PRECAUTIONS_MAP.get(disease_key, ["No specific precautions found."])

        return {"prediction": predicted_label, "precautions": precautions}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Text prediction error: {e}")


# -------------------- VOICE PREDICTION --------------------
@app.post("/predict/voice")
async def predict_voice(file: UploadFile):
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        audio, sr = librosa.load(tmp_path, sr=None)
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
        mfccs_scaled = voice_scaler.transform([mfccs])
        prediction = voice_model.predict(mfccs_scaled)
        predicted_index = np.argmax(prediction, axis=1)[0]
        predicted_label = voice_encoder.inverse_transform([predicted_index])[0]

        disease_key = predicted_label.strip().lower()
        precautions = PRECAUTIONS_MAP.get(disease_key, ["No specific precautions found."])

        return {
            "prediction": predicted_label,
            "confidence": round(float(np.max(prediction)), 4),
            "precautions": precautions
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Voice prediction error: {e}" )