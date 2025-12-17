import os
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pandas as pd
import joblib

# ----------------------------
# PATHS
# ----------------------------
IMAGE_MODEL_PATH = r"E:\archive\backend\skin_model.pth"
VOICE_MODEL_PATH = r"E:\archive\backend\voice_disease_model.h5"
TEXT_DATA_PATH = r"E:\project1\archive (3)\DiseaseAndSymptoms.csv"
CLASSES_PATH = r"E:\archive\backend\classes.json"
META_PATH = r"E:\archive\backend\model_meta.json"

# ----------------------------
# IMAGE MODEL EVALUATION
# ----------------------------
print("\n📸 Evaluating IMAGE model...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open(CLASSES_PATH, "r") as f:
    class_names = json.load(f)
num_classes = len(class_names)

meta = json.load(open(META_PATH))
model_name = meta["architecture"]

# Load model
model = getattr(models, model_name)(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# Prepare test data
test_dir = r"E:\dermnet_project\data\dermnet\test"
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

y_true, y_pred = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ✅ Only use classes present in predictions
unique_classes = np.unique(np.concatenate((y_true, y_pred)))
unique_class_names = [class_names[i] for i in unique_classes]

print("\n📊 IMAGE MODEL PERFORMANCE:\n")
print(classification_report(
    y_true,
    y_pred,
    labels=unique_classes,
    target_names=unique_class_names,
    zero_division=0
))

# Metrics
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

print(f"✅ Accuracy: {acc:.4f}")
print(f"✅ Precision: {prec:.4f}")
print(f"✅ Recall: {rec:.4f}")
print(f"✅ F1-score: {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - IMAGE Model")
plt.savefig("image_confusion_matrix.png")
plt.close()
print("🖼️ Confusion matrix saved for IMAGE model.\n")

# ---------------------------
# VOICE MODEL EVALUATION
# ---------------------------
# -------------------------------------------------------
# 🎤 VOICE MODEL EVALUATION (Robust Version)
# -------------------------------------------------------

print("\n🎤 Evaluating VOICE model...")

try:
    print(f"📏 Voice scaler expects {voice_scaler.n_features_in_} features per sample.")
except Exception:
    pass

# --- Predict Voice Data ---
preds_voice = voice_disease_model.predict(X_voice)
y_pred_labels_voice = voice_encoder.inverse_transform(np.argmax(preds_voice, axis=1))

# --- Handle unseen labels gracefully ---
known_labels = set(voice_encoder.classes_)
X_voice_final, y_voice_final, y_pred_final = [], [], []

for i, label in enumerate(y_voice):
    if label in known_labels:
        X_voice_final.append(X_voice[i])
        y_voice_final.append(label)
        y_pred_final.append(y_pred_labels_voice[i])
    else:
        print(f"⚠️ Skipping unseen label: {label}")

if len(y_voice_final) == 0:
    print("⚠️ No valid labels for evaluation. Please ensure labels match the encoder classes.")
else:
    y_true_enc = voice_encoder.transform(y_voice_final)
    y_pred_enc = voice_encoder.transform(y_pred_final)

    print("\n🎤 VOICE MODEL PERFORMANCE:\n")
    print(classification_report(
        y_true_enc,
        y_pred_enc,
        target_names=voice_encoder.classes_,
        zero_division=0
    ))

    # --- Metrics ---
    accuracy = accuracy_score(y_true_enc, y_pred_enc)
    precision = precision_score(y_true_enc, y_pred_enc, average='weighted', zero_division=0)
    recall = recall_score(y_true_enc, y_pred_enc, average='weighted', zero_division=0)
    f1 = f1_score(y_true_enc, y_pred_enc, average='weighted', zero_division=0)

    print(f"✅ Accuracy: {accuracy:.4f}")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall: {recall:.4f}")
    print(f"✅ F1-score: {f1:.4f}")

    # --- Confusion Matrix ---
    cm_voice = confusion_matrix(y_true_enc, y_pred_enc)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_voice,
        annot=True,
        fmt="d",
        xticklabels=voice_encoder.classes_,
        yticklabels=voice_encoder.classes_,
        cmap="Oranges"
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("VOICE Model Confusion Matrix")
    plt.savefig("voice_confusion_matrix.png")
    plt.close()
    print("🖼️ Confusion matrix saved for VOICE model.\n")

#text
print("\n🧠 Evaluating TEXT model...")

TEXT_DATASET_PATH = r"E:\project1\archive (3)\DiseaseAndSymptoms"
TEXT_MODEL_PATH = r"E:\archive\backend\textbased_model.h5"
TEXT_TOKENIZER_PATH = r"E:\archive\backend\text_tokenizer.pkl"
TEXT_ENCODER_PATH = r"E:\archive\backend\text_encoder.pkl"

# Load tokenizer, encoder, and model
tokenizer = joblib.load(TEXT_TOKENIZER_PATH)
encoder = joblib.load(TEXT_ENCODER_PATH)
model = load_model(TEXT_MODEL_PATH)

texts = []
labels = []

# Each file in the dataset folder is assumed to be <disease_name>.csv
for file in os.listdir(TEXT_DATASET_PATH):
    if file.endswith(".csv"):
        disease_name = file.replace(".csv", "")
        file_path = os.path.join(TEXT_DATASET_PATH, file)
        df = pd.read_csv(file_path)
        # Assuming your CSV has a column like 'Symptoms'
        if 'Symptoms' in df.columns:
            for s in df['Symptoms'].dropna().tolist():
                texts.append(str(s))
                labels.append(disease_name)

if len(texts) == 0:
    print("⚠️ No text data found in the given path.")
else:
    # Tokenize and pad
    sequences = tokenizer.texts_to_sequences(texts)
    X_text = pad_sequences(sequences, maxlen=100)
    y_text = np.array(labels)

    # Predict
    preds = model.predict(X_text)
    y_pred_labels = encoder.inverse_transform(np.argmax(preds, axis=1))

    # Encode actual labels
    y_true_enc = encoder.transform(y_text)
    y_pred_enc = encoder.transform(y_pred_labels)

    print("\n📊 TEXT MODEL PERFORMANCE:\n")
    print(classification_report(y_true_enc, y_pred_enc, target_names=encoder.classes_, zero_division=0))

    accuracy = accuracy_score(y_true_enc, y_pred_enc)
    precision = precision_score(y_true_enc, y_pred_enc, average='weighted', zero_division=0)
    recall = recall_score(y_true_enc, y_pred_enc, average='weighted', zero_division=0)
    f1 = f1_score(y_true_enc, y_pred_enc, average='weighted', zero_division=0)

    print(f"✅ Accuracy: {accuracy:.4f}")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall: {recall:.4f}")
    print(f"✅ F1-score: {f1:.4f}")

    # Confusion matrix
    cm_text = confusion_matrix(y_true_enc, y_pred_enc)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_text, annot=True, fmt="d", xticklabels=encoder.classes_, yticklabels=encoder.classes_, cmap="Greens")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("TEXT Model Confusion Matrix")
    plt.savefig("text_confusion_matrix.png")
    plt.close()
    print("🖼️ Confusion matrix saved for TEXT model.\n")