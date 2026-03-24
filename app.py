from flask import Flask, render_template, request, send_file
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import joblib
import timm
import cv2
import os
import torch.nn as nn
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------------- APP CONFIG ----------------

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

device = torch.device("cpu")
seg_device = torch.device("cpu")

torch.serialization.default_restore_location = lambda storage, loc: storage

# ---------------- STAGE 1 ENSEMBLE ----------------

class Stage1Ensemble(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = timm.create_model("resnet50", pretrained=False, num_classes=2)
        self.effnet = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)

    def forward(self, x):
        out1 = self.resnet(x)
        out2 = self.effnet(x)
        return (out1 + out2) / 2

# ---------------- LOAD MODELS ----------------

def load_models():

    stage1_model = Stage1Ensemble()
    stage1_model.load_state_dict(
        torch.load("./models/leaf_classifier/best_stage1.pt", map_location=device)
    )
    stage1_model.to(device).eval()

    data = joblib.load("./models/disease_models/stage2_ensemble_model.pkl")

    for key in ["resnet_weights", "efficientnet_weights", "densenet_weights"]:
        data[key] = {k: v.cpu() for k, v in data[key].items()}

    svm_model = data["svm_model"]
    rf_model = data["rf_model"]
    class_names = data["class_names"]
    scaler = data.get("scaler", None)

    resnet = timm.create_model("resnet50", pretrained=False, num_classes=3)
    efficientnet = timm.create_model("efficientnet_b4", pretrained=False, num_classes=3)
    densenet = timm.create_model("densenet121", pretrained=False, num_classes=3)

    resnet.load_state_dict(data["resnet_weights"])
    efficientnet.load_state_dict(data["efficientnet_weights"])
    densenet.load_state_dict(data["densenet_weights"])

    resnet.to(device).eval()
    efficientnet.to(device).eval()
    densenet.to(device).eval()

    return stage1_model, resnet, efficientnet, densenet, svm_model, rf_model, scaler, class_names


stage1_model, resnet, efficientnet, densenet, svm_model, rf_model, scaler, class_names = load_models()

# ---------------- SEGMENTATION MODEL ----------------

seg_model = smp.Unet(
    encoder_name='efficientnet-b3',
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None
)

seg_model.load_state_dict(torch.load("./models/segmentation/best_model.pth", map_location=seg_device))
seg_model.to(seg_device)
seg_model.eval()

print("Segmentation model loaded")

# ---------------- TRANSFORMS ----------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

seg_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# ---------------- LEAF DETECTION ----------------

def is_leaf(image):
    img = np.array(image)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    mask = cv2.inRange(hsv, np.array([25, 40, 40]), np.array([85, 255, 255]))
    green_ratio = np.sum(mask > 0) / (img.shape[0] * img.shape[1])

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = max([cv2.contourArea(c) for c in contours], default=0)
    area_ratio = max_area / (img.shape[0] * img.shape[1])

    edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 50, 150)
    edge_ratio = np.sum(edges > 0) / (img.shape[0] * img.shape[1])

    return (green_ratio > 0.1 and area_ratio > 0.05 and edge_ratio > 0.01)

# ---------------- SEGMENTATION ----------------

def segment_image(image_path, threshold=0.5):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig = image.copy()

    augmented = seg_transform(image=image)
    input_tensor = augmented['image'].unsqueeze(0).to(seg_device)

    with torch.no_grad():
        output = seg_model(input_tensor)
        prob = torch.sigmoid(output)
        mask = (prob > threshold).float()

    mask = mask.squeeze().cpu().numpy()
    mask = cv2.resize(mask, (orig.shape[1], orig.shape[0]))

    return orig, mask


def disease_percentage(mask):
    total_pixels = mask.size
    disease_pixels = (mask > 0).sum()
    return (disease_pixels / total_pixels) * 100


def create_overlay(image, mask, save_path):
    overlay = image.copy()
    overlay[mask > 0] = [255, 0, 0]
    overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, overlay)

# ---------------- DISEASE INFO ----------------

disease_info = {
    "Early Blight": {
        "symptoms": ["Brown spots", "Yellow halo", "Dry leaves"],
        "cause": "Fungus (Alternaria solani)",
        "prevention": ["Use fungicide", "Remove infected leaves", "Crop rotation"]
    },
    "Late Blight": {
        "symptoms": ["Dark lesions", "White mold underside"],
        "cause": "Phytophthora infestans",
        "prevention": ["Avoid moisture", "Use resistant varieties"]
    },
    "Healthy": {
        "symptoms": ["No visible disease"],
        "cause": "None",
        "prevention": ["Maintain good farming practices"]
    }
}

# ---------------- ROUTES ----------------

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]
    path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(path)

    image = Image.open(path).convert("RGB")

    if not is_leaf(image):
        return render_template("result.html", status="not_leaf", img=path)

    img_tensor = transform(image).unsqueeze(0).to(device)

    # Stage 1
    with torch.no_grad():
        outputs = stage1_model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred1 = torch.argmax(probs, dim=1).item()

    if pred1 == 0:
        return render_template("result.html", status="not_potato", img=path)

    # Stage 2
    with torch.no_grad():
        p1 = torch.softmax(resnet(img_tensor), 1).cpu().numpy()
        p2 = torch.softmax(efficientnet(img_tensor), 1).cpu().numpy()
        p3 = torch.softmax(densenet(img_tensor), 1).cpu().numpy()

        features = resnet.forward_features(img_tensor)
        features = torch.mean(features, dim=[2, 3]).cpu().numpy()

    if scaler:
        features = scaler.transform(features)

    p4 = svm_model.predict_proba(features)
    p5 = rf_model.predict_proba(features)

    final = (p1 + p2 + p3 + p4 + p5) / 5
    pred = np.argmax(final)

    disease_raw = class_names[pred]
    disease = disease_raw.replace("Potato___", "").replace("_", " ").title()
    confidence = float(np.max(final))

    info = disease_info.get(disease, {})

    # SEGMENTATION
    mask_path = None
    disease_percent = 0

    if disease.lower() != "healthy":
        image_np, mask = segment_image(path)
        disease_percent = disease_percentage(mask)

        mask_filename = "mask_" + file.filename
        mask_path = os.path.join(app.config["UPLOAD_FOLDER"], mask_filename)

        create_overlay(image_np, mask, mask_path)

    return render_template(
        "result.html",
        status="success",
        img=path,
        mask_img=mask_path,
        disease=disease,
        confidence=confidence,
        disease_percent=round(disease_percent, 2),
        info=info
    )

# ---------------- RUN ----------------

if __name__ == "__main__":
    app.run(debug=True)