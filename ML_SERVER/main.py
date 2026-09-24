import os
import json
from io import BytesIO
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, File, UploadFile, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError, ImageFile
import torch
import torch.nn as nn
from torchvision import transforms, models
import uvicorn

# Allow truncated image bytes to be read safely without crashing
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ============================================================
# API Application Setup
# ============================================================
app = FastAPI(
    title="AI Crop Disease Triage API",
    description="Two-stage MobileNetV3 Triage Pipeline (OOD Rejection -> 38-Class Disease Detection)",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Global Configuration & Thresholds
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

LEAF_MODEL_PATH = os.path.join(MODEL_DIR, "leaf_classifier.pth")
LEAF_MAP_PATH = os.path.join(MODEL_DIR, "leaf_class_names.json")

DISEASE_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
DISEASE_MAP_PATH = os.path.join(MODEL_DIR, "class_names.json")

DISEASE_INFO_PATH = os.path.join(BASE_DIR, "disease_info.json")

LEAF_CONFIDENCE_THRESHOLD = 0.70
DISEASE_CONFIDENCE_THRESHOLD = 0.60

# Device selection (Render Free uses CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Globals for caching models in memory
leaf_model: Optional[nn.Module] = None
disease_model: Optional[nn.Module] = None
leaf_classes: Dict[int, str] = {}
disease_classes: Dict[int, str] = {}
disease_info_db: Dict[str, Any] = {}

# Exact preprocessing transforms matching training pipeline
inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# Helper Functions
# ============================================================
def parse_label(class_label: str):
    """Parses PlantVillage label into plant name and disease name."""
    parts = class_label.split("___")
    plant = parts[0].replace("_", " ").strip()
    disease = parts[1].replace("_", " ").strip() if len(parts) > 1 else None

    if disease and disease.lower() == "healthy":
        return plant, None, "healthy"
    return plant, disease, "diseased"

def get_confidence_level(score: float) -> str:
    """Classifies confidence into qualitative tiers."""
    if score >= 0.80: return "high"
    elif score >= 0.60: return "medium"
    return "low"

def load_mobile_net(weights_path: str, num_classes: int) -> nn.Module:
    """Instantiates MobileNetV3-Small and maps saved weights securely."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model file missing: {weights_path}")
        
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def load_json_mapping(filepath: str) -> Dict[int, str]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Mapping file missing: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}

# ============================================================
# Server Startup Lifecycle
# ============================================================
@app.on_event("startup")
def startup_event():
    """Loads all models and mappings exactly ONCE during boot."""
    global leaf_model, disease_model, leaf_classes, disease_classes, disease_info_db

    print(f"--- Booting Server on Device: {device} ---")

    # 1. Load Mappings & Disease Knowledge
    leaf_classes = load_json_mapping(LEAF_MAP_PATH)
    disease_classes = load_json_mapping(DISEASE_MAP_PATH)
    
    # Verify LEAF map integrity
    leaf_vals = set(leaf_classes.values())
    if "LEAF" not in leaf_vals or "NOT_LEAF" not in leaf_vals:
        raise ValueError(f"CRITICAL: Invalid leaf mapping. Found {leaf_vals}")

    if not os.path.exists(DISEASE_INFO_PATH):
        print(f"Warning: Disease info database missing at {DISEASE_INFO_PATH}")
    else:
        with open(DISEASE_INFO_PATH, "r", encoding="utf-8") as f:
            disease_info_db = json.load(f)

    # 2. Load Models into Memory
    print("Loading Leaf OOD Classifier...")
    leaf_model = load_mobile_net(LEAF_MODEL_PATH, 2)
    
    print("Loading 38-Class Disease Classifier...")
    disease_model = load_mobile_net(DISEASE_MODEL_PATH, 38)
    
    print("--- Models Loaded Successfully! ---")

# ============================================================
# API Endpoints
# ============================================================
@app.get("/")
def root():
    return {
        "success": True,
        "service": "AI Crop Disease Triage API",
        "status": "online"
    }

@app.get("/health")
def health():
    return {
        "success": True,
        "status": "healthy",
        "leaf_model_loaded": leaf_model is not None,
        "disease_model_loaded": disease_model is not None,
        "device": str(device)
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Validate File Upload
    if not file or not file.filename:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"success": False, "error": "No file uploaded."}
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"success": False, "error": "Uploaded file must be a valid image."}
        )

    # 2. Decode Image Safely
    try:
        image_bytes = await file.read()
        if not image_bytes:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"success": False, "error": "Empty image file."}
            )
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, ValueError, Exception):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"success": False, "error": "Corrupted or unreadable image format."}
        )

    # 3. CPU/GPU Inference Block
    try:
        tensor = inference_transforms(image).unsqueeze(0).to(device)
    except Exception:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "error": "Image tensor preprocessing failed."}
        )

    with torch.no_grad():
        # --- STAGE 1: LEAF CLASSIFIER (OOD REJECTION) ---
        l_outputs = leaf_model(tensor)
        l_probs = torch.nn.functional.softmax(l_outputs[0], dim=0)
        l_conf, l_idx = torch.max(l_probs, 0)
        
        leaf_prediction = leaf_classes[l_idx.item()]
        leaf_confidence = round(float(l_conf.item()), 4)

        if leaf_prediction == "NOT_LEAF" or leaf_confidence < LEAF_CONFIDENCE_THRESHOLD:
            return {
                "success": True,
                "status": "not_leaf",
                "message": "Please upload a clear photo of a crop leaf.",
                "leaf_prediction": "NOT_LEAF",
                "leaf_confidence": leaf_confidence
            }

        # --- STAGE 2: DISEASE CLASSIFIER ---
        d_outputs = disease_model(tensor)
        d_probs = torch.nn.functional.softmax(d_outputs[0], dim=0)
        
        # Get Top-3
        top_k_prob, top_k_idx = torch.topk(d_probs, 3)
        top_k_prob = top_k_prob.cpu().numpy().tolist()
        top_k_idx = top_k_idx.cpu().numpy().tolist()

        best_idx = top_k_idx[0]
        best_prob = top_k_prob[0]
        best_label = disease_classes[best_idx]

        top_predictions = []
        for p, i in zip(top_k_prob, top_k_idx):
            lbl = disease_classes[i]
            t_plant, t_dis, _ = parse_label(lbl)
            top_predictions.append({
                "class_label": lbl,
                "plant": t_plant,
                "disease": t_dis,
                "confidence": round(float(p), 4),
                "confidence_percent": round(float(p) * 100, 2)
            })

        # --- UNCERTAINTY GATE ---
        if best_prob < DISEASE_CONFIDENCE_THRESHOLD:
            return {
                "success": True,
                "status": "uncertain",
                "message": "The image is not clear enough for a reliable prediction. Please take another clear photo.",
                "leaf_prediction": "LEAF",
                "leaf_confidence": leaf_confidence,
                "disease_confidence": round(best_prob, 4),
                "top_predictions": top_predictions
            }

        # --- FINAL SUCCESS RESPONSE ---
        plant, disease, health_status = parse_label(best_label)
        
        disease_info = disease_info_db.get(best_label, {
            "name": disease if disease else f"Healthy {plant}",
            "description": "No detailed information currently available in the database.",
            "symptoms": [],
            "causes": [],
            "prevention": [],
            "recommended_action": []
        })

        return {
            "success": True,
            "status": health_status,
            "prediction": {
                "class_index": best_idx,
                "class_label": best_label,
                "plant": plant,
                "disease": disease,
                "confidence": round(best_prob, 4),
                "confidence_percent": round(best_prob * 100, 2),
                "confidence_level": get_confidence_level(best_prob)
            },
            "disease_info": disease_info,
            "top_predictions": top_predictions
        }

# ============================================================
# Local Server Execution
# ============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )
