import os
import json
from io import BytesIO
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, File, UploadFile, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
import torch
import torch.nn as nn
from torchvision import transforms, models
import uvicorn

# ============================================================
# Application Setup & CORS
# ============================================================

app = FastAPI(
    title="AI Crop Disease Triage API",
    description="Backend API powered by MobileNetV3-Small for 38-class crop disease detection",
    version="2.0.0"
)

# Development CORS configuration allowing web clients and Android apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Global Variables and Configuration
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pth")
CLASS_MAPPING_PATH = os.path.join(BASE_DIR, "model", "class_names.json")
DISEASE_INFO_PATH = os.path.join(BASE_DIR, "disease_info.json")

CONFIDENCE_THRESHOLD = 0.60  # 60% rule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model: Optional[nn.Module] = None
class_mapping: Dict[int, str] = {}
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
    """
    Parses PlantVillage formatted label into plant name, disease name, and health status.
    Example: 'Apple___Black_rot' -> ('Apple', 'Black rot', 'diseased')
    """
    parts = class_label.split("___")
    plant = parts[0].replace("_", " ").strip()
    disease = parts[1].replace("_", " ").strip() if len(parts) > 1 else None

    if disease and disease.lower() == "healthy":
        return plant, None, "healthy"
    return plant, disease, "diseased"


def get_confidence_level(score: float) -> str:
    """Classifies confidence into qualitative levels."""
    if score >= 0.80:
        return "high"
    elif score >= 0.60:
        return "medium"
    return "low"

# ============================================================
# Lifecycle Startup
# ============================================================

@app.on_event("startup")
def load_assets():
    """Load model, class mapping, and disease knowledge base once on startup."""
    global model, class_mapping, disease_info_db

    print(f"Initializing service on device: {device}")

    # 1. Load Class Mapping
    if not os.path.exists(CLASS_MAPPING_PATH):
        raise FileNotFoundError(f"Class mapping file not found at {CLASS_MAPPING_PATH}")
    with open(CLASS_MAPPING_PATH, "r", encoding="utf-8") as f:
        raw_mapping = json.load(f)
        class_mapping = {int(k): v for k, v in raw_mapping.items()}

    num_classes = len(class_mapping)
    print(f"Loaded {num_classes} classes from {CLASS_MAPPING_PATH}")

    # 2. Load Disease Information Database
    if not os.path.exists(DISEASE_INFO_PATH):
        raise FileNotFoundError(f"Disease info file not found at {DISEASE_INFO_PATH}")
    with open(DISEASE_INFO_PATH, "r", encoding="utf-8") as f:
        disease_info_db = json.load(f)
    print(f"Loaded disease database with {len(disease_info_db)} entries.")

    # 3. Load Trained Model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}")

    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("MobileNetV3-Small loaded successfully in evaluation mode.")

# ============================================================
# API Endpoints
# ============================================================

@app.get("/")
def root():
    return {
        "success": True,
        "service": "AI Crop Disease Triage API",
        "status": "online",
        "device": str(device),
        "classes_loaded": len(class_mapping)
    }


@app.get("/health")
def health():
    return {
        "success": True,
        "status": "healthy",
        "model_ready": model is not None
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Validate file presence & type
    if not file or not file.filename:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"success": False, "error": "No file uploaded."}
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"success": False, "error": "Uploaded file must be a valid image (e.g., JPEG, PNG)."}
        )

    # 2. Read and decode image safely
    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"success": False, "error": "Uploaded image file is empty."}
            )
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, ValueError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"success": False, "error": "Corrupted or unreadable image file."}
        )
    except Exception:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "error": "Internal server error while processing image."}
        )

    # 3. Model Preprocessing
    try:
        tensor = inference_transforms(image).unsqueeze(0).to(device)
    except Exception:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "error": "Image transform preprocessing failed."}
        )

    # 4. PyTorch Inference (No gradients)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # 5. Extract Top-3 Predictions
    top_k_prob, top_k_idx = torch.topk(probabilities, 3)
    top_k_prob = top_k_prob.cpu().numpy().tolist()
    top_k_idx = top_k_idx.cpu().numpy().tolist()

    top_predictions: List[Dict[str, Any]] = []
    for prob, idx in zip(top_k_prob, top_k_idx):
        lbl = class_mapping.get(idx, "Unknown")
        p_plant, p_disease, _ = parse_label(lbl)
        top_predictions.append({
            "class_index": idx,
            "class_label": lbl,
            "plant": p_plant,
            "disease": p_disease,
            "confidence": round(float(prob), 4),
            "confidence_percent": round(float(prob) * 100, 2)
        })

    # Top-1 result details
    top1 = top_predictions[0]
    top_conf = top1["confidence"]
    top_label = top1["class_label"]
    top_index = top1["class_index"]
    plant, disease, health_status = parse_label(top_label)

    # 6. Apply 60% Confidence Rule
    if top_conf < CONFIDENCE_THRESHOLD:
        return {
            "success": True,
            "status": "retake",
            "message": "The image is not clear enough for a reliable prediction. Please take another clear photo of the leaf.",
            "confidence": top_conf,
            "confidence_percent": top1["confidence_percent"],
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "top_predictions": top_predictions
        }

    # 7. Fetch Information from Disease Database
    disease_info = disease_info_db.get(top_label, {
        "name": disease if disease else f"Healthy {plant}",
        "description": "No specific database description is available for this class.",
        "symptoms": [],
        "causes": [],
        "prevention": [],
        "recommended_action": []
    })

    return {
        "success": True,
        "status": health_status,
        "prediction": {
            "class_index": top_index,
            "class_label": top_label,
            "plant": plant,
            "disease": disease,
            "confidence": top_conf,
            "confidence_percent": top1["confidence_percent"],
            "confidence_level": get_confidence_level(top_conf)
        },
        "disease_info": disease_info,
        "top_predictions": top_predictions
    }

# ============================================================
# Server Execution
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )