# server.py
# AI Crop Disease Triage - FastAPI Server

import os
import json
import io

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="AI Crop Disease Triage API",
    description="API for crop leaf disease detection",
    version="1.0.0"
)

# Allow Android app / website to access the server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "model")

DISEASE_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_model.pth"
)

LEAF_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "leaf_classifier.pth"
)

CLASS_NAMES_PATH = os.path.join(
    MODEL_DIR,
    "class_names.json"
)

LEAF_CLASS_NAMES_PATH = os.path.join(
    MODEL_DIR,
    "leaf_class_names.json"
)

DISEASE_INFO_PATH = os.path.join(
    BASE_DIR,
    "disease_info.json"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 224

LEAF_THRESHOLD = 0.70
DISEASE_THRESHOLD = 0.60

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# GLOBAL VARIABLES
# ============================================================

leaf_model = None
disease_model = None

leaf_class_names = []
class_names = []

disease_info = {}


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# CREATE MODEL
# ============================================================

def create_mobilenet(num_classes):

    model = models.mobilenet_v3_small(
        weights=None
    )

    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        num_classes
    )

    return model


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():

    global leaf_model
    global disease_model
    global leaf_class_names
    global class_names
    global disease_info

    print("Loading models...")

    # -----------------------------
    # Load class names
    # -----------------------------

    leaf_class_names = load_json(
        LEAF_CLASS_NAMES_PATH
    )

    class_names = load_json(
        CLASS_NAMES_PATH
    )

    disease_info = load_json(
        DISEASE_INFO_PATH
    )

    # -----------------------------
    # Leaf classifier
    # -----------------------------

    leaf_model = create_mobilenet(
        len(leaf_class_names)
    )

    leaf_checkpoint = torch.load(
        LEAF_MODEL_PATH,
        map_location=DEVICE
    )

    if isinstance(leaf_checkpoint, dict) and "model_state_dict" in leaf_checkpoint:
        leaf_model.load_state_dict(
            leaf_checkpoint["model_state_dict"]
        )
    else:
        leaf_model.load_state_dict(
            leaf_checkpoint
        )

    leaf_model.to(DEVICE)
    leaf_model.eval()

    # -----------------------------
    # Disease classifier
    # -----------------------------

    disease_model = create_mobilenet(
        len(class_names)
    )

    disease_checkpoint = torch.load(
        DISEASE_MODEL_PATH,
        map_location=DEVICE
    )

    if isinstance(disease_checkpoint, dict) and "model_state_dict" in disease_checkpoint:
        disease_model.load_state_dict(
            disease_checkpoint["model_state_dict"]
        )
    else:
        disease_model.load_state_dict(
            disease_checkpoint
        )

    disease_model.to(DEVICE)
    disease_model.eval()

    print("Models loaded successfully.")
    print("Device:", DEVICE)


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup():

    load_models()


# ============================================================
# ROOT GET
# ============================================================

@app.get("/")
def root():

    return {
        "success": True,
        "message": "AI Crop Disease Triage API is running",
        "version": "1.0.0"
    }


# ============================================================
# HEALTH GET
# ============================================================

@app.get("/health")
def health():

    return {
        "success": True,
        "status": "healthy",
        "device": str(DEVICE),
        "leaf_model_loaded": leaf_model is not None,
        "disease_model_loaded": disease_model is not None
    }


# ============================================================
# GET DISEASE LIST
# ============================================================

@app.get("/diseases")
def get_diseases():

    return {
        "success": True,
        "count": len(class_names),
        "diseases": class_names
    }


# ============================================================
# GET DISEASE INFORMATION
# ============================================================

@app.get("/disease/{disease_name}")
def get_disease(disease_name: str):

    # Direct match
    if disease_name in disease_info:

        return {
            "success": True,
            "disease": disease_name,
            "info": disease_info[disease_name]
        }

    # Try normalized matching
    for key, value in disease_info.items():

        if key.lower() == disease_name.lower():

            return {
                "success": True,
                "disease": key,
                "info": value
            }

    return {
        "success": False,
        "message": "Disease information not found"
    }


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image):

    image = image.convert("RGB")

    tensor = transform(image)

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(DEVICE)

    # ========================================================
    # STEP 1 - LEAF / NOT LEAF
    # ========================================================

    with torch.no_grad():

        leaf_output = leaf_model(tensor)

        leaf_probabilities = torch.softmax(
            leaf_output,
            dim=1
        )

        leaf_confidence, leaf_index = torch.max(
            leaf_probabilities,
            dim=1
        )

    leaf_confidence = float(
        leaf_confidence.item()
    )

    leaf_index = int(
        leaf_index.item()
    )

    leaf_prediction = leaf_class_names[
        leaf_index
    ]

    # ========================================================
    # NOT A LEAF
    # ========================================================

    if (
        leaf_prediction.upper() != "LEAF"
        or leaf_confidence < LEAF_THRESHOLD
    ):

        return {
            "success": True,
            "status": "not_leaf",
            "message": "Please upload a clear photo of a crop leaf.",
            "leaf_prediction": leaf_prediction,
            "leaf_confidence": leaf_confidence
        }

    # ========================================================
    # STEP 2 - DISEASE CLASSIFICATION
    # ========================================================

    with torch.no_grad():

        disease_output = disease_model(tensor)

        disease_probabilities = torch.softmax(
            disease_output,
            dim=1
        )

        top_values, top_indices = torch.topk(
            disease_probabilities,
            k=min(3, len(class_names)),
            dim=1
        )

    # Main prediction

    disease_index = int(
        top_indices[0][0].item()
    )

    disease_confidence = float(
        top_values[0][0].item()
    )

    disease_prediction = class_names[
        disease_index
    ]

    # ========================================================
    # TOP 3
    # ========================================================

    top_predictions = []

    for i in range(
        top_indices.shape[1]
    ):

        index = int(
            top_indices[0][i].item()
        )

        confidence = float(
            top_values[0][i].item()
        )

        top_predictions.append({
            "prediction": class_names[index],
            "confidence": confidence
        })

    # ========================================================
    # LOW CONFIDENCE
    # ========================================================

    if disease_confidence < DISEASE_THRESHOLD:

        return {
            "success": True,
            "status": "uncertain",
            "message": "The model is not confident enough. Please upload a clearer leaf image.",
            "leaf_prediction": leaf_prediction,
            "leaf_confidence": leaf_confidence,
            "prediction": disease_prediction,
            "confidence": disease_confidence,
            "top_predictions": top_predictions
        }

    # ========================================================
    # DISEASE INFORMATION
    # ========================================================

    info = disease_info.get(
        disease_prediction,
        {}
    )

    # ========================================================
    # HEALTHY
    # ========================================================

    is_healthy = (
        "healthy" in disease_prediction.lower()
    )

    status = (
        "healthy"
        if is_healthy
        else "diseased"
    )

    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    return {
        "success": True,
        "status": status,

        "leaf_prediction": leaf_prediction,
        "leaf_confidence": leaf_confidence,

        "prediction": disease_prediction,
        "confidence": disease_confidence,

        "disease_info": info,

        "top_predictions": top_predictions
    }


# ============================================================
# POST /predict
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    try:

        # Check content type

        if not file.content_type.startswith("image/"):

            return {
                "success": False,
                "message": "Please upload an image file."
            }

        # Read image

        image_bytes = await file.read()

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        # Predict

        result = predict_image(
            image
        )

        return result

    except UnidentifiedImageError:

        return {
            "success": False,
            "message": "Invalid image file."
        }

    except Exception as e:

        print("Prediction error:", e)

        return {
            "success": False,
            "message": "Prediction failed.",
            "error": str(e)
        }


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
