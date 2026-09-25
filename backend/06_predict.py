import os
import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageFile

# Allow loading of truncated/corrupted images without instantly crashing
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ============================================================================
# CONFIGURATION THRESHOLDS
# ============================================================================
LEAF_THRESHOLD = 0.70
DISEASE_THRESHOLD = 0.60

# Model Paths
LEAF_MODEL_PATH = r"D:\Hackathon\model\leaf_classifier.pth"
LEAF_MAP_PATH = r"D:\Hackathon\model\leaf_class_names.json"

DISEASE_MODEL_PATH = r"D:\Hackathon\model\best_model.pth"
DISEASE_MAP_PATH = r"D:\Hackathon\model\class_names.json"

# ============================================================================
# HELPERS
# ============================================================================
def get_transform():
    """Exact preprocessing used during training."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

def load_custom_model(model_path, num_classes, device):
    """Recreates MobileNetV3-Small and loads specific weights."""
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def load_mapping(map_path):
    """Loads a JSON mapping file."""
    with open(map_path, "r") as f:
        class_map = json.load(f)
    return class_map

# ============================================================================
# TWO-STAGE PIPELINE LOGIC
# ============================================================================
def run_pipeline(image_path, leaf_model, disease_model, leaf_map, disease_map, transform, device):
    """Runs the two-stage logic: 1. Leaf OOD Check -> 2. Disease Classification"""
    try:
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        return None, None, None, None, f"ERROR: {e}"

    with torch.no_grad():
        # --- STAGE 1: LEAF CLASSIFIER ---
        leaf_outputs = leaf_model(tensor)
        leaf_probs = torch.softmax(leaf_outputs[0], dim=0)
        leaf_conf, leaf_idx = torch.max(leaf_probs, 0)
        
        leaf_pred = leaf_map[str(leaf_idx.item())]
        leaf_conf_val = leaf_conf.item()

        # If it's not a leaf, or confidence is too low -> Reject immediately
        if leaf_pred == "NOT_LEAF" or leaf_conf_val < LEAF_THRESHOLD:
            return leaf_pred, leaf_conf_val, "N/A", "N/A", "NOT_LEAF"

        # --- STAGE 2: DISEASE CLASSIFIER ---
        disease_outputs = disease_model(tensor)
        disease_probs = torch.softmax(disease_outputs[0], dim=0)
        disease_conf, disease_idx = torch.max(disease_probs, 0)

        disease_pred = disease_map[str(disease_idx.item())]
        disease_conf_val = disease_conf.item()

        # If disease confidence is too low -> Uncertain
        if disease_conf_val < DISEASE_THRESHOLD:
            return leaf_pred, leaf_conf_val, disease_pred, disease_conf_val, "UNCERTAIN"

        # Determine if it's healthy or diseased based on the class label
        if "healthy" in disease_pred.lower():
            final_status = "LEAF + HEALTHY"
        else:
            final_status = "LEAF + DISEASE"

        return leaf_pred, leaf_conf_val, disease_pred, disease_conf_val, final_status

# ============================================================================
# MAIN SCRIPT EXECUTION
# ============================================================================
def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_model.py <path_to_image_or_folder>")
        sys.exit(1)

    input_path = sys.argv[1]

    # Verify all files exist
    required_files = [LEAF_MODEL_PATH, LEAF_MAP_PATH, DISEASE_MODEL_PATH, DISEASE_MAP_PATH]
    for path in required_files:
        if not os.path.exists(path):
            print(f"Error: Required file missing -> {path}")
            sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Mappings
    try:
        leaf_map = load_mapping(LEAF_MAP_PATH)
        disease_map = load_mapping(DISEASE_MAP_PATH)
    except Exception as e:
        print(f"Mapping Error: {e}")
        sys.exit(1)

    # Load Models
    print("Loading Leaf and Disease models...")
    try:
        leaf_model = load_custom_model(LEAF_MODEL_PATH, 2, device)
        disease_model = load_custom_model(DISEASE_MODEL_PATH, 38, device)
        transform = get_transform()
    except Exception as e:
        print(f"Model Loading Error: {e}")
        sys.exit(1)

    # --- BATCH TESTING (FOLDER) ---
    if os.path.isdir(input_path):
        print(f"\nTesting directory: {input_path}")
        print(f"{'Filename':<30} | {'Leaf Pred':<10} | {'Leaf Conf':<9} | {'Disease Pred':<30} | {'Dis Conf':<8} | {'Final Status'}")
        print("-" * 115)
        
        valid_extensions = ('.png', '.jpg', '.jpeg')
        files = [f for f in os.listdir(input_path) if f.lower().endswith(valid_extensions)]
        
        if not files:
            print("No valid images found in the directory.")
            return

        for filename in sorted(files):
            img_path = os.path.join(input_path, filename)
            l_pred, l_conf, d_pred, d_conf, status = run_pipeline(
                img_path, leaf_model, disease_model, leaf_map, disease_map, transform, device
            )
            
            if l_pred is None:
                print(f"{filename[:28]:<30} | ERROR: {status}")
            else:
                l_conf_str = f"{l_conf:.4f}"
                d_conf_str = f"{d_conf:.4f}" if d_conf != "N/A" else "N/A"
                print(f"{filename[:28]:<30} | {l_pred:<10} | {l_conf_str:<9} | {d_pred[:28]:<30} | {d_conf_str:<8} | {status}")

    # --- SINGLE IMAGE TESTING ---
    else:
        l_pred, l_conf, d_pred, d_conf, status = run_pipeline(
            input_path, leaf_model, disease_model, leaf_map, disease_map, transform, device
        )
        
        if l_pred is None:
            print(f"FAILED TO PREDICT:\n{status}")
            return

        print("\n==============================")
        print("TRIAGE PIPELINE RESULT")
        print("==============================")
        print(f"Filename: {os.path.basename(input_path)}")
        print(f"Leaf Prediction: {l_pred}")
        print(f"Leaf Confidence: {l_conf:.4f}")
        
        if d_pred != "N/A":
            print(f"Disease Prediction: {d_pred}")
            print(f"Disease Confidence: {d_conf:.4f}")
        else:
            print("Disease Prediction: Skipped (Not a clear leaf)")
            print("Disease Confidence: N/A")
            
        print(f"Final Status: {status}")
        print("==============================\n")

if __name__ == "__main__":
    main()