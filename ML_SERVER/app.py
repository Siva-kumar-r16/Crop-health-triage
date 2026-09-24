import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageFile
import gradio as gr
import spaces

# Allow loading of truncated/corrupted images without instantly crashing
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ============================================================================
# CONFIGURATION THRESHOLDS & PATHS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

LEAF_MODEL_PATH = os.path.join(MODEL_DIR, "leaf_classifier.pth")
LEAF_MAP_PATH = os.path.join(MODEL_DIR, "leaf_class_names.json")
DISEASE_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
DISEASE_MAP_PATH = os.path.join(MODEL_DIR, "class_names.json")
DISEASE_INFO_PATH = os.path.join(BASE_DIR, "disease_info.json")

LEAF_CONFIDENCE_THRESHOLD = 0.70
DISEASE_CONFIDENCE_THRESHOLD = 0.60

# ============================================================================
# HELPERS
# ============================================================================
def load_json_mapping(filepath):
    if not os.path.exists(filepath):
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Ensure keys are integers if they represent class indices
    if all(k.isdigit() for k in data.keys()):
        return {int(k): v for k, v in data.items()}
    return data

def load_mobile_net(weights_path, num_classes):
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    # Load onto CPU globally to prevent persistent VRAM allocation
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model

def parse_label(class_label):
    parts = class_label.split("___")
    plant = parts[0].replace("_", " ").strip()
    disease = parts[1].replace("_", " ").strip() if len(parts) > 1 else None
    
    if disease and disease.lower() == "healthy":
        return plant, "Healthy", "Healthy"
    return plant, disease, "Diseased"

def get_conf_level(score):
    if score >= 0.80: return "High"
    if score >= 0.60: return "Medium"
    return "Low"

def format_list(items):
    if isinstance(items, list):
        return "\n".join([f"- {item}" for item in items])
    return str(items)

def format_top3(probs, indices, disease_classes):
    top3_md = ""
    for p, idx in zip(probs, indices):
        lbl = disease_classes[idx]
        plant, dis, _ = parse_label(lbl)
        dis_display = dis if dis else "Healthy"
        top3_md += f"- **{plant} - {dis_display}**: {p*100:.2f}%\n"
    return top3_md

# ============================================================================
# GLOBAL INITIALIZATION (CPU)
# ============================================================================
leaf_classes = load_json_mapping(LEAF_MAP_PATH)
disease_classes = load_json_mapping(DISEASE_MAP_PATH)
disease_info_db = load_json_mapping(DISEASE_INFO_PATH)

leaf_model = load_mobile_net(LEAF_MODEL_PATH, 2)
disease_model = load_mobile_net(DISEASE_MODEL_PATH, 38)

inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================================================
# INFERENCE LOGIC (ZEROGPU)
# ============================================================================
@spaces.GPU
def predict(image):
    if image is None:
        return "⚠️ Please upload an image."

    # ZeroGPU safely intercepts and assigns devices dynamically
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move models to the active execution device dynamically
    leaf_model.to(device)
    disease_model.to(device)

    try:
        img_rgb = image.convert("RGB")
        tensor = inference_transforms(img_rgb).unsqueeze(0).to(device)
    except Exception as e:
        return f"### ⚠️ Error processing image\n{str(e)}"

    with torch.no_grad():
        # --- STAGE 1: LEAF CLASSIFIER ---
        l_outputs = leaf_model(tensor)
        l_probs = torch.nn.functional.softmax(l_outputs[0], dim=0)
        l_conf, l_idx = torch.max(l_probs, 0)
        
        leaf_pred = leaf_classes[l_idx.item()]
        leaf_conf_val = l_conf.item()

        if leaf_pred == "NOT_LEAF" or leaf_conf_val < LEAF_CONFIDENCE_THRESHOLD:
            return (
                f"### 🛑 Status: Rejected\n\n"
                f"**Message:** Please upload a clear photo of a crop leaf.\n\n"
                f"**Leaf Prediction:** {leaf_pred}\n"
                f"**Leaf Confidence:** {leaf_conf_val * 100:.2f}%"
            )

        # --- STAGE 2: DISEASE CLASSIFIER ---
        d_outputs = disease_model(tensor)
        d_probs = torch.nn.functional.softmax(d_outputs[0], dim=0)
        
        top_k_prob, top_k_idx = torch.topk(d_probs, 3)
        top_probs_list = top_k_prob.cpu().numpy().tolist()
        top_indices_list = top_k_idx.cpu().numpy().tolist()

        best_prob = top_probs_list[0]
        best_idx = top_indices_list[0]
        best_label = disease_classes[best_idx]
        
        top3_markdown = format_top3(top_probs_list, top_indices_list, disease_classes)

        if best_prob < DISEASE_CONFIDENCE_THRESHOLD:
            return (
                f"### ⚠️ Status: Uncertain\n\n"
                f"**Message:** The image is not clear enough for a reliable prediction. Please take another clear photo.\n\n"
                f"**Leaf Confidence:** {leaf_conf_val * 100:.2f}%\n"
                f"**Disease Confidence:** {best_prob * 100:.2f}%\n\n"
                f"#### 📊 Top 3 Predictions\n{top3_markdown}"
            )

        # --- FINAL SUCCESS RESPONSE ---
        plant, disease, status = parse_label(best_label)
        disease_display = disease if disease else "Healthy"
        
        info = disease_info_db.get(best_label, {})
        
        return (
            f"### ✅ Status: {status}\n\n"
            f"**Plant:** {plant}\n"
            f"**Disease:** {disease_display}\n"
            f"**Confidence:** {best_prob * 100:.2f}% ({get_conf_level(best_prob)})\n\n"
            f"---\n\n"
            f"#### 📖 Disease Information\n"
            f"**Description:**\n{info.get('description', 'Information not available.')}\n\n"
            f"**Symptoms:**\n{format_list(info.get('symptoms', ['Information not available.']))}\n\n"
            f"**Causes:**\n{format_list(info.get('causes', ['Information not available.']))}\n\n"
            f"**Prevention:**\n{format_list(info.get('prevention', ['Information not available.']))}\n\n"
            f"**Recommended Action:**\n{format_list(info.get('recommended_action', ['Information not available.']))}\n\n"
            f"---\n\n"
            f"#### 📊 Top 3 Predictions\n{top3_markdown}"
        )

# ============================================================================
# GRADIO INTERFACE
# ============================================================================
with gr.Blocks(title="AI Crop Disease Triage", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌱 AI Crop Disease Triage\nUpload a clear photo of a crop leaf to diagnose its health status.")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Leaf Image")
            submit_btn = gr.Button("Diagnose", variant="primary")
        with gr.Column():
            output_ui = gr.Markdown(label="Diagnosis Report")
            
    submit_btn.click(fn=predict, inputs=image_input, outputs=output_ui)

if __name__ == "__main__":
    demo.launch()
