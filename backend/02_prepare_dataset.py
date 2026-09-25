import os
import io
import glob
import json
import hashlib
from collections import defaultdict
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

TARGET_38_CLASSES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

WILD_TO_38_MAP = {
    "apple_black_rot": "Apple___Black_rot",
    "apple_rust": "Apple___Cedar_apple_rust",
    "apple_scab": "Apple___Apple_scab",
    "bell_pepper_bacterial_spot": "Pepper,_bell___Bacterial_spot",
    "cherry_powdery_mildew": "Cherry_(including_sour)___Powdery_mildew",
    "corn_gray_leaf_spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "corn_northern_leaf_blight": "Corn_(maize)___Northern_Leaf_Blight",
    "corn_rust": "Corn_(maize)___Common_rust_",
    "grape_black_rot": "Grape___Black_rot",
    "potato_early_blight": "Potato___Early_blight",
    "potato_late_blight": "Potato___Late_blight",
    "squash_powdery_mildew": "Squash___Powdery_mildew",
    "strawberry_leaf_scorch": "Strawberry___Leaf_scorch",
    "tomato_bacterial_leaf_spot": "Tomato___Bacterial_spot",
    "tomato_early_blight": "Tomato___Early_blight",
    "tomato_late_blight": "Tomato___Late_blight",
    "tomato_leaf_mold": "Tomato___Leaf_Mold",
    "tomato_septoria_leaf_spot": "Tomato___Septoria_leaf_spot",
    "tomato_yellow_leaf_curl_virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "tomato_mosaic_virus": "Tomato___Tomato_mosaic_virus",
}

def find_dataset_root():
    candidates = [r"D:\Hackathon\dataset", r"D:\dataset", r"D:\\"]
    for path in candidates:
        if os.path.exists(os.path.join(path, "train-00000-of-00002.parquet")):
            return path
    return r"D:\Hackathon\dataset"

def get_hash(byte_data):
    return hashlib.md5(byte_data).hexdigest()

def match_wild_class(file_path):
    filename = os.path.basename(file_path).lower()
    folder = os.path.basename(os.path.dirname(file_path)).lower()
    for prefix, target_class in WILD_TO_38_MAP.items():
        if prefix in filename or prefix in folder:
            return target_class
    for target in TARGET_38_CLASSES:
        if folder == target.lower():
            return target
    return None

def main():
    dataset_dir = find_dataset_root()
    base_out = r"D:\Hackathon\dataset_processed"
    disease_out = os.path.join(base_out, "disease")
    leaf_out = os.path.join(base_out, "leaf_classifier")

    seen_hashes = set()
    disease_records = []
    unmapped_records = []
    corrupt_count = 0
    duplicate_count = 0

    print("==================================================")
    print("STEP 1: Processing Parquet Files (PlantVillage)")
    print("==================================================")
    parquet_files = glob.glob(os.path.join(dataset_dir, "*.parquet"))
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        for _, row in tqdm(df.iterrows(), total=len(df), desc=os.path.basename(pf)):
            label = row.get("class_label")
            if label not in TARGET_38_CLASSES:
                unmapped_records.append(label)
                continue

            raw_img = row.get("image")
            img_bytes = raw_img["bytes"] if isinstance(raw_img, dict) and "bytes" in raw_img else raw_img
            if not img_bytes:
                corrupt_count += 1
                continue

            h = get_hash(img_bytes)
            if h in seen_hashes:
                duplicate_count += 1
                continue
            seen_hashes.add(h)

            try:
                img = Image.open(io.BytesIO(img_bytes))
                img.verify()
                disease_records.append({"label": label, "bytes": img_bytes, "hash": h})
            except Exception:
                corrupt_count += 1

    print("\n==================================================")
    print("STEP 2: Processing All Real-World Datasets")
    print("==================================================")
    wild_folders = ["plantseg", "plantwild", "plantwild_v2"]
    
    for wild_name in wild_folders:
        wild_dir = os.path.join(dataset_dir, wild_name)
        if not os.path.exists(wild_dir):
            continue
            
        all_wild_files = glob.glob(os.path.join(wild_dir, "**", "*.*"), recursive=True)
        all_wild_files = [f for f in all_wild_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for file_path in tqdm(all_wild_files, desc=f"Parsing {wild_name}"):
            target_class = match_wild_class(file_path)

            try:
                with open(file_path, "rb") as f:
                    img_bytes = f.read()
                
                h = get_hash(img_bytes)
                if h in seen_hashes:
                    duplicate_count += 1
                    continue
                seen_hashes.add(h)

                img = Image.open(io.BytesIO(img_bytes))
                img.verify()

                if target_class:
                    disease_records.append({"label": target_class, "bytes": img_bytes, "hash": h})
                else:
                    unmapped_records.append(os.path.basename(file_path))
            except Exception:
                corrupt_count += 1

    print(f"\nValid Disease Images Collected: {len(disease_records):,}")
    print(f"Duplicates Removed: {duplicate_count:,}")

    print("\n==================================================")
    print("STEP 3: Stratified 70/15/15 Splitting (Disease Model)")
    print("==================================================")
    labels = [r["label"] for r in disease_records]
    train_val, test_records = train_test_split(disease_records, test_size=0.15, stratify=labels, random_state=42)
    train_val_labels = [r["label"] for r in train_val]
    train_records, val_records = train_test_split(train_val, test_size=0.17647, stratify=train_val_labels, random_state=42)

    def write_split(records, split_name):
        for r in tqdm(records, desc=f"Writing disease/{split_name}"):
            folder = os.path.join(disease_out, split_name, r["label"])
            os.makedirs(folder, exist_ok=True)
            out_file = os.path.join(folder, f"{r['hash']}.jpg")
            with Image.open(io.BytesIO(r["bytes"])).convert("RGB") as im:
                im.save(out_file, "JPEG", quality=95)

    write_split(train_records, "train")
    write_split(val_records, "val")
    write_split(test_records, "test")

    print("\n==================================================")
    print("STEP 4: Constructing LEAF / NOT_LEAF Dataset Base")
    print("==================================================")
    splits_dict = {"train": train_records, "val": val_records, "test": test_records}
    
    for split_name, recs in splits_dict.items():
        leaf_dir = os.path.join(leaf_out, split_name, "LEAF")
        not_leaf_dir = os.path.join(leaf_out, split_name, "NOT_LEAF")
        os.makedirs(leaf_dir, exist_ok=True)
        os.makedirs(not_leaf_dir, exist_ok=True)

        selected_leaf = recs[:min(len(recs), 6000)]
        for r in tqdm(selected_leaf, desc=f"Writing leaf_classifier/{split_name}/LEAF"):
            with Image.open(io.BytesIO(r["bytes"])).convert("RGB") as im:
                im.save(os.path.join(leaf_dir, f"{r['hash']}.jpg"), "JPEG", quality=90)

    class_counts = defaultdict(int)
    for r in disease_records:
        class_counts[r["label"]] += 1

    report = {
        "total_images_processed": len(disease_records),
        "train_count": len(train_records),
        "val_count": len(val_records),
        "test_count": len(test_records),
        "duplicates_removed": duplicate_count,
        "corrupt_images_skipped": corrupt_count,
        "unmapped_images_count": len(unmapped_records),
        "per_class_counts": class_counts
    }

    with open(os.path.join(base_out, "dataset_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("\n==================================================")
    print("IMPORTANT ACTION REQUIRED FOR LEAF CLASSIFIER")
    print("==================================================")
    print(f"The directories for NOT_LEAF have been created at:\n{leaf_out}\\*\\NOT_LEAF")
    print("Please manually copy REAL non-leaf images (cars, people, buildings, sky, etc.)")
    print("into these train/val/test NOT_LEAF folders before running 03_train_leaf_classifier.py.")
    print("Do NOT use random synthetic noise.")

if __name__ == "__main__":
    main()