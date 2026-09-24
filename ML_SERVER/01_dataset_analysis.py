import os
import glob
import json
import pandas as pd
from collections import defaultdict

def find_dataset_root():
    candidates = [r"D:\Hackathon\dataset", r"D:\dataset", r"D:\\"]
    for path in candidates:
        if os.path.exists(os.path.join(path, "train-00000-of-00002.parquet")):
            return path
    return r"D:\Hackathon\dataset"

def main():
    dataset_dir = find_dataset_root()
    print(f"==================================================")
    print(f"Scanning Dataset Directory: {dataset_dir}")
    print(f"==================================================")

    analysis_report = {
        "dataset_dir": dataset_dir,
        "parquet_datasets": {},
        "wild_datasets": {},
    }

    # 1. Inspect Parquet Files
    parquet_files = glob.glob(os.path.join(dataset_dir, "*.parquet"))
    print(f"\n[1] Found {len(parquet_files)} Parquet shards:")
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
            rows = len(df)
            cols = list(df.columns)
            analysis_report["parquet_datasets"][os.path.basename(pf)] = {"rows": rows, "columns": cols}
            print(f"  - {os.path.basename(pf)}: {rows:,} rows")
        except Exception as e:
            print(f"  - Error reading {pf}: {e}")

    # 2. Inspect All Real-World (Wild) Folders
    wild_folders = ["plantseg", "plantwild", "plantwild_v2"]
    
    print("\n[2] Scanning Real-World 'Wild' Datasets:")
    for wild_name in wild_folders:
        wild_dir = os.path.join(dataset_dir, wild_name)
        if not os.path.exists(wild_dir):
            print(f"  - WARNING: {wild_name} directory not found at {wild_dir}")
            continue
            
        print(f"  - Found dataset: {wild_name}")
        image_files = glob.glob(os.path.join(wild_dir, "**", "*.*"), recursive=True)
        image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        class_samples = defaultdict(int)
        for f in image_files:
            filename = os.path.basename(f)
            folder = os.path.basename(os.path.dirname(f))
            if folder.lower() not in ["train", "val", "test", "images"]:
                class_key = folder
            else:
                parts = filename.split("_")
                class_key = "_".join([p for p in parts if not p.replace('.png', '').replace('.jpg', '').isdigit()])
            class_samples[class_key] += 1
            
        analysis_report["wild_datasets"][wild_name] = {
            "total_images": len(image_files),
            "detected_classes": len(class_samples),
            "sample_classes": dict(list(class_samples.items())[:10])
        }
        print(f"    Total Images: {len(image_files):,}")
        print(f"    Unique Class Patterns Found: {len(class_samples)}")

    os.makedirs(r"D:\Hackathon\dataset_processed", exist_ok=True)
    out_path = r"D:\Hackathon\dataset_processed\dataset_inventory.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(analysis_report, f, indent=4)

    print(f"\nDataset inspection report saved to: {out_path}")

if __name__ == "__main__":
    main()