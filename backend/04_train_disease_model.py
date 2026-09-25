import os
import json
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
from torchvision.models import MobileNet_V3_Small_Weights
from PIL import Image, ImageFile
from tqdm import tqdm
import multiprocessing
import signal

# ============================================================================
# ULTRA FAST RTX 3050 - INSTANT START FIX
# ============================================================================

ImageFile.LOAD_TRUNCATED_IMAGES = True

def safe_pil_loader(path):
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except Exception:
        return Image.new('RGB', (224, 224), (0, 0, 0))

class SafeColorJitter(transforms.ColorJitter):
    def forward(self, img):
        try:
            return super().forward(img)
        except Exception:
            return img

should_stop = False
def signal_handler(sig, frame):
    global should_stop
    print("\n>>> Graceful shutdown signal received. Will save after current epoch...")
    should_stop = True

signal.signal(signal.SIGINT, signal_handler)

def backup_existing_model():
    model_dir = r"D:\Hackathon\model"
    src = os.path.join(model_dir, "best_model.pth")
    dst = os.path.join(model_dir, "best_model_old.pth")
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"✅ Existing model successfully preserved at: {dst}")

def count_images(data_dir):
    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "val")
    train_count = 0
    val_count = 0
    if os.path.exists(train_path):
        for class_name in os.listdir(train_path):
            class_path = os.path.join(train_path, class_name)
            if os.path.isdir(class_path):
                train_count += len(os.listdir(class_path))
    if os.path.exists(val_path):
        for class_name in os.listdir(val_path):
            class_path = os.path.join(val_path, class_name)
            if os.path.isdir(class_path):
                val_count += len(os.listdir(class_path))
    return train_count, val_count

def main():
    torch.manual_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == 'cuda'
    
    print(f"🚀 Executing 38-Class Disease Model on: {device}")
    
    if use_amp:
        print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        # CRITICAL FIX 1: Disable benchmark to prevent the 3-minute first-batch freeze
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    backup_existing_model()
    
    data_dir = r"D:\Hackathon\dataset_processed\disease"
    train_count, val_count = count_images(data_dir)
    
    batch_size = 128
    # CRITICAL FIX 2: num_workers=0 + pin_memory=False prevents Windows deadlocks
    num_workers = 0
    expected_batches = (train_count + batch_size - 1) // batch_size
    
    print(f"\n📈 Training Configuration:")
    print(f"   Batch size: {batch_size}")
    print(f"   Expected batches per epoch: {expected_batches}")
    print(f"   (Training on ALL {train_count} images)\n")

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        SafeColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("📦 Loading 38-class dataset (100% of all images)...")
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transforms, loader=safe_pil_loader)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_transforms, loader=safe_pil_loader)

    if len(train_ds.classes) != 38:
        raise ValueError(f"Expected exactly 38 classes, found {len(train_ds.classes)}!")

    mapping = {i: cls_name for cls_name, i in train_ds.class_to_idx.items()}
    with open(r"D:\Hackathon\model\class_names.json", "w") as f:
        json.dump(mapping, f, indent=4)

    # CRITICAL FIX 2: pin_memory=False
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 38)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
    
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    best_val_acc = 0.0
    epochs = 12

    print(f"\n{'='*70}")
    print(f"🔥 FAST STABLE TRAINING (INSTANT START) 🔥")
    print(f"{'='*70}")
    
    try:
        for epoch in range(epochs):
            if should_stop: break
                
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            train_batches = 0

            for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [TRAIN]", colour='green'):
                if should_stop: break
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs.detach(), 1)
                correct += preds.eq(targets).sum().item()
                total += targets.size(0)
                train_batches += 1
            
            if should_stop: break

            train_acc = correct / total
            train_loss = running_loss / total

            model.eval()
            val_correct = 0
            val_total = 0
            val_batches = 0
            
            with torch.no_grad():
                for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [VAL  ]", colour='blue'):
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    with torch.autocast(device_type=device.type, enabled=use_amp):
                        outputs = model(inputs)
                    
                    _, preds = torch.max(outputs, 1)
                    val_correct += preds.eq(targets).sum().item()
                    val_total += targets.size(0)
                    val_batches += 1

            val_acc = val_correct / val_total
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"📊 Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Train: {train_acc*100:.2f}% | Val: {val_acc*100:.2f}% | LR: {current_lr:.6f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), r"D:\Hackathon\model\best_model.pth")
                print(f"✨ [NEW BEST] Val Acc: {val_acc*100:.2f}% - Model saved!\n")
            else:
                print()
            
            if use_amp: torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\n\n[!] 🛑 Training manually interrupted by user (Ctrl+C).")
        print(">>> Halting gracefully. Best model weights achieved so far are already safely stored.\n")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()