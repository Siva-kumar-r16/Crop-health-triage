import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
from torchvision.models import MobileNet_V3_Small_Weights
from PIL import Image, ImageFile
from tqdm import tqdm
import multiprocessing

# BULLETPROOF FIX 1: Force PIL to load partially corrupted images without throwing exceptions
ImageFile.LOAD_TRUNCATED_IMAGES = True

# BULLETPROOF FIX 2: Safe loader that never crashes, even if an image is completely unreadable
def safe_pil_loader(path):
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except Exception:
        # If the user-provided image is completely broken, return a blank black image
        # This prevents the entire training loop from crashing.
        return Image.new('RGB', (224, 224), (0, 0, 0))

def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Binary Leaf Classifier Training on Device: {device}")

    # Standard, safe PyTorch settings
    num_workers = 0 
    batch_size = 32

    data_dir = r"D:\Hackathon\dataset_processed\leaf_classifier"

    not_leaf_train = os.path.join(data_dir, "train", "NOT_LEAF")
    if not os.path.exists(not_leaf_train) or len(os.listdir(not_leaf_train)) == 0:
        print(f"\nERROR: The NOT_LEAF directory is empty: {not_leaf_train}")
        print("Please add real non-leaf images (people, cars, buildings) before training.")
        return

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("Mapping datasets from disk using Safe Loader...")
    # Inject the safe loader to prevent PIL decoder crashes
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transforms, loader=safe_pil_loader)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_transforms, loader=safe_pil_loader)

    os.makedirs(r"D:\Hackathon\model", exist_ok=True)
    with open(r"D:\Hackathon\model\leaf_class_names.json", "w") as f:
        mapping = {i: cls_name for cls_name, i in train_ds.class_to_idx.items()}
        json.dump(mapping, f, indent=4)

    # Standard dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)

    best_val_acc = 0.0
    epochs = 8

    print(f"\nStarting Bulletproof Standard PyTorch Training...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Training Loop
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

        train_acc = correct / total
        train_loss = running_loss / total

        # Validation Loop (Now with a progress bar so you know it's not frozen)
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val  ]"):
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_correct += preds.eq(targets).sum().item()
                val_total += targets.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1} Results -> Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%\n")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), r"D:\Hackathon\model\leaf_classifier.pth")
            print(">>> Saved best leaf classifier weights!\n")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()