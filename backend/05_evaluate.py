import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import multiprocessing

def evaluate_checkpoint(model_path, test_loader, num_classes, device):
    if not os.path.exists(model_path):
        return None, None
    model = models.mobilenet_v3_small()
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc=f"Evaluating {os.path.basename(model_path)}"):
            inputs = inputs.to(device)
            # Standard forward pass (No AMP)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    return all_targets, all_preds

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating Models on {device} (Standard Mode)...")

    data_dir = r"D:\Hackathon\dataset_processed\disease\test"
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_ds = datasets.ImageFolder(data_dir, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
    class_names = test_ds.classes

    new_model_path = r"D:\Hackathon\model\best_model.pth"
    old_model_path = r"D:\Hackathon\model\best_model_old.pth"

    y_true, y_pred_new = evaluate_checkpoint(new_model_path, test_loader, 38, device)
    new_report = classification_report(y_true, y_pred_new, target_names=class_names, output_dict=True)
    new_acc = accuracy_score(y_true, y_pred_new)

    print("\n==================================================")
    print(f"NEW MODEL TEST ACCURACY: {new_acc*100:.2f}%")
    print("Note: This reflects test-split accuracy, not full real-world robustness.")
    print("==================================================")

    _, y_pred_old = evaluate_checkpoint(old_model_path, test_loader, 38, device)
    if y_pred_old is not None:
        old_acc = accuracy_score(y_true, y_pred_old)
        print(f"OLD MODEL TEST ACCURACY: {old_acc*100:.2f}%")
        print(f"ROBUSTNESS / ACCURACY GAIN: +{(new_acc - old_acc)*100:.2f}%")

    os.makedirs(r"D:\Hackathon\results", exist_ok=True)
    with open(r"D:\Hackathon\results\evaluation_report.json", "w") as f:
        json.dump(new_report, f, indent=4)

    cm = confusion_matrix(y_true, y_pred_new)
    plt.figure(figsize=(22, 18))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('38-Class Disease Model Test Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(r"D:\Hackathon\results\confusion_matrix.png")
    print("Saved evaluation_report.json and confusion_matrix.png in D:\\Hackathon\\results\\")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()