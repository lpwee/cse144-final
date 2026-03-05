import csv
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import datasets, models, transforms

DATA_DIR = Path(__file__).parent / "ucsc-cse-144-winter-2026-final-project"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


NUM_CLASSES = 100


def get_model(num_classes=NUM_CLASSES):
    # --- EfficientNet-B0 (default) ---
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features  # type: ignore[union-attr]
    model.classifier[1] = nn.Linear(in_features, num_classes)

    # --- ResNet-50 ---
    # model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # model.fc = nn.Linear(model.fc.in_features, num_classes)

    # --- ViT-B/16 ---
    # model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    # model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    return model


def predict_test(model, device="cpu"):
    model.eval()
    model.to(device)
    test_dir = DATA_DIR / "test"
    image_files = sorted(test_dir.glob("*.jpg"), key=lambda p: int(p.stem))

    predictions = []
    with torch.no_grad():
        for img_path in image_files:
            img = Image.open(img_path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            pred = model(x).argmax(dim=1).item()
            predictions.append((img_path.name, pred))

    out_path = Path(__file__).parent / "submission.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Label"])
        writer.writerows(predictions)

    print(f"Saved {len(predictions)} predictions to {out_path}")
    return out_path


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=transform)
    print(f"Train: {len(train_dataset)} images, {len(train_dataset.classes)} classes")

    model = get_model()
    print(f"Model: EfficientNet-B0, output classes: {NUM_CLASSES}")

    # TODO: finetune model on train_dataset

    predict_test(model, device)


if __name__ == "__main__":
    main()
