"""
train.py
--------
Trains ONE model (custom_cnn / mobilenet_v2 / resnet50 / efficientnet_b0)
and saves:
  outputs/models/<model_name>.pt        - best checkpoint (state_dict)
  outputs/logs/<model_name>_history.json - per-epoch loss/accuracy

Usage:
    python -m src.train --model custom_cnn --epochs 8
    python -m src.train --model resnet50 --epochs 10 --no_pretrained
    python -m src.train --model all               # trains every model in sequence
"""

import os
import sys
import json
import time
import argparse

import torch
import torch.nn as nn
from torch.optim import Adam

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from src.dataset import get_dataloaders  # noqa: E402
from src.models.transfer_models import build_model  # noqa: E402


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def train_model(model_name, epochs=None, pretrained=True, batch_size=None):
    epochs = epochs or config.NUM_EPOCHS
    device = torch.device(config.DEVICE)
    print(f"\n{'='*60}\nTraining: {model_name}  |  device={device}  |  epochs={epochs}\n{'='*60}")

    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size)

    model = build_model(model_name, config.NUM_CLASSES, pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_path = os.path.join(config.MODELS_DIR, f"{model_name}.pt")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d}/{epochs}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}  ({dt:.1f}s)")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_name": model_name,
                "state_dict": model.state_dict(),
                "class_names": config.CLASS_NAMES,
                "val_acc": val_acc,
            }, best_path)

    history_path = os.path.join(config.LOGS_DIR, f"{model_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"[OK] Best val_acc={best_val_acc:.3f}  -> saved {best_path}")
    return best_path, best_val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="custom_cnn",
                         choices=config.MODEL_NAMES + ["all"])
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--no_pretrained", action="store_true",
                         help="Train transfer-learning models from random init "
                              "(use if you have no internet access to download ImageNet weights).")
    args = parser.parse_args()

    models_to_run = config.MODEL_NAMES if args.model == "all" else [args.model]
    for m in models_to_run:
        pretrained = (m != "custom_cnn") and (not args.no_pretrained)
        train_model(m, epochs=args.epochs, pretrained=pretrained, batch_size=args.batch_size)
