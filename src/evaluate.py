"""
evaluate.py
-----------
Evaluates every trained model on the held-out test split and produces
the model-comparison table (Accuracy / F1 / Inference time) plus a
confusion matrix image per model.

Usage:
    python -m src.evaluate            # evaluates every model that has a checkpoint
    python -m src.evaluate --model resnet50
"""

import os
import sys
import time
import json
import argparse

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from src.dataset import get_dataloaders  # noqa: E402
from src.models.transfer_models import build_model  # noqa: E402


@torch.no_grad()
def evaluate_model(model_name, test_loader, device):
    ckpt_path = os.path.join(config.MODELS_DIR, f"{model_name}.pt")
    if not os.path.exists(ckpt_path):
        print(f"[SKIP] No checkpoint for {model_name} at {ckpt_path}. Train it first.")
        return None

    checkpoint = torch.load(ckpt_path, map_location=device)
    model = build_model(model_name, config.NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    all_preds, all_labels = [], []
    total_time, total_images = 0.0, 0

    for images, labels in test_loader:
        images = images.to(device)
        t0 = time.time()
        outputs = model(images)
        total_time += (time.time() - t0)
        total_images += images.size(0)

        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    avg_inference_ms = (total_time / max(total_images, 1)) * 1000

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(config.NUM_CLASSES)))
    report = classification_report(
        all_labels, all_preds, target_names=config.CLASS_NAMES, zero_division=0
    )

    return {
        "model_name": model_name,
        "accuracy": acc,
        "f1_macro": f1,
        "avg_inference_ms": avg_inference_ms,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def save_confusion_matrix(cm, model_name):
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(config.NUM_CLASSES))
    ax.set_yticks(range(config.NUM_CLASSES))
    ax.set_xticklabels(config.CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(config.CLASS_NAMES, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=7)
    fig.colorbar(im)
    fig.tight_layout()
    out_path = os.path.join(config.REPORTS_DIR, f"confusion_matrix_{model_name}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def run_comparison(models_to_eval=None):
    device = torch.device(config.DEVICE)
    _, _, test_loader = get_dataloaders()

    models_to_eval = models_to_eval or config.MODEL_NAMES
    results = []
    for model_name in models_to_eval:
        res = evaluate_model(model_name, test_loader, device)
        if res is None:
            continue
        cm_path = save_confusion_matrix(res["confusion_matrix"], model_name)
        print(f"\n[{model_name}]  accuracy={res['accuracy']:.3f}  "
              f"f1_macro={res['f1_macro']:.3f}  "
              f"avg_inference={res['avg_inference_ms']:.1f} ms")
        print(res["classification_report"])
        print(f"Confusion matrix saved -> {cm_path}")
        results.append(res)

    # Comparison table (markdown + json)
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    table_lines = ["| Model | Accuracy | F1 (macro) | Inference Time (ms/img) |",
                   "|---|---|---|---|"]
    json_summary = []
    for r in results:
        table_lines.append(
            f"| {r['model_name']} | {r['accuracy']:.3f} | {r['f1_macro']:.3f} | {r['avg_inference_ms']:.2f} |"
        )
        json_summary.append({
            "model": r["model_name"],
            "accuracy": r["accuracy"],
            "f1_macro": r["f1_macro"],
            "avg_inference_ms": r["avg_inference_ms"],
        })

    md_path = os.path.join(config.REPORTS_DIR, "model_comparison.md")
    with open(md_path, "w") as f:
        f.write("# Model Comparison\n\n")
        f.write("\n".join(table_lines))
        f.write("\n")

    json_path = os.path.join(config.REPORTS_DIR, "model_comparison.json")
    with open(json_path, "w") as f:
        json.dump(json_summary, f, indent=2)

    print(f"\n[OK] Comparison table saved -> {md_path}")
    print("\n".join(table_lines))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                         help="Evaluate a single model instead of all.")
    args = parser.parse_args()
    run_comparison([args.model] if args.model else None)
