"""
gradcam.py
----------
Grad-CAM implementation (works for custom_cnn, mobilenet_v2, resnet50,
efficientnet_b0) using forward/backward hooks on the last conv layer.
Produces a heatmap overlay showing WHY the model predicted a given class
(e.g. highlighting the signature/MICR/layout region on a cheque).

Usage (standalone):
    python -m src.gradcam --image path/to/doc.png --model resnet50
"""

import os
import sys
import argparse

import cv2
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from src.models.transfer_models import build_model, get_gradcam_target_layer  # noqa: E402
from src.dataset import EVAL_TRANSFORMS  # noqa: E402
from PIL import Image  # noqa: E402


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        gradients = self.gradients[0]          # (C, H, W)
        activations = self.activations[0]      # (C, H, W)
        weights = gradients.mean(dim=(1, 2))    # (C,)

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy(), class_idx, torch.softmax(output, dim=1)[0, class_idx].item()


def overlay_heatmap(original_bgr, cam, alpha=0.45):
    h, w = original_bgr.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_bgr, 1 - alpha, heatmap, alpha, 0)
    return overlay


def run_gradcam(image_path, model_name, save_path=None):
    device = torch.device(config.DEVICE)
    ckpt_path = os.path.join(config.MODELS_DIR, f"{model_name}.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)

    model = build_model(model_name, config.NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    target_layer = get_gradcam_target_layer(model_name, model)
    cam_tool = GradCAM(model, target_layer)

    pil_img = Image.open(image_path).convert("RGB")
    input_tensor = EVAL_TRANSFORMS(pil_img).unsqueeze(0).to(device)

    cam, class_idx, confidence = cam_tool.generate(input_tensor)

    original_bgr = cv2.cvtColor(np.array(pil_img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))), cv2.COLOR_RGB2BGR)
    overlay = overlay_heatmap(original_bgr, cam)

    save_path = save_path or os.path.join(
        config.GRADCAM_DIR, f"gradcam_{model_name}_{os.path.basename(image_path)}"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, overlay)

    predicted_class = config.CLASS_NAMES[class_idx]
    print(f"Prediction: {predicted_class}  (confidence={confidence:.3f})")
    print(f"Grad-CAM saved -> {save_path}")
    return save_path, predicted_class, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="custom_cnn", choices=config.MODEL_NAMES)
    args = parser.parse_args()
    run_gradcam(args.image, args.model)
