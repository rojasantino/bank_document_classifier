"""
custom_cnn.py
-------------
A baseline convolutional network built from scratch (no pretrained weights).
Used as the "CNN" row in the model-comparison table.
"""

import torch
import torch.nn as nn


class CustomCNN(nn.Module):
    def __init__(self, num_classes, image_size=224):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),  # /2

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),  # /4

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),  # /8

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),  # /16
        )

        feat_size = image_size // 16
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_gradcam_target_layer(self):
        """Last conv layer -- used by Grad-CAM."""
        return self.features[-3]  # the last Conv2d before final pool/BN/ReLU block


if __name__ == "__main__":
    model = CustomCNN(num_classes=8)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print("Output shape:", out.shape)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_params:,}")
