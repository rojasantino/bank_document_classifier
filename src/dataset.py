"""
dataset.py
----------
Builds PyTorch datasets/dataloaders from data/processed (or data/raw if
processed doesn't exist), with:
  - train/val/test split (stratified by class)
  - data augmentation on the training split
  - class-balanced sampling to handle any class imbalance
"""

import os
import sys
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

random.seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
torch.manual_seed(config.RANDOM_SEED)


TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.RandomRotation(5),
    transforms.RandomAffine(degrees=0, translate=(0.03, 0.03)),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.RandomHorizontalFlip(p=0.0),  # documents are orientation-sensitive; kept off by default
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class BankDocumentDataset(Dataset):
    def __init__(self, samples, transform):
        """samples: list of (filepath, label_idx) tuples."""
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label


def _collect_samples(data_dir):
    samples = []
    for class_idx, class_name in enumerate(config.CLASS_NAMES):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                samples.append((os.path.join(class_dir, fname), class_idx))
    return samples


def get_data_dir():
    """Prefer preprocessed data; fall back to raw if preprocessing hasn't run yet."""
    if os.path.isdir(config.DATA_PROCESSED_DIR) and _collect_samples(config.DATA_PROCESSED_DIR):
        return config.DATA_PROCESSED_DIR
    return config.DATA_RAW_DIR


def build_splits(data_dir=None):
    data_dir = data_dir or get_data_dir()
    samples = _collect_samples(data_dir)
    if not samples:
        raise RuntimeError(
            f"No images found in {data_dir}. Run data/generate_synthetic_data.py "
            "or add your own images under data/raw/<class_name>/."
        )

    paths = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=(config.VAL_SPLIT + config.TEST_SPLIT),
        stratify=labels, random_state=config.RANDOM_SEED,
    )
    rel_test_size = config.TEST_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=rel_test_size,
        stratify=temp_labels, random_state=config.RANDOM_SEED,
    )

    train_samples = list(zip(train_paths, train_labels))
    val_samples = list(zip(val_paths, val_labels))
    test_samples = list(zip(test_paths, test_labels))
    return train_samples, val_samples, test_samples


def make_balanced_sampler(train_samples):
    labels = [lbl for _, lbl in train_samples]
    class_counts = np.bincount(labels, minlength=config.NUM_CLASSES)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = [class_weights[lbl] for lbl in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def get_dataloaders(batch_size=None, data_dir=None):
    batch_size = batch_size or config.BATCH_SIZE
    train_samples, val_samples, test_samples = build_splits(data_dir)

    train_ds = BankDocumentDataset(train_samples, TRAIN_TRANSFORMS)
    val_ds = BankDocumentDataset(val_samples, EVAL_TRANSFORMS)
    test_ds = BankDocumentDataset(test_samples, EVAL_TRANSFORMS)

    sampler = make_balanced_sampler(train_samples)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"[OK] Dataset -> train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    get_dataloaders()
