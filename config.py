"""
config.py
---------
Central configuration for the Bank Document Classification project.
All paths, hyperparameters and class names live here so every script
(data generation, training, evaluation, API) stays in sync.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")
LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")
GRADCAM_DIR = os.path.join(OUTPUTS_DIR, "gradcam")

# ---------------------------------------------------------------------------
# Document classes (8 banking document types)
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "cheque",
    "bank_statement",
    "card_document",
    "deposit_slip",
    "withdrawal_slip",
    "account_opening_form",
    "kyc_document",
    "other_financial_document",
]
NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------------------------------------------------------
# Image / training hyperparameters
# ---------------------------------------------------------------------------
IMAGE_SIZE = 224          # standard input size for CNN + transfer-learning backbones
BATCH_SIZE = 16
NUM_EPOCHS = 8
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# Models to train / compare. Keys are used everywhere (filenames, API, reports).
MODEL_NAMES = ["custom_cnn", "mobilenet_v2", "resnet50", "efficientnet_b0"]

# Device
import torch  # noqa: E402
DEVICE = "cuda" if (torch.cuda.is_available() and os.environ.get("FORCE_CPU") != "1") else "cpu"

# ---------------------------------------------------------------------------
# Synthetic data generator settings (used only when no real dataset exists)
# ---------------------------------------------------------------------------
SYNTHETIC_IMAGES_PER_CLASS = 60
SYNTHETIC_IMAGE_SIZE = (900, 500)
