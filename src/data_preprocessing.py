"""
data_preprocessing.py
----------------------
Practical OpenCV preprocessing steps for scanned bank-document images:
  1. Read + convert to consistent color format
  2. Denoise
  3. Deskew (rotation correction)
  4. Auto-crop to document boundary
  5. Resize + normalize for model input

These are used both offline (building data/processed/) and online
(inside the FastAPI /predict endpoint) so behaviour matches exactly.
"""

import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402


def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def denoise(img):
    return cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)


def deskew(img):
    """Estimate and correct rotation using minAreaRect on thresholded text/ink pixels."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] < 20:
        return img  # not enough signal to estimate angle safely

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Ignore near-zero corrections and implausible angles
    if abs(angle) < 0.3 or abs(angle) > 15:
        return img

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def auto_crop(img, margin_ratio=0.01):
    """Crop away uniform border/background around the document."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img

    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    H, W = img.shape[:2]
    mx, my = int(W * margin_ratio), int(H * margin_ratio)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(W, x + w + mx), min(H, y + h + my)

    if (x1 - x0) < 30 or (y1 - y0) < 30:
        return img  # degenerate crop, bail out safely
    return img[y0:y1, x0:x1]


def resize_normalize(img, size=None):
    size = size or config.IMAGE_SIZE
    resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized


def preprocess_image(path_or_array, for_model=True):
    """
    Full pipeline: read -> denoise -> deskew -> crop -> resize/normalize.
    Accepts either a filesystem path or an already-loaded BGR numpy array.
    """
    img = read_image(path_or_array) if isinstance(path_or_array, str) else path_or_array
    img = denoise(img)
    img = deskew(img)
    img = auto_crop(img)
    if for_model:
        return resize_normalize(img)
    return img


def build_processed_dataset(raw_dir=None, processed_dir=None):
    """Run the pipeline over every image in data/raw and save to data/processed,
    preserving class sub-folders. Saved as uint8 PNG (denoised/deskewed/cropped,
    but NOT resized/normalized -- resizing happens at load time in dataset.py)."""
    raw_dir = raw_dir or config.DATA_RAW_DIR
    processed_dir = processed_dir or config.DATA_PROCESSED_DIR

    count = 0
    for class_name in sorted(os.listdir(raw_dir)):
        class_raw = os.path.join(raw_dir, class_name)
        if not os.path.isdir(class_raw):
            continue
        class_processed = os.path.join(processed_dir, class_name)
        os.makedirs(class_processed, exist_ok=True)

        for fname in os.listdir(class_raw):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            src_path = os.path.join(class_raw, fname)
            img = read_image(src_path)
            img = denoise(img)
            img = deskew(img)
            img = auto_crop(img)
            cv2.imwrite(os.path.join(class_processed, fname), img)
            count += 1

    print(f"[OK] Preprocessed {count} images -> {processed_dir}")
    return count


if __name__ == "__main__":
    build_processed_dataset()
