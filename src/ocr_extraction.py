"""
ocr_extraction.py
------------------
Phase 2: once a document is classified as a "cheque" (or other type),
run OCR to pull out structured fields:
    - Date
    - Payee name
    - Amount
    - Account number
    - IFSC code

Uses pytesseract (Tesseract OCR). Region-of-interest (ROI) coordinates
below are tuned for a 900x500 synthetic cheque layout (see
data/generate_synthetic_data.py::draw_cheque) or a similarly-cropped
real cheque. For production, calibrate ROIs to your own scanner/bank
template, or replace with a text-detection model (e.g. CRAFT / EAST)
for layout-agnostic extraction.

Usage:
    python -m src.ocr_extraction --image path/to/cheque.png
"""

import os
import re
import sys
import argparse

import cv2
import pytesseract

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from src.data_preprocessing import read_image, denoise, deskew, auto_crop  # noqa: E402


# Approximate ROIs as fractions of (width, height) -- robust to minor resizing.
CHEQUE_ROIS = {
    "date": (0.74, 0.02, 0.98, 0.11),
    "payee": (0.06, 0.16, 0.55, 0.24),
    "amount": (0.72, 0.24, 0.98, 0.34),
    "account_number": (0.10, 0.42, 0.36, 0.50),
    "ifsc": (0.02, 0.50, 0.35, 0.58),
    "signature": (0.38, 0.56, 0.60, 0.64),
}


def crop_roi(img, roi):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = roi
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def ocr_text(img_crop, config_str="--psm 7"):
    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(gray, config=config_str)
    return text.strip()


def extract_cheque_fields(image_path_or_array):
    img = read_image(image_path_or_array) if isinstance(image_path_or_array, str) else image_path_or_array
    img = denoise(img)
    img = deskew(img)
    img = auto_crop(img)

    fields = {}
    for field_name, roi in CHEQUE_ROIS.items():
        crop = crop_roi(img, roi)
        if crop.size == 0:
            fields[field_name] = ""
            continue
        raw_text = ocr_text(crop)
        fields[field_name] = raw_text

    # Light cleanup / validation
    amount_match = re.search(r"[\d,]+\.?\d*", fields.get("amount", ""))
    fields["amount_clean"] = amount_match.group(0) if amount_match else None

    account_match = re.search(r"\d{6,18}", fields.get("account_number", ""))
    fields["account_number_clean"] = account_match.group(0) if account_match else None

    ifsc_match = re.search(r"[A-Z]{4}0[A-Z0-9]{6}", fields.get("ifsc", "").upper())
    fields["ifsc_clean"] = ifsc_match.group(0) if ifsc_match else None

    return fields


def match_courtesy_and_legal_amount(courtesy_amount_text, legal_amount_text):
    """
    'Courtesy amount' = numeric figure box (e.g. Rs. 8,900.00)
    'Legal amount'    = amount written in words on the Rupees line.
    Full words->number verification needs a dedicated parser; here we
    do a light sanity check (presence + non-empty) and flag for manual
    review otherwise -- extend with a word-to-number library for full
    automation.
    """
    courtesy_ok = bool(re.search(r"\d", courtesy_amount_text or ""))
    legal_ok = bool((legal_amount_text or "").strip())
    return {
        "courtesy_amount_detected": courtesy_ok,
        "legal_amount_detected": legal_ok,
        "requires_manual_review": not (courtesy_ok and legal_ok),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    result = extract_cheque_fields(args.image)
    print("\nExtracted cheque fields:")
    for k, v in result.items():
        print(f"  {k}: {v}")
