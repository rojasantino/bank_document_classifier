"""
generate_synthetic_data.py
---------------------------
Generates synthetic bank-document images (cheques, statements, deposit
slips, KYC forms, etc.) so the full pipeline can be trained and tested
end-to-end WITHOUT needing a real scanned document dataset.

Replace this with your own scanned images later:
    data/raw/<class_name>/*.png

Usage:
    python data/generate_synthetic_data.py
    python data/generate_synthetic_data.py --per_class 100
"""

import os
import sys
import random
import argparse

from PIL import Image, ImageDraw, ImageFont

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

random.seed(config.RANDOM_SEED)

FIRST_NAMES = ["Ravi", "Priya", "Arun", "Divya", "Karthik", "Meena", "Suresh", "Anitha"]
LAST_NAMES = ["Kumar", "Raj", "Nair", "Iyer", "Pillai", "Sharma", "Menon", "Reddy"]
BANKS = ["HDFC BANK", "STATE BANK OF INDIA", "ICICI BANK", "AXIS BANK", "CANARA BANK"]


def random_name():
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def random_account_no():
    return "".join(str(random.randint(0, 9)) for _ in range(12))


def random_ifsc():
    return random.choice(["HDFC", "SBIN", "ICIC", "UTIB", "CNRB"]) + "0" + "".join(
        str(random.randint(0, 9)) for _ in range(6)
    )


def random_amount():
    return f"{random.randint(500, 95000):,}.00"


def random_date():
    d = random.randint(1, 28)
    m = random.randint(1, 12)
    y = random.randint(2022, 2026)
    return f"{d:02d}/{m:02d}/{y}"


def get_font(size=20):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size
        )
    except Exception:
        return ImageFont.load_default()


def draw_box(draw, xy, outline=(60, 60, 60), width=1):
    draw.rectangle(xy, outline=outline, width=width)


def base_canvas():
    w, h = config.SYNTHETIC_IMAGE_SIZE
    bg = tuple(random.randint(235, 250) for _ in range(3))
    img = Image.new("RGB", (w, h), bg)
    return img


def add_noise_texture(draw, w, h, density=250):
    for _ in range(density):
        x, y = random.randint(0, w - 1), random.randint(0, h - 1)
        c = random.randint(200, 235)
        draw.point((x, y), fill=(c, c, c))


# ---------------------------------------------------------------------------
# One drawing function per document class. Each produces a distinctive
# layout so the CNN has real structural signal to learn from.
# ---------------------------------------------------------------------------

def draw_cheque(img):
    d = ImageDraw.Draw(img)
    w, h = img.size
    f_bank = get_font(26)
    f = get_font(18)
    add_noise_texture(d, w, h)
    d.text((20, 15), random.choice(BANKS), font=f_bank, fill=(20, 20, 90))
    draw_box(d, (w - 220, 15, w - 20, 55))
    date = random_date()
    for i, ch in enumerate(date.replace("/", "")):
        d.text((w - 210 + i * 20, 20), ch, font=f)
    d.text((20, 90), "Pay", font=f)
    d.text((70, 90), random_name(), font=f, fill=(0, 0, 150))
    d.text((20, 140), "Rupees", font=f)
    d.text((100, 140), "Only", font=f, fill=(0, 0, 150))
    draw_box(d, (w - 250, 130, w - 20, 170))
    d.text((w - 240, 138), "Rs. " + random_amount(), font=f)
    d.text((20, 220), "A/c No.", font=f)
    draw_box(d, (100, 215, 320, 245))
    d.text((105, 222), random_account_no(), font=f)
    d.text((20, 260), "IFSC: " + random_ifsc(), font=f)
    d.line((350, 300, 520, 300), fill=(0, 0, 150), width=2)
    d.text((350, 305), "(signature)", font=get_font(12), fill=(120, 120, 120))
    d.text((20, h - 40), "MICR: " + "".join(str(random.randint(0, 9)) for _ in range(9)), font=f)
    return img


def draw_bank_statement(img):
    d = ImageDraw.Draw(img)
    w, h = img.size
    add_noise_texture(d, w, h)
    d.text((20, 15), random.choice(BANKS) + " - ACCOUNT STATEMENT", font=get_font(22), fill=(20, 20, 90))
    d.text((20, 55), f"Account Holder: {random_name()}", font=get_font(16))
    d.text((20, 80), f"Account No: {random_account_no()}", font=get_font(16))
    d.text((20, 105), f"Period: {random_date()} to {random_date()}", font=get_font(16))
    headers = ["Date", "Description", "Debit", "Credit", "Balance"]
    col_x = [20, 150, 420, 520, 620]
    for x, htext in zip(col_x, headers):
        d.text((x, 150), htext, font=get_font(15), fill=(0, 0, 0))
    d.line((20, 175, w - 20, 175), fill=(0, 0, 0), width=1)
    y = 185
    for _ in range(9):
        d.text((col_x[0], y), random_date(), font=get_font(13))
        d.text((col_x[1], y), random.choice(["UPI TRANSFER", "ATM WDL", "SALARY CREDIT", "POS PURCHASE"]), font=get_font(13))
        d.text((col_x[2], y), str(random.randint(100, 5000)), font=get_font(13))
        d.text((col_x[3], y), str(random.randint(100, 5000)), font=get_font(13))
        d.text((col_x[4], y), str(random.randint(1000, 90000)), font=get_font(13))
        y += 28
    return img


def draw_card_document(img):
    d = ImageDraw.Draw(img)
    w, h = img.size
    d.rectangle((0, 0, w, h), fill=(40, 40, 70))
    d.rectangle((60, 100, w - 60, h - 100), fill=(70, 70, 110))
    d.text((90, 130), random.choice(BANKS), font=get_font(22), fill=(255, 255, 255))
    d.text((90, 250), " ".join(["".join(str(random.randint(0, 9)) for _ in range(4)) for _ in range(4)]),
           font=get_font(24), fill=(255, 255, 255))
    d.text((90, 320), random_name().upper(), font=get_font(18), fill=(230, 230, 230))
    d.text((90, 350), f"VALID THRU {random.randint(1,12):02d}/{random.randint(26,30)}", font=get_font(14), fill=(200, 200, 200))
    return img


def draw_deposit_slip(img):
    d = ImageDraw.Draw(img)
    w, h = img.size
    add_noise_texture(d, w, h)
    d.text((20, 15), random.choice(BANKS) + " - DEPOSIT SLIP", font=get_font(22), fill=(20, 90, 20))
    d.text((20, 60), f"Date: {random_date()}", font=get_font(16))
    d.text((20, 90), f"Depositor Name: {random_name()}", font=get_font(16))
    d.text((20, 120), f"Account No: {random_account_no()}", font=get_font(16))
    draw_box(d, (20, 160, 400, 260))
    d.text((30, 170), "Cash", font=get_font(14))
    d.text((30, 200), "Cheque", font=get_font(14))
    d.text((30, 230), "Total", font=get_font(14))
    d.text((250, 170), "Rs. " + random_amount(), font=get_font(14))
    d.line((450, 300, 620, 300), fill=(0, 0, 0), width=1)
    d.text((450, 305), "Depositor Signature", font=get_font(12))
    return img


def draw_withdrawal_slip(img):
    d = ImageDraw.Draw(img)
    w, h = img.size
    add_noise_texture(d, w, h)
    d.text((20, 15), random.choice(BANKS) + " - WITHDRAWAL SLIP", font=get_font(22), fill=(150, 30, 30))
    d.text((20, 60), f"Date: {random_date()}", font=get_font(16))
    d.text((20, 90), f"Account No: {random_account_no()}", font=get_font(16))
    d.text((20, 120), f"Amount: Rs. {random_amount()}", font=get_font(16))
    d.text((20, 150), f"Name: {random_name()}", font=get_font(16))
    d.line((450, 300, 620, 300), fill=(0, 0, 0), width=1)
    d.text((450, 305), "Signature", font=get_font(12))
    return img


def draw_account_opening_form(img):
    d = ImageDraw.Draw(img)
    w, h = img.size
    add_noise_texture(d, w, h)
    d.text((20, 15), "ACCOUNT OPENING FORM", font=get_font(24), fill=(20, 20, 90))
    fields = ["Applicant Name", "Father/Spouse Name", "Date of Birth", "PAN No",
              "Aadhaar No", "Mobile No", "Address", "Nominee Name"]
    y = 60
    for field in fields:
        d.text((20, y), field + ":", font=get_font(15))
        draw_box(d, (250, y - 3, 620, y + 20))
        y += 40
    return img


def draw_kyc_document(img):
    d = ImageDraw.Draw(img)
    w, h = img.size
    add_noise_texture(d, w, h)
    d.text((20, 15), "KNOW YOUR CUSTOMER (KYC)", font=get_font(24), fill=(90, 60, 20))
    draw_box(d, (w - 180, 60, w - 30, 210))
    d.text((w - 165, 120), "PHOTO", font=get_font(14), fill=(150, 150, 150))
    d.text((20, 70), f"Name: {random_name()}", font=get_font(16))
    d.text((20, 100), f"ID No: {''.join(str(random.randint(0,9)) for _ in range(12))}", font=get_font(16))
    d.text((20, 130), f"DOB: {random_date()}", font=get_font(16))
    d.text((20, 160), "Address: 12, MG Road, Chennai", font=get_font(16))
    d.text((20, 190), "Document Type: Aadhaar Card", font=get_font(16))
    return img


def draw_other_financial_document(img):
    d = ImageDraw.Draw(img)
    w, h = img.size
    add_noise_texture(d, w, h)
    d.text((20, 15), random.choice(["LOAN APPLICATION", "FIXED DEPOSIT RECEIPT", "INSURANCE PREMIUM RECEIPT"]),
           font=get_font(22), fill=(60, 60, 60))
    y = 60
    for _ in range(6):
        d.text((20, y), random.choice(["Ref No: " + str(random.randint(100000, 999999)),
                                        "Amount: Rs. " + random_amount(),
                                        "Date: " + random_date(),
                                        "Branch: Main Branch"]), font=get_font(15))
        y += 30
    return img


DRAW_FUNCS = {
    "cheque": draw_cheque,
    "bank_statement": draw_bank_statement,
    "card_document": draw_card_document,
    "deposit_slip": draw_deposit_slip,
    "withdrawal_slip": draw_withdrawal_slip,
    "account_opening_form": draw_account_opening_form,
    "kyc_document": draw_kyc_document,
    "other_financial_document": draw_other_financial_document,
}


def generate(per_class):
    for class_name in config.CLASS_NAMES:
        out_dir = os.path.join(config.DATA_RAW_DIR, class_name)
        os.makedirs(out_dir, exist_ok=True)
        draw_fn = DRAW_FUNCS[class_name]
        for i in range(per_class):
            img = base_canvas()
            img = draw_fn(img)
            # small random rotation to mimic scan skew -> preprocessing will fix this
            angle = random.uniform(-3, 3)
            img = img.rotate(angle, expand=False, fillcolor=(245, 245, 245))
            img.save(os.path.join(out_dir, f"{class_name}_{i:04d}.png"))
        print(f"[OK] {class_name}: {per_class} images -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_class", type=int, default=config.SYNTHETIC_IMAGES_PER_CLASS)
    args = parser.parse_args()
    generate(args.per_class)
    print("\nSynthetic dataset generation complete.")
    print(f"Total images: {args.per_class * len(config.CLASS_NAMES)}")
