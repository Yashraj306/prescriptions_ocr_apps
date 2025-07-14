# ocr_utils.py

import easyocr
import cv2
import numpy as np
from PIL import Image
import re
import requests
from fuzzywuzzy import process

# Load OCR Reader with multilingual support
reader = easyocr.Reader(['en', 'hi', 'mr', 'kn', 'ta', 'bn'], gpu=False)


# Known global medicines (extendable)
known_meds = [
    "DOLO 650", "VOMILAST", "ZOCLAR", "CROCIN", "AZITHROMYCIN", "AMOXICILLIN", "METFORMIN",
    "PARACETAMOL", "IBUPROFEN", "PANTOPRAZOLE", "DOXYCYCLINE", "CETRIZINE", "FOLIC ACID",
    "OMEGA 3", "ACECLOFENAC", "AUGMENTIN", "PAN D", "KETO", "ASPIRIN", "CODEINE", "WARFARIN",
    "ENZFLAM", "HEXIGEL", "CANDID B", "VICKS", "SINAREST", "RANTAC", "KETOROL", "CETRIMIDE"
]

# 🔍 Get uses & warnings from FDA API
def get_medicine_info_online(med_name):
    try:
        url = f"https://api.fda.gov/drug/label.json?search=openfda.brand_name:\"{med_name}\"&limit=1"
        res = requests.get(url, timeout=5)
        data = res.json()
        use = data['results'][0].get('indications_and_usage', ['Not available'])[0]
        warn = data['results'][0].get('warnings', ['No warnings found'])[0]
        return use.strip(), warn.strip()
    except:
        return "❌ No info found online", "❌ No warning found"

# 🧼 Preprocess image for OCR
def preprocess_image(pil_img):
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=30)
    adaptive = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 10)
    sharpened = cv2.filter2D(adaptive, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
    return Image.fromarray(sharpened)

# 📏 Resize image if too small
def resize_image(image, min_width=1000):
    width, height = image.size
    if width < min_width:
        scale = min_width / width
        return image.resize((int(width * scale), int(height * scale)))
    return image

# 🎯 Extract all structured info
def extract_text_and_info(image):
    image = resize_image(image)
    image = preprocess_image(image)
    img_np = np.array(image)
    results = reader.readtext(img_np)
    lines = [res[1].strip() for res in results if res[1].strip()]

    meds = []
    diagnosis_set = set()
    uses, risks, remedies = [], [], []

    current = {}
    for line in lines:
        upper = line.upper()
        if re.search(r"(TAB|CAP)[. ]", upper):
            if current:
                meds.append(current)
            current = {"Medicine Name": upper.strip(), "Dosage": "", "Duration": ""}
        elif any(kw in upper for kw in ["MORNING", "NIGHT", "AFTER FOOD"]):
            current["Dosage"] += " " + upper.strip()
        elif "DAYS" in upper:
            match = re.search(r"\d+\s+DAYS", upper)
            if match:
                current["Duration"] = match.group()
    if current:
        meds.append(current)

    for m in meds:
        raw = m["Medicine Name"].upper()
        match, score = process.extractOne(raw, known_meds)
        m["Matched Name"] = match if score > 75 else raw

        use, warn = get_medicine_info_online(match)
        uses.append(f"{match}: {use}")
        risks.append(f"{match}: {warn}")
        if "MALARIA" in use.upper():
            diagnosis_set.add("Malaria")
        elif "FEVER" in use.upper():
            diagnosis_set.add("Viral Fever")
        elif "VOMITING" in use.upper():
            diagnosis_set.add("Stomach Infection")
        elif "ACIDITY" in use.upper():
            diagnosis_set.add("Acid Reflux")

    diagnosis = ", ".join(diagnosis_set) if diagnosis_set else "⚠️ Diagnosis not found"
    return meds, diagnosis, "\n".join(uses), "\n".join(risks), "🌿 Home remedies vary based on condition."

# 👉 For Gradio
def ocr_image(pil_img):
    return extract_text_and_info(pil_img)

def load_image(file):
    return Image.open(file)
