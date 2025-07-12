import gradio as gr
import easyocr
import numpy as np
import pandas as pd
import re
import cv2
from PIL import Image
import pytesseract

# ✅ Load EasyOCR model (English-only for speed)
reader = easyocr.Reader(['en'], gpu=False)

# ✅ Fix common OCR errors
def clean_text(text):
    corrections = {
        "s00mg": "500mg", "mgmg": "mg", "TOt": "Tot",
        "tabtab": "tab", "capcap": "cap", "doctar": "doctor",
        "0.": "D.", "1)": "1 )"
    }
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    return text

# ✅ Tesseract fallback if EasyOCR fails
def fallback_ocr(image):
    try:
        gray = image.convert("L")
        return pytesseract.image_to_string(gray).strip()
    except:
        return ""

# ✅ Enhance blurry images
def preprocess_image(pil_img):
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    if lap_var < 100:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharp = cv2.filter2D(gray, -1, kernel)
        _, thresh = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(thresh)
    else:
        return pil_img

# ✅ Resize large images
def resize_image(image, max_dim=1024):
    width, height = image.size
    if max(width, height) > max_dim:
        scale = max_dim / max(width, height)
        return image.resize((int(width * scale), int(height * scale)))
    return image

# ✅ Main extraction function
def extract_text_and_info(image):
    image = resize_image(image)
    image = preprocess_image(image)

    img_np = np.array(image)
    results = reader.readtext(img_np)

    # 🛡️ Fallback if OCR fails
    if len(results) < 5:
        extracted_text = clean_text(fallback_ocr(image))
        lines = extracted_text.split("\n")
    else:
        lines = [res[1] for res in results]
        extracted_text = clean_text(" ".join(lines))

    # 🩺 Diagnosis
    diagnosis = ""
    for line in lines:
        if "diagnosis" in line.lower():
            parts = re.split(r"diagnosis[:\-\s]*", line, flags=re.I)
            if len(parts) > 1:
                diagnosis = parts[1].strip().upper()

    # 📅 Follow-up
    follow_up = ""
    for line in lines:
        match = re.search(r"\b\d{2}-\d{2}-\d{4}\b", line)
        if match:
            follow_up = match.group()

    # ⚠️ Advice
    advice = []
    for line in lines:
        if any(kw in line.lower() for kw in ["bed rest", "outside food", "digest food", "boiled", "eat"]):
            advice.append(line.strip())

    # 💊 Medications
    meds = []
    current = {}
    for line in lines:
        if re.search(r"^\s*(\d+\))?\s*(TAB|CAP)[. ]+", line, re.I):
            if current:
                meds.append(current)
            current = {"Medicine Name": line.strip(), "Dosage": "", "Duration": ""}
        elif any(kw in line.lower() for kw in ["morning", "night", "after food", "days", "cap", "tab"]):
            if any(t in line.lower() for t in ["morning", "night", "after food"]):
                current["Dosage"] += " " + line.strip()
            if "days" in line.lower():
                match = re.search(r"\d+\s+Days", line, re.I)
                if match:
                    current["Duration"] = match.group()
    if current:
        meds.append(current)

    # 🛡️ Ensure fields are not empty
    if not diagnosis:
        diagnosis = "⚠️ Diagnosis not found"
    if not follow_up:
        follow_up = "⚠️ Date not detected"
    if not meds:
        meds = [{"Medicine Name": "⚠️ None found", "Dosage": "", "Duration": ""}]

    df = pd.DataFrame(meds)
    return extracted_text, df, diagnosis, "\n".join(advice), follow_up

# ✅ Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🩺 Medical Prescription OCR App (Upgraded for Reliability 🚀)")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="📤 Upload Prescription Image")
            submit_btn = gr.Button("🔍 Extract Text")

        with gr.Column():
            extracted_text = gr.Textbox(label="📄 Extracted Raw Text", lines=5)
            diagnosis = gr.Textbox(label="🩺 Diagnosis")
            advice = gr.Textbox(label="⚠️ Advice / Precautions", lines=4)
            followup = gr.Textbox(label="📅 Follow-Up Date")
            meds_df = gr.Dataframe(label="💊 Medications & Dosage")

    submit_btn.click(
        fn=extract_text_and_info,
        inputs=image_input,
        outputs=[extracted_text, meds_df, diagnosis, advice, followup]
    )

if __name__ == "__main__":
    demo.launch()
