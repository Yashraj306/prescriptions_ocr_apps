import easyocr
import pytesseract
import cv2
import numpy as np
from PIL import Image
import warnings

# ✅ EasyOCR: Multilingual reader
reader = easyocr.Reader(['en', 'hi', 'mr'], gpu=False)

# ✅ Resize large images for speed
def resize_image(image, max_dim=1024):
    width, height = image.size
    if max(width, height) > max_dim:
        scale = max_dim / max(width, height)
        return image.resize((int(width * scale), int(height * scale)))
    return image

# ✅ Image preprocessing: grayscale, denoise, adaptive threshold, sharpen
def preprocess_image(pil_img):
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=30)

    adaptive = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 10
    )

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(adaptive, -1, kernel)
    return Image.fromarray(sharpened)

# ✅ Load image from upload path or stream
def load_image(image_file):
    return Image.open(image_file)

# ✅ Fallback OCR using pytesseract
def tesseract_ocr(pil_img):
    try:
        text = pytesseract.image_to_string(pil_img)
        return text.strip()
    except Exception as e:
        return f"Tesseract failed: {str(e)}"

# ✅ Main OCR handler function
def ocr_image(pil_image):
    image = resize_image(pil_image)
    processed = preprocess_image(image)
    img = np.array(processed)

    # 🧠 Run EasyOCR
    try:
        results = reader.readtext(img)
    except Exception as e:
        warnings.warn(f"EasyOCR error: {e}")
        results = []

    if results:
        full_text = " ".join([res[1] for res in results])
        avg_confidence = round(np.mean([res[2] for res in results]) * 100, 2)
        languages_used = ", ".join(reader.lang_list)
    else:
        # 🔁 Use Tesseract fallback if EasyOCR fails
        full_text = tesseract_ocr(processed)
        avg_confidence = "N/A (Tesseract fallback)"
        languages_used = "Fallback only: English"

    return {
        "raw_results": results,
        "text": full_text.strip(),
        "confidence": avg_confidence,
        "languages": languages_used
    }

