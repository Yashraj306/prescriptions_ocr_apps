import easyocr
import pytesseract
import cv2
import numpy as np
from PIL import Image
import warnings

# ✅ EasyOCR: Multilingual reader
reader = easyocr.Reader(['en', 'hi', 'mr'], gpu=False)

# ✅ Resize large images for speed
def resize_image(image, min_width=1000):
    width, height = image.size
    if width < min_width:
        scale = min_width / width
        return image.resize((int(width * scale), int(height * scale)))
    return image


# ✅ Image preprocessing: grayscale, denoise, adaptive threshold, sharpen
def preprocess_image(pil_img, debug=False, filename="processed_debug.jpg"):
    # Convert to RGB if needed
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=30)

    # Adaptive Thresholding
    adaptive = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 10
    )

    # Sharpen
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(adaptive, -1, kernel)

    # ✅ Save debug image if needed
    if debug:
        cv2.imwrite(filename, sharpened)

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
    # Step 1: Resize for OCR accuracy
    image = resize_image(pil_image, min_width=1000)

    # Step 2: Preprocess and save debug image
    processed = preprocess_image(image, debug=True, filename="processed_debug.jpg")

    img = np.array(processed)

    # Step 3: Try EasyOCR with paragraph detection
    try:
        results = reader.readtext(img, detail=1, paragraph=True)
    except Exception as e:
        print("⚠️ EasyOCR failed:", e)
        results = []

    # Step 4: Handle EasyOCR result
    if results:
        text = " ".join([res[1] for res in results])
        conf = round(np.mean([res[2] for res in results]) * 100, 2)
        langs = ", ".join(reader.lang_list)
    else:
        # Step 5: Fallback to Tesseract if EasyOCR fails
        print("🔁 Using Tesseract fallback...")
        text = pytesseract.image_to_string(processed, lang="eng+hin+mar")
        conf = "N/A"
        langs = "Fallback: Tesseract (eng+hin+mar)"

    return {
        "raw_results": results,
        "text": text.strip(),
        "confidence": conf,
        "languages": langs
    }
