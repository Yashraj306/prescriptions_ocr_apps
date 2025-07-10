# ocr_utils.py

import easyocr
import cv2
import numpy as np
from PIL import Image

# Load EasyOCR once (only English language)
reader = easyocr.Reader(['en'])

def load_image(image_file):
    """Load image from uploaded file."""
    return Image.open(image_file)

def ocr_image(pil_image):
    """Perform OCR using EasyOCR."""
    img = np.array(pil_image)
    results = reader.readtext(img)

    # Combine all detected text into one string
    full_text = " ".join([res[1] for res in results])
    return results, full_text
