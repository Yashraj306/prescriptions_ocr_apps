import easyocr
import cv2
import numpy as np
from PIL import Image
preprocess_image()
# Add desired languages (example: English + Hindi + Marathi)
reader = easyocr.Reader(['en', 'hi', 'mr'])

def load_image(image_file):
    return Image.open(image_file)

def ocr_image(pil_image):
    img = np.array(pil_image)
    results = reader.readtext(img)
    full_text = " ".join([res[1] for res in results])
    return results, full_text
