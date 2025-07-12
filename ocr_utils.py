import easyocr
import cv2
import numpy as np
from PIL import Image
preprocess_image()
# Add desired languages (example: English + Hindi + Marathi)
reader = easyocr.Reader(['en', 'hi', 'mr'])

def preprocess_image(pil_img):
    img = np.array(pil_img)

    # 🛡️ Auto-fix grayscale/single channel images
    if len(img.shape) == 2:  # Already grayscale
        gray = img
    elif img.shape[2] == 1:  # Single channel
        gray = img[:, :, 0]
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Denoise
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

    return Image.fromarray(sharpened)

def load_image(image_file):
    return Image.open(image_file)

def ocr_image(pil_image):
    img = np.array(pil_image)
    results = reader.readtext(img)
    full_text = " ".join([res[1] for res in results])
    return results, full_text
