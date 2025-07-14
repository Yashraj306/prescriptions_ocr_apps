# trocr_ocr.py

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# ✅ Load TrOCR model (base for handwriting)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# ✅ Run TrOCR on input PIL image
def run_trocr(pil_image):
    image = pil_image.convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()
