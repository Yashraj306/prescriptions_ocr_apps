# streamlit_app.py

import streamlit as st
from ocr_utils import load_image, ocr_image

st.set_page_config(page_title="Prescription OCR App", layout="centered")

st.title("📄 Medical Prescription OCR")
st.write("Upload a prescription image to extract text using AI.")

# Upload image
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Prescription", use_column_width=True)

    # Convert image to PIL format
    image = load_image(uploaded_file)

    if st.button("🔍 Extract Text"):
        with st.spinner("Analyzing image..."):
            results, text = ocr_image(image)

        st.subheader("📋 Extracted Text")
        st.text_area("OCR Output", text, height=300)

        st.success("✅ Text extraction complete!")
