# app.py

import gradio as gr
from ocr_utils import load_image, ocr_image

# 💊 Prescription AI Assistant UI
with gr.Blocks() as demo:
    gr.Markdown("""
    # 🩺 Global Prescription AI Assistant
    Upload a prescription image (handwritten/printed). We'll extract:
    - ✅ Medicines, Dosage, Duration
    - ✅ Diagnosis based on medicines
    - ✅ Uses, Warnings, Home Remedies
    """)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="📤 Upload Prescription Image")
            submit_btn = gr.Button("🔍 Analyze Prescription")

        with gr.Column():
            diagnosis = gr.Textbox(label="🧠 Inferred Diagnosis")
            uses = gr.Textbox(label="💊 Medicine Use", lines=4)
            risks = gr.Textbox(label="⚠️ What Happens If Ignored", lines=4)
            remedies = gr.Textbox(label="🌿 Home Remedies", lines=3)
            meds_df = gr.Dataframe(headers=["Medicine Name", "Dosage", "Duration"], label="📋 Extracted Medicines")

    submit_btn.click(
        fn=ocr_image,
        inputs=image_input,
        outputs=[meds_df, diagnosis, uses, risks, remedies]
    )

if __name__ == "__main__":
    demo.launch(share=True)  # Set share=True for remote access
