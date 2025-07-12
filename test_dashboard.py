import os
import csv
from PIL import Image
from app import extract_text_and_info

TEST_FOLDER = "test_images"
OUTPUT_CSV = "ocr_test_results.csv"
# Check if output CSV is locked (e.g., open in Excel)
if os.path.exists(OUTPUT_CSV):
    try:
        os.rename(OUTPUT_CSV, OUTPUT_CSV)  # triggers error if file is locked
    except PermissionError:
        print("❌ Please close 'ocr_test_results.csv' before running the test.")
        exit()

# ✅ Run extraction for all images and log results
with open(OUTPUT_CSV, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Image", "Diagnosis", "Follow-Up", "Advice", "Med Count"])

    for filename in os.listdir(TEST_FOLDER):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(TEST_FOLDER, filename)
            image = Image.open(path)

            try:
                _, meds_df, diag, advice, follow = extract_text_and_info(image)
                writer.writerow([
                    filename,
                    diag,
                    follow,
                    advice.replace("\n", " "),
                    len(meds_df)
                ])
            except Exception as e:
                writer.writerow([filename, "ERROR", "ERROR", "ERROR", f"❌ {str(e)}"])

print("✅ Test completed! Results saved in ocr_test_results.csv")