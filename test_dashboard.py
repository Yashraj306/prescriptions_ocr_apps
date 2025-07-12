import os
import csv
from datetime import datetime
from PIL import Image
from app import extract_text_and_info
import pandas as pd

# ✅ Folder containing test images
TEST_FOLDER = "test_images"

# ✅ Output file with timestamp (avoids overwrite and permission errors)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_CSV = f"ocr_test_results_{timestamp}.csv"

# ✅ Create output directory if needed
os.makedirs("outputs", exist_ok=True)
OUTPUT_PATH = os.path.join("outputs", OUTPUT_CSV)

# ✅ Run extraction and store results
with open(OUTPUT_PATH, "w", newline="", encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Image", "Diagnosis", "Follow-Up", "Advice", "Med Count"])

    for filename in os.listdir(TEST_FOLDER):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(TEST_FOLDER, filename)
            try:
                image = Image.open(path)
                _, meds_df, diag, advice, follow = extract_text_and_info(image)

                writer.writerow([
                    filename,
                    diag,
                    follow,
                    advice.replace("\n", " "),
                    len(meds_df)
                ])

                print(f"✅ Processed: {filename} | Diagnosis: {diag} | Meds: {len(meds_df)}")

            except Exception as e:
                writer.writerow([filename, "ERROR", "ERROR", "ERROR", f"❌ {str(e)}"])
                print(f"❌ Error in {filename}: {str(e)}")

# ✅ Accuracy Summary
df = pd.read_csv(OUTPUT_PATH)
total = len(df)
success = df[~df['Diagnosis'].str.contains("ERROR|❌", na=False)].shape[0]
accuracy = (success / total) * 100 if total > 0 else 0

print("\n📊 OCR Test Summary")
print(f"Total Files Tested: {total}")
print(f"Successful Extractions: {success}")
print(f"Test Accuracy: {accuracy:.2f}%")
print(f"📁 Results saved to: {OUTPUT_PATH}")
