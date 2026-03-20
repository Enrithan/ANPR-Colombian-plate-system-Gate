import os
import shutil

SOURCE = "dataset"
DEST = "curated_dataset"

if not os.path.exists(DEST):
    os.makedirs(DEST)
    print(f"Created: {DEST}")

files = [f for f in os.listdir(SOURCE) if f.endswith(".jpg")]
print(f"Sorting {len(files)} images...")

for file in files:
    # Filename: plate_{car_id}_{timestamp}_{ocr_result}.jpg
    # Example: plate_2_20260318_141341_NEM621.jpg
    name_no_ext = os.path.splitext(file)[0]
    parts = name_no_ext.split("_")
    
    # Extract the OCR result (last part)
    plate_name = parts[-1] if len(parts) >= 4 else "unknown"
    
    target_dir = os.path.join(DEST, plate_name)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    shutil.copy(os.path.join(SOURCE, file), os.path.join(target_dir, file))

print(f"\nSorting Complete!")
print(f"1. Open the '{DEST}' folder.")
print(f"2. Delete folders that are NOT plates (False Positives).")
print(f"3. Rename folders if the Plate number is slightly wrong (e.g. NEMG21 -> NEM621).")
print(f"4. You now have a clean dataset ready for training!")
