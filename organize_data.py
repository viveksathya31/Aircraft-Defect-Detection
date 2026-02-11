import os
import shutil
import pandas as pd
from tqdm import tqdm

# CONFIGURATION
# Point this to your main data folder containing 'train', 'valid', 'test'
BASE_DIR = "yolo_classify_data" 

def organize_split(split_name):
    split_dir = os.path.join(BASE_DIR, split_name)
    csv_path = os.path.join(split_dir, "_classes.csv")
    
    if not os.path.exists(csv_path):
        print(f"⚠️ Skipping {split_name}: No _classes.csv found.")
        return

    print(f"📂 Processing {split_name} set...")
    
    # Read the CSV
    # The screenshot shows columns: filename, crack, dent, missing-head, etc.
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Clean column names (remove spaces)
    df.columns = [c.strip() for c in df.columns]
    
    # Get class names (all columns except 'filename')
    class_names = [c for c in df.columns if c != 'filename']
    print(f"   Found classes: {class_names}")

    # Create subfolders for each class
    for cls in class_names:
        os.makedirs(os.path.join(split_dir, cls), exist_ok=True)

    # Move images
    count = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        filename = row['filename']
        src_path = os.path.join(split_dir, filename)
        
        # Check which class is '1' (Active)
        for cls in class_names:
            if row[cls] == 1:
                dst_path = os.path.join(split_dir, cls, filename)
                
                # Only move if source exists
                if os.path.exists(src_path):
                    shutil.move(src_path, dst_path)
                    count += 1
                break
    
    # Optional: Remove the CSV after we are done
    # os.remove(csv_path)
    print(f"✅ Moved {count} images in {split_name}.\n")

if __name__ == "__main__":
    # Install pandas if you don't have it: pip install pandas
    if not os.path.exists(BASE_DIR):
        print(f"❌ Error: Could not find directory '{BASE_DIR}'")
    else:
        for split in ['train', 'valid', 'test']:
            if os.path.exists(os.path.join(BASE_DIR, split)):
                organize_split(split)
        print("🎉 Organization complete! You can now train.")