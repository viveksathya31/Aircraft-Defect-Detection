import os
import glob
from tqdm import tqdm

# Update this path to your specific labels folder
LABEL_DIR = 'yolo_data/train/labels' 

def check_labels(directory):
    print(f"Checking labels in: {directory}")
    txt_files = glob.glob(os.path.join(directory, '*.txt'))
    
    corrupt_files = []
    mixed_format_files = []
    
    for file_path in tqdm(txt_files):
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            parts = list(map(float, line.strip().split()))
            
            # Check 1: mixed segmentation (more than 5 values means it's a polygon, not a box)
            if len(parts) > 5:
                mixed_format_files.append((file_path, i+1, len(parts)))
                continue
                
            # Check 2: Standard YOLO format is class, x, y, w, h
            if len(parts) == 5:
                cls, x, y, w, h = parts
                
                # Check for negative sizes or out of bounds
                if w <= 0 or h <= 0 or x < 0 or y < 0:
                    corrupt_files.append((file_path, "Invalid dimensions (<=0)"))
                
                # Check for normalized coordinates > 1 (YOLO expects 0-1)
                if x > 1 or y > 1 or w > 1 or h > 1:
                     corrupt_files.append((file_path, "Coordinates not normalized (>1)"))

    print("\n--- RESULTS ---")
    if mixed_format_files:
        print(f"⚠️ Found {len(mixed_format_files)} lines with segmentation data (polygons).")
        print(f"Example: {mixed_format_files[0]}")
    
    if corrupt_files:
        print(f"❌ Found {len(corrupt_files)} files with corrupt coordinates.")
        for f in corrupt_files[:5]: # Show first 5
            print(f"  {f}")
    else:
        print("✅ No corrupt coordinates found.")

if __name__ == "__main__":
    check_labels(LABEL_DIR)