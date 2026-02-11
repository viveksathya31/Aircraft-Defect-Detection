import os
import glob
from tqdm import tqdm

# Update this to your labels path
LABEL_DIR = 'yolo_data/train/labels'

def convert_polygon_to_box(polygon_coords):
    """
    Converts a list of polygon coordinates [x1, y1, x2, y2, ...] 
    to YOLO box format [x_center, y_center, width, height]
    """
    # Separate X and Y coordinates
    xs = polygon_coords[0::2]
    ys = polygon_coords[1::2]
    
    # Find the bounding box
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    # Calculate center, width, height
    width = max_x - min_x
    height = max_y - min_y
    x_center = min_x + (width / 2)
    y_center = min_y + (height / 2)
    
    return [x_center, y_center, width, height]

def clean_labels(directory):
    print(f"Cleaning labels in: {directory}")
    txt_files = glob.glob(os.path.join(directory, '*.txt'))
    fixed_count = 0
    
    for file_path in tqdm(txt_files):
        needs_rewrite = False
        new_lines = []
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = list(map(float, line.strip().split()))
            class_id = int(parts[0])
            coords = parts[1:]
            
            # If we find a polygon (more than 4 coords), convert it
            if len(coords) > 4:
                box = convert_polygon_to_box(coords)
                # Format: class x_center y_center width height
                new_line = f"{class_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n"
                new_lines.append(new_line)
                needs_rewrite = True
            else:
                # Keep existing valid box lines
                new_lines.append(line)
        
        if needs_rewrite:
            with open(file_path, 'w') as f:
                f.writelines(new_lines)
            fixed_count += 1

    print(f"\n✅ Fixed {fixed_count} files. All polygons converted to bounding boxes.")

if __name__ == "__main__":
    clean_labels(LABEL_DIR)