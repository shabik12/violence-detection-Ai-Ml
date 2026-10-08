#!/usr/bin/env python3
"""
Organize video frames into YOLO dataset format
"""

import os
import cv2
import shutil
from pathlib import Path
import random

def organize_yolo_dataset(source_frames_dir, output_dir, split_ratio=0.8):
    """
    Organize video frames into YOLO dataset structure

    Args:
        source_frames_dir: Directory containing your video frames
        output_dir: Output directory for YOLO dataset
        split_ratio: Training/validation split (default 0.8)
    """

    # Create YOLO directory structure
    yolo_dirs = {
        'images_train': os.path.join(output_dir, 'images', 'train'),
        'images_val': os.path.join(output_dir, 'images', 'val'),
        'labels_train': os.path.join(output_dir, 'labels', 'train'),
        'labels_val': os.path.join(output_dir, 'labels', 'val')
    }

    for dir_path in yolo_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []

    for ext in image_extensions:
        image_files.extend(Path(source_frames_dir).glob(f'*{ext}'))

    print(f"Found {len(image_files)} images in {source_frames_dir}")

    # Shuffle and split
    random.shuffle(image_files)
    split_idx = int(len(image_files) * split_ratio)

    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    print(f"Split: {len(train_files)} train, {len(val_files)} validation")

    # YOLO class mapping
    class_names = ['person', 'fighting', 'assault', 'abuse', 'burglary',
                   'arrest', 'normal', 'weapon', 'vandalism', 'crowd']

    def create_sample_label(img_path, label_dir):
        """Create a sample YOLO label file"""
        img_name = Path(img_path).stem
        label_path = os.path.join(label_dir, f"{img_name}.txt")

        # Read image to get dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            return False

        height, width = img.shape[:2]

        # Create sample annotations (replace with your actual detection logic)
        with open(label_path, 'w') as f:
            # Sample: Random person detection
            f.write(f"0 {0.5:.3f} {0.5:.3f} {0.1:.3f} {0.2:.3f}\n")

            # Add random violence class for demonstration
            violence_class = random.choice([1, 2, 3, 4, 5])  # fighting, assault, abuse, burglary, arrest
            if random.random() > 0.3:  # 70% chance of violence
                x = random.uniform(0.2, 0.8)
                y = random.uniform(0.2, 0.8)
                w = random.uniform(0.05, 0.15)
                h = random.uniform(0.1, 0.25)
                f.write(f"{violence_class} {x:.3f} {y:.3f} {w:.3f} {h:.3f}\n")

        return True

    # Process training files
    for img_path in train_files:
        if create_sample_label(img_path, yolo_dirs['labels_train']):
            shutil.copy2(img_path, yolo_dirs['images_train'])

    # Process validation files
    for img_path in val_files:
        if create_sample_label(img_path, yolo_dirs['labels_val']):
            shutil.copy2(img_path, yolo_dirs['images_val'])

    print(f"✅ Dataset organized successfully!")
    print(f"📁 Images: {yolo_dirs['images_train']}")
    print(f"📁 Labels: {yolo_dirs['labels_train']}")
    print(f"📝 Use data.yaml for training")

def create_dataset_info():
    """Create dataset information file"""
    info = """# Violence Detection YOLO Dataset
# Created: Today
# Classes: 10 (person, fighting, assault, abuse, burglary, arrest, normal, weapon, vandalism, crowd)

# Structure:
# datasets/
# ├── violence_yolo/
# │   ├── images/
# │   │   ├── train/
# │   │   └── val/
# │   └── labels/
# │       ├── train/
# │       └── val/
# └── data.yaml

# Usage:
# yolo train data=datasets/data.yaml model=yolov8n.pt epochs=100
# yolo predict model=yolov8n.pt source=path/to/image.jpg
"""
    with open("datasets/DATASET_INFO.txt", 'w', encoding='utf-8') as f:
        f.write(info)
    print("📄 Dataset info saved to datasets/DATASET_INFO.txt")

if __name__ == "__main__":
    # Example usage
    source_dir = "path/to/your/video/frames"  # Replace with your frames directory
    output_dir = "datasets/violence_yolo"

    print("🔧 YOLO Dataset Organizer")
    print("=" * 50)
    print("This script will organize your video frames into YOLO format")
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print()

    # Create dataset info
    create_dataset_info()

    # Uncomment and modify the line below to organize your actual frames:
    # organize_yolo_dataset(source_dir, output_dir)

    print("✅ Ready to organize your dataset!")
    print("📝 Edit this script and set your source_frames_dir")
