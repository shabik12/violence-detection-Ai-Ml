#!/usr/bin/env python3
"""
Convert CNN detection results to YOLO format for training
"""

import os
import cv2
import json
from pathlib import Path

def convert_to_yolo_format(image_dir, output_dir, class_mapping):
    """
    Convert detection results to YOLO format

    Args:
        image_dir: Directory containing video frames
        output_dir: Directory to save YOLO labels
        class_mapping: Dictionary mapping CNN labels to YOLO class IDs
    """

    # Create output directories
    os.makedirs(f"{output_dir}/images/train", exist_ok=True)
    os.makedirs(f"{output_dir}/images/val", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/train", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/val", exist_ok=True)

    # YOLO class mapping for violence detection
    yolo_classes = {
        'person': 0,
        'fighting': 1,
        'assault': 2,
        'abuse': 3,
        'burglary': 4,
        'arrest': 5,
        'normal': 6,
        'weapon': 7,
        'vandalism': 8,
        'crowd': 9
    }

    # Process each image
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    for i, img_file in enumerate(image_files):
        img_path = os.path.join(image_dir, img_file)

        # Read image to get dimensions
        img = cv2.imread(img_path)
        if img is None:
            continue

        height, width = img.shape[:2]

        # Create corresponding label file
        label_file = img_file.rsplit('.', 1)[0] + '.txt'
        label_path = f"{output_dir}/labels/train/{label_file}"

        # Example: Create sample detection (replace with actual detection logic)
        with open(label_path, 'w') as f:
            # Sample detection - replace with your actual detection results
            # Format: <class_id> <x_center> <y_center> <width> <height>
            f.write(f"0 0.5 0.5 0.1 0.2\n")  # Person
            # Add more detections as needed

        # Copy image to train directory
        import shutil
        shutil.copy2(img_path, f"{output_dir}/images/train/{img_file}")

        if i % 100 == 0:
            print(f"Processed {i}/{len(image_files)} images")

    print(f"✅ Converted {len(image_files)} images to YOLO format")
    print(f"📁 Images saved to: {output_dir}/images/train")
    print(f"📁 Labels saved to: {output_dir}/labels/train")

def create_sample_dataset():
    """Create a minimal sample dataset for testing"""

    # Create sample images directory
    sample_dir = "datasets/violence_detect_sample"
    os.makedirs(f"{sample_dir}/images/train", exist_ok=True)
    os.makedirs(f"{sample_dir}/labels/train", exist_ok=True)

    # Create sample label
    with open(f"{sample_dir}/labels/train/violence_001.txt", 'w') as f:
        f.write("1 0.45 0.32 0.12 0.25\n")  # fighting
        f.write("0 0.60 0.40 0.08 0.15\n")  # person

    print("✅ Sample dataset created at:", sample_dir)
    print("📝 violence_data.yaml configured for this structure")

if __name__ == "__main__":
    # Create sample dataset
    create_sample_dataset()

    # Example usage:
    # convert_to_yolo_format("path/to/frames", "datasets/violence_detect", {})
