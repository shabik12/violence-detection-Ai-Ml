#!/usr/bin/env python3
"""
YOLO Label Creator - Simple tool to create label files for your weapons dataset
"""

import os
import cv2
import numpy as np

# Dataset paths
image_dir = "datasets/weapons/images/train"
label_dir = "datasets/weapons/labels/train"

# Class mapping
classes = {
    0: "gun",
    1: "knife",
    2: "mobile"
}

def create_label_file(image_name, class_id, bbox):
    """
    Create YOLO label file for an image

    Args:
        image_name: Name of the image file (e.g., "gun.png")
        class_id: Class ID (0=gun, 1=knife, 2=mobile)
        bbox: Bounding box [x1, y1, x2, y2] in pixels
    """
    # Get image size
    image_path = os.path.join(image_dir, image_name)
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    # Convert bbox to YOLO format (normalized 0-1)
    x_center = (bbox[0] + bbox[2]) / 2 / width
    y_center = (bbox[1] + bbox[3]) / 2 / height
    box_width = (bbox[2] - bbox[0]) / width
    box_height = (bbox[3] - bbox[1]) / height

    # Create label file
    label_name = os.path.splitext(image_name)[0] + ".txt"
    label_path = os.path.join(label_dir, label_name)

    with open(label_path, 'w') as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

    print(f"✅ Created label: {label_name}")
    print(f"   Class: {classes[class_id]}")
    print(f"   BBox: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
    print(f"   Normalized: {x_center:.3f} {y_center:.3f} {box_width:.3f} {box_height:.3f}")

def show_labeling_guide():
    """Show instructions for manual labeling"""
    print("=" * 60)
    print("YOLO LABELING GUIDE")
    print("=" * 60)
    print("\n📸 Found images:")

    # List images
    for img_file in os.listdir(image_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"   - {img_file}")

    print("\n🎯 Classes:")
    for id, name in classes.items():
        print(f"   {id}: {name}")

    print("\n📝 How to create labels:")
    print("1. Open each image in an image viewer")
    print("2. Note the bounding box coordinates: [x1, y1, x2, y2]")
    print("   - (x1, y1) = top-left corner")
    print("   - (x2, y2) = bottom-right corner")
    print("3. Run: create_label_file('image_name.jpg', class_id, [x1, y1, x2, y2])")
    print("\n💡 Example:")
    print("   create_label_file('gun.png', 0, [50, 30, 200, 150])")
    print("=" * 60)

if __name__ == "__main__":
    show_labeling_guide()

    # Example usage (you can modify these)
    print("\n🔧 Example label creation:")
    print("create_label_file('gun.png', 0, [50, 30, 200, 150])")
    print("create_label_file('knife.jpg', 1, [100, 80, 300, 250])")
    print("create_label_file('mobile.png', 2, [75, 60, 180, 200])")
