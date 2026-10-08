#!/usr/bin/env python3
"""
Create sample labels for the weapons dataset
"""

import os
import sys
sys.path.append('.')

from create_labels import create_label_file

# Create sample labels for your 3 images
print("Creating sample labels for your weapons dataset...")

# For gun.png - assuming gun is in the center area
create_label_file('gun.png', 0, [100, 80, 280, 200])

# For knife.jpg - assuming knife is diagonal
create_label_file('knife.jpg', 1, [120, 100, 320, 280])

# For mobile.png - assuming mobile is rectangular
create_label_file('mobile.png', 2, [90, 70, 210, 190])

print("\n✅ All label files created!")
print("\n📁 Dataset structure:")
print("datasets/weapons/")
print("├── images/train/")
print("│   ├── gun.png")
print("│   ├── knife.jpg")
print("│   └── mobile.png")
print("└── labels/train/")
print("    ├── gun.txt")
print("    ├── knife.txt")
print("    └── mobile.txt")
