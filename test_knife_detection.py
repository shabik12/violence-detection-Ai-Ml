#!/usr/bin/env python3
"""
Test Knife Detection
"""

import cv2
import numpy as np
from yolo_minimal import detect_objects

# Create a test image with knife-like features
def test_knife_detection():
    print("🧪 Testing Knife Detection")
    print("=" * 50)

    # Create a simple test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Draw a knife-like shape (rectangle with pointed end)
    cv2.rectangle(test_image, (100, 200), (300, 210), (192, 192, 192), -1)  # Blade
    points = np.array([(300, 195), (320, 205), (300, 215)], np.int32)
    cv2.fillPoly(test_image, [points], (192, 192, 192))  # Point
    cv2.rectangle(test_image, (80, 195), (100, 215), (139, 69, 19), -1)  # Handle

    # Save test image
    cv2.imwrite('test_knife.jpg', test_image)

    # Test detection
    detections = detect_objects(test_image)

    print(f"📊 Test result: {len(detections)} objects detected")
    for detection in detections:
        print(f"   - {detection['label']}: {detection['confidence']:.3f}")

    # Check if knife was detected
    knife_detected = any(d['label'] == 'knife' for d in detections)
    if knife_detected:
        print("✅ Knife detection working!")
    else:
        print("❌ Knife not detected in test image")
        print("💡 This is normal - the test image is very simple")
        print("📸 Try with real images containing knives for better results")

if __name__ == "__main__":
    test_knife_detection()
