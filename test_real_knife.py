#!/usr/bin/env python3
"""
Test Knife Detection with Real Image
"""

import cv2
import numpy as np
import urllib.request
from yolo_minimal import detect_objects

def test_real_knife_detection():
    print("🔪 Testing Knife Detection with Real Image")
    print("=" * 60)

    # Try to download a test image with knife
    test_urls = [
        "https://raw.githubusercontent.com/ultralytics/yolov5/master/test/images/zidane.jpg",  # Person with objects
        "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg",     # Multiple objects
    ]

    for i, url in enumerate(test_urls):
        try:
            print(f"\n📥 Downloading test image {i+1}...")
            urllib.request.urlretrieve(url, f'test_{i}.jpg')

            # Load and test
            img = cv2.imread(f'test_{i}.jpg')
            if img is not None:
                detections = detect_objects(img)

                print(f"📊 Image {i+1} results: {len(detections)} objects detected")
                for detection in detections:
                    print(f"   - {detection['label']}: {detection['confidence']:.3f}")

                    if detection['label'] == 'knife':
                        print("🎉 KNIFE DETECTED!")
                        return True
            else:
                print(f"❌ Failed to load image {i+1}")

        except Exception as e:
            print(f"❌ Error with image {i+1}: {e}")

    # Create a more realistic knife test
    print("\n🔧 Creating enhanced knife test image...")
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Draw a more realistic knife
    # Blade (silver/gray)
    cv2.rectangle(img, (200, 220), (400, 235), (192, 192, 192), -1)
    # Point
    points = np.array([(400, 215), (420, 227), (400, 240)], np.int32)
    cv2.fillPoly(img, [points], (192, 192, 192))
    # Handle (brown)
    cv2.rectangle(img, (180, 225), (200, 235), (101, 67, 33), -1)
    # Add some texture/shading
    cv2.line(img, (200, 220), (400, 220), (160, 160, 160), 1)
    cv2.line(img, (200, 235), (400, 235), (160, 160, 160), 1)

    cv2.imwrite('enhanced_knife_test.jpg', img)

    detections = detect_objects(img)
    print(f"\n📊 Enhanced test results: {len(detections)} objects detected")
    for detection in detections:
        print(f"   - {detection['label']}: {detection['confidence']:.3f}")

    knife_detected = any(d['label'] == 'knife' for d in detections)
    if knife_detected:
        print("🎉 KNIFE DETECTED in enhanced test!")
    else:
        print("\n💡 Important notes about knife detection:")
        print("   • YOLOv8n was trained on COCO dataset")
        print("   • Knife (ID 43) is included but may be rare")
        print("   • Best results with clear, real knife images")
        print("   • Test images may not trigger detection")
        print("   • Try with real photos containing knives")

    return knife_detected

if __name__ == "__main__":
    test_real_knife_detection()
