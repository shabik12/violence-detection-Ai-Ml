#!/usr/bin/env python3
"""
Final Knife Detection Test
"""

import cv2
import numpy as np
from yolo_enhanced import detect_objects

def create_knife_test_images():
    """Create multiple test images with knives"""

    print("🔪 Creating Knife Test Images")
    print("=" * 50)

    # Test 1: Clear knife on plain background
    img1 = np.zeros((480, 640, 3), dtype=np.uint8) + 255  # White background

    # Draw a clear knife
    # Blade
    cv2.rectangle(img1, (250, 200), (450, 220), (180, 180, 180), -1)
    # Point
    points = np.array([(450, 195), (470, 210), (450, 225)], np.int32)
    cv2.fillPoly(img1, [points], (180, 180, 180))
    # Handle
    cv2.rectangle(img1, (220, 205), (250, 215), (139, 69, 19), -1)
    # Add edge highlight
    cv2.line(img1, (250, 200), (450, 200), (100, 100, 100), 2)

    cv2.imwrite('knife_test_1.jpg', img1)

    # Test 2: Knife with hand (person holding knife)
    img2 = np.zeros((480, 640, 3), dtype=np.uint8) + 255

    # Draw simple person hand
    cv2.ellipse(img2, (320, 240), (40, 60), 0, 0, 360, (255, 220, 177), -1)  # Skin color

    # Draw knife in hand
    cv2.rectangle(img2, (340, 220), (480, 235), (180, 180, 180), -1)
    points = np.array([(480, 215), (500, 227), (480, 240)], np.int32)
    cv2.fillPoly(img2, [points], (180, 180, 180))
    cv2.rectangle(img2, (320, 225), (340, 230), (139, 69, 19), -1)

    cv2.imwrite('knife_test_2.jpg', img2)

    # Test 3: Multiple sharp objects
    img3 = np.zeros((480, 640, 3), dtype=np.uint8) + 255

    # Knife
    cv2.rectangle(img3, (100, 200), (200, 210), (180, 180, 180), -1)
    points = np.array([(200, 195), (220, 205), (200, 215)], np.int32)
    cv2.fillPoly(img3, [points], (180, 180, 180))

    # Fork
    for i in range(4):
        cv2.line(img3, (250, 200+i*5), (280, 200+i*5), (180, 180, 180), 2)
    cv2.rectangle(img3, (240, 210), (250, 220), (139, 69, 19), -1)

    # Scissors
    cv2.line(img3, (350, 200), (380, 230), (180, 180, 180), 3)
    cv2.line(img3, (350, 230), (380, 200), (180, 180, 180), 3)
    cv2.circle(img3, (365, 215), 5, (139, 69, 19), -1)

    cv2.imwrite('knife_test_3.jpg', img3)

    return [img1, img2, img3]

def test_knife_detection():
    """Test knife detection with multiple images"""

    images = create_knife_test_images()

    for i, img in enumerate(images, 1):
        print(f"\n🖼️ Testing Image {i}:")
        print("-" * 30)

        detections = detect_objects(img)

        if detections:
            print(f"✅ Detected {len(detections)} objects:")
            for detection in detections:
                label = detection['label']
                conf = detection['confidence']
                icon = "🔪" if label == 'knife' else "🔧" if label in ['fork', 'spoon', 'scissors'] else "👤"
                print(f"   {icon} {label}: {conf:.3f}")

                if label == 'knife':
                    print("   🎉 SUCCESS! Knife detected!")
        else:
            print("❌ No objects detected")

    print("\n" + "=" * 50)
    print("📝 SUMMARY:")
    print("• Enhanced YOLO detector with 15% confidence threshold")
    print("• Image preprocessing for sharp object detection")
    print("• Knife class ID: 43 (confirmed)")
    print("• Best results with real knife images")
    print("• Test images may not trigger detection")
    print("\n💡 For best knife detection:")
    print("   • Use real photos with clear knives")
    print("   • Good lighting and contrast")
    print("   • Knife should be visible and in focus")

if __name__ == "__main__":
    test_knife_detection()
