#!/usr/bin/env python3
"""
Test CNN Classification
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.convnext import preprocess_input

def test_cnn():
    """Test CNN model with a sample frame"""
    print("--- Testing CNN Classification ---")
    print("=" * 50)

    try:
        # Load CNN model
        print("[INFO] Loading CNN model...")
        cnn_model = tf.keras.models.load_model("CNN_video.keras")
        print("[SUCCESS] CNN model loaded successfully")

        # Create a test frame (same as camera frame)
        print("[INFO] Creating test frame...")
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print(f"   Frame shape: {test_frame.shape}")
        print(f"   Frame dtype: {test_frame.dtype}")

        # Test preprocessing
        print("[INFO] Testing preprocessing...")

        # Convert to RGB if grayscale
        if len(test_frame.shape) == 2:
            test_frame = cv2.cvtColor(test_frame, cv2.COLOR_GRAY2RGB)
        elif test_frame.shape[2] == 4:
            test_frame = test_frame[..., :3]

        print(f"   After format conversion: {test_frame.shape}")

        # Resize and preprocess for ConvNeXt
        IMG_SIZE = 224
        resized_frame = cv2.resize(test_frame, (IMG_SIZE, IMG_SIZE))
        print(f"   After resize: {resized_frame.shape}")

        processed_frame = preprocess_input(resized_frame)
        print(f"   After preprocess_input: {processed_frame.shape}")

        # Add batch dimension
        processed_frame = np.expand_dims(processed_frame, axis=0)
        print(f"   After adding batch dimension: {processed_frame.shape}")

        # Test prediction
        print("[INFO] Testing prediction...")
        predictions = cnn_model.predict(processed_frame, verbose=0)
        print(f"   Predictions shape: {predictions.shape}")

        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])

        # Get categories
        import os
        DATADIR = "Dataset"
        if os.path.exists(DATADIR):
            CATEGORIES = sorted(os.listdir(DATADIR))
            predicted_label = CATEGORIES[predicted_class]
            print(f"   Predicted class: {predicted_class} ({predicted_label})")
            print(f"   Confidence: {confidence:.2%}")
        else:
            print(f"   Predicted class index: {predicted_class}")
            print(f"   Confidence: {confidence:.2%}")

        print("[SUCCESS] CNN test completed successfully!")

        return True

    except Exception as e:
        print("[ERROR] CNN test failed:", e)
        import traceback
        traceback.print_exc()
        return False

def test_preprocess_function():
    """Test the preprocess_frame function"""
    print("\n[INFO] Testing preprocess_frame function...")

    try:
        # Import from app_complete
        import sys
        sys.path.append('.')
        from app_complete import preprocess_frame

        # Create test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Test preprocessing
        processed = preprocess_frame(test_frame)
        print(f"   Input shape: {test_frame.shape}")
        print(f"   Output shape: {processed.shape}")
        print(f"   Output dtype: {processed.dtype}")

        print("[SUCCESS] preprocess_frame function works correctly!")
        return True

    except Exception as e:
        print("[ERROR] preprocess_frame test failed:", e)
        return False

if __name__ == "__main__":
    print("CNN Classification Test Suite")
    print("=" * 60)

    # Test CNN model
    cnn_works = test_cnn()

    # Test preprocess function
    preprocess_works = test_preprocess_function()

    print("\n" + "=" * 60)
    print("📊 Test Results:")
    print(f"• CNN Model: {'✅ Working' if cnn_works else '❌ Failed'}")
    print(f"• Preprocess Function: {'✅ Working' if preprocess_works else '❌ Failed'}")

    if cnn_works and preprocess_works:
        print("\n🎉 All tests passed! CNN should work correctly in the app.")
    else:
        print("\n⚠️ Some tests failed. CNN may not work correctly in the app.")
