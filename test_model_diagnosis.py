#!/usr/bin/env python3
"""
Comprehensive Model Diagnosis Script
Tests the CNN model to identify why it always predicts "Assault"
"""

import os
import numpy as np
import tensorflow as tf
import cv2
from load_cnn_safe import load_cnn_model_safely

# Configuration
IMG_SIZE = 224
DATADIR = "Dataset"
CATEGORIES = ["Abuse", "Arrest", "Assault", "Burglary", "Fighting", "Normal"]

def convnext_preprocess(x):
    """Preprocess input for ConvNeXt - normalize to [-1, 1] range."""
    return (x / 127.5) - 1.0

def preprocess_frame(frame, target_size=(IMG_SIZE, IMG_SIZE)):
    """Preprocess frame exactly as used in training."""
    # Convert BGR (OpenCV) to RGB
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Handle other formats
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif len(frame.shape) == 3 and frame.shape[2] == 4:
        frame = frame[..., :3]

    # Resize to model input size
    frame_resized = cv2.resize(frame, target_size)

    # Use SAME preprocessing as model was trained with
    frame_preprocessed = convnext_preprocess(frame_resized)

    return np.expand_dims(frame_preprocessed, axis=0)

def extract_frame_from_video(video_path):
    """Extract a single frame from video for testing."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def test_model_with_samples():
    """Test model with sample frames from each class."""
    print("=== COMPREHENSIVE MODEL DIAGNOSIS ===")
    print(f"Expected class order: {CATEGORIES}")

    # Load model
    print("\n1. Loading model...")
    model = load_cnn_model_safely("CNN_video.h5")
    if model is None:
        print("❌ Failed to load model")
        return

    print("✅ Model loaded successfully")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")

    # Test with random input first
    print("\n2. Testing with random input...")
    random_input = np.random.random((1, IMG_SIZE, IMG_SIZE, 3))
    random_pred = model.predict(random_input, verbose=0)
    print(f"Random input prediction: {random_pred[0]}")
    print(f"Random predicted class: {np.argmax(random_pred[0])} ({CATEGORIES[np.argmax(random_pred[0])]})")

    # Test with actual dataset video frames
    print("\n3. Testing with actual dataset video frames...")
    for class_idx, category in enumerate(CATEGORIES):
        class_dir = os.path.join(DATADIR, category)
        if not os.path.exists(class_dir):
            print(f"⚠️ Directory not found: {class_dir}")
            continue

        # Find first video in the class directory
        video_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        if not video_files:
            print(f"⚠️ No videos found in {class_dir}")
            continue

        # Test first video
        video_path = os.path.join(class_dir, video_files[0])
        print(f"\n--- Testing {category} (index {class_idx}) ---")
        print(f"Video: {video_path}")

        # Extract frame from video
        frame = extract_frame_from_video(video_path)
        if frame is None:
            print(f"❌ Failed to extract frame from video")
            continue

        processed_frame = preprocess_frame(frame)

        # Predict
        predictions = model.predict(processed_frame, verbose=0)
        probs = predictions[0]
        predicted_class = np.argmax(probs)
        confidence = float(probs[predicted_class])
        predicted_label = CATEGORIES[predicted_class]

        # Show all probabilities
        print("All probabilities:")
        for i, (class_name, prob) in enumerate(zip(CATEGORIES, probs)):
            marker = "👉" if i == predicted_class else "  "
            print(f"{marker} {i}: {class_name} = {prob:.4f}")

        print(f"Predicted: {predicted_label} (index {predicted_class}) - Confidence: {confidence:.4f}")

        # Check if prediction is correct
        if predicted_class == class_idx:
            print("✅ CORRECT prediction")
        else:
            print(f"❌ INCORRECT prediction (expected {category})")

    print("\n=== DIAGNOSIS COMPLETE ===")
    print("\n🔍 ANALYSIS:")
    print("- If model always predicts same class (index 2 = Assault),")
    print("  the issue is likely in preprocessing or model architecture mismatch")
    print("- If predictions vary but are often wrong, model may need retraining")

if __name__ == "__main__":
    test_model_with_samples()
