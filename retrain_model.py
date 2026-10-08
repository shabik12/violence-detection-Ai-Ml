#!/usr/bin/env python3
"""
Complete Model Retraining Script
Fixes the corrupted model by training from scratch with proper architecture
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ConvNeXtTiny
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30
DATADIR = "Dataset"
CATEGORIES = ["Abuse", "Arrest", "Assault", "Burglary", "Fighting", "Normal"]
NUM_CLASSES = len(CATEGORIES)

def convnext_preprocess(x):
    """Preprocess input for ConvNeXt - normalize to [-1, 1] range."""
    return (x / 127.5) - 1.0

def extract_frames_from_video(video_path, max_frames=20, target_size=(IMG_SIZE, IMG_SIZE)):
    """Extract frames from video with proper preprocessing."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame interval to get evenly distributed frames
    if total_frames > 0:
        interval = max(1, total_frames // max_frames)
    else:
        interval = 5

    while len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize
        frame = cv2.resize(frame, target_size)
        frames.append(frame)

        frame_count += interval

    cap.release()
    return frames

def load_dataset():
    """Load and preprocess the video dataset."""
    X, y = [], []

    print("Loading dataset...")
    for class_idx, category in enumerate(CATEGORIES):
        class_dir = os.path.join(DATADIR, category)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} not found")
            continue

        print(f"Processing class: {category} (index {class_idx})")

        video_files = [f for f in os.listdir(class_dir)
                     if f.lower().endswith(('.mp4', '.avi', '.mov'))]

        for video_file in video_files:
            video_path = os.path.join(class_dir, video_file)
            try:
                frames = extract_frames_from_video(video_path, max_frames=15)
                X.extend(frames)
                y.extend([class_idx] * len(frames))
                print(f"  - {video_file}: {len(frames)} frames")
            except Exception as e:
                print(f"Error processing {video_file}: {e}")

    return np.array(X), np.array(y)

def create_model():
    """Create a fresh ConvNeXt model with correct architecture."""
    print("Creating new ConvNeXt model...")

    # Load pre-trained ConvNeXt-Tiny
    base_model = ConvNeXtTiny(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling='avg'
    )

    # Freeze backbone initially
    for layer in base_model.layers:
        layer.trainable = False

    # Create model
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Apply our preprocessing
    x = layers.Lambda(convnext_preprocess)(inputs)

    # Get features
    features = base_model(x, training=False)

    # Add classification head
    x = layers.Dropout(0.5)(features)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model, base_model

def main():
    print("=== VIOLENCE DETECTION MODEL RETRAINING ===")
    print(f"Classes: {CATEGORIES}")
    print(f"Number of classes: {NUM_CLASSES}")

    # Load dataset
    X, y = load_dataset()
    print(f"\nLoaded {len(X)} frames")
    print(f"Class distribution:")
    for i, category in enumerate(CATEGORIES):
        count = np.sum(y == i)
        print(f"  {category}: {count} frames")

    # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set: {len(X_train)} frames")
    print(f"Validation set: {len(X_val)} frames")

    # Create model
    model, base_model = create_model()

    # Callbacks
    callbacks = [
        ModelCheckpoint('CNN_video_new.h5', save_best_only=True,
                    monitor='val_accuracy', mode='max'),
        EarlyStopping(patience=8, restore_best_weights=True,
                    monitor='val_accuracy', mode='max'),
        ReduceLROnPlateau(factor=0.5, patience=4,
                         monitor='val_accuracy', mode='max')
    ]

    # Phase 1: Train with frozen backbone
    print("\n=== PHASE 1: Training with frozen backbone ===")
    history1 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Phase 2: Fine-tune with unfrozen backbone
    print("\n=== PHASE 2: Fine-tuning with unfrozen backbone ===")

    # Unfreeze last few layers
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history2 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    model.save('CNN_video_fixed.keras')
    print("\n✅ Model saved as 'CNN_video_fixed.keras'")

    # Test the model
    print("\n=== TESTING TRAINED MODEL ===")
    test_predictions = model.predict(X_val[:10], verbose=0)
    for i in range(10):
        pred_class = np.argmax(test_predictions[i])
        true_class = y_val[i]
        pred_label = CATEGORIES[pred_class]
        true_label = CATEGORIES[true_class]
        confidence = float(test_predictions[i][pred_class])

        status = "✅" if pred_class == true_class else "❌"
        print(f"{status} True: {true_label} | Predicted: {pred_label} ({confidence:.3f})")

    print("\n🎉 RETRAINING COMPLETE!")
    print("Use 'CNN_video_fixed.keras' as your new model file")

if __name__ == "__main__":
    main()
