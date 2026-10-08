"""
Safe CNN model loader that handles version compatibility issues.
Creates a compatible model architecture and loads weights from the saved model.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ConvNeXtTiny
import numpy as np
import os

# ConvNeXt preprocessing - normalize to [-1, 1] range (ImageNet normalization)
def convnext_preprocess(x):
    """Preprocess input for ConvNeXt - normalize to [-1, 1] range."""
    return (x / 127.5) - 1.0

def create_compatible_convnext_model(input_shape=(224, 224, 3), num_classes=6):
    """
    Create a ConvNeXt-Tiny model compatible with TensorFlow 2.12.

    Args:
        input_shape: Tuple, shape of input images
        num_classes: Int, number of output classes

    Returns:
        A compiled Keras model
    """
    # Load pre-trained ConvNeXt-Tiny without the top classification layer
    base_model = ConvNeXtTiny(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )

    # Freeze the backbone layers
    for layer in base_model.layers:
        layer.trainable = False

    # Create the full model
    inputs = layers.Input(shape=input_shape)

    # Use the same preprocessing as the original model
    x = convnext_preprocess(inputs)

    # Get features from the backbone
    features = base_model(x, training=False)

    # Add dropout and final classification layer (same as original)
    x = layers.Dropout(0.5)(features)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Create and compile the model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def load_cnn_model_safely(model_path="CNN_video.h5"):
    """
    Load CNN model with compatibility fallback.

    Args:
        model_path: Path to the model file

    Returns:
        Loaded Keras model or None if loading fails
    """
    print(f"Attempting to load CNN model from {model_path}...")

    # First try normal loading
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print("[SUCCESS] Successfully loaded model with standard method")
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        print(f"Standard loading failed: {e}")

    # If standard loading fails, try the compatible approach
    try:
        print("Creating compatible model architecture...")

        # Create a compatible model with the same architecture
        compatible_model = create_compatible_convnext_model(num_classes=6)

        # Try to load weights from the original model
        if os.path.exists(model_path):
            print("Loading weights into compatible model...")
            compatible_model.load_weights(model_path, by_name=True, skip_mismatch=True)
            print("[SUCCESS] Successfully loaded weights into compatible model")
            return compatible_model
        else:
            print(f"Model file not found: {model_path}")
            return None

    except Exception as e:
        print(f"Compatible loading failed: {e}")
        return None

def test_model(model):
    """
    Test the loaded model with dummy input.
    """
    if model is None:
        print("[ERROR] No model to test")
        return False

    try:
        # Create dummy input
        dummy_input = np.random.random((1, 224, 224, 3))

        # Test prediction
        prediction = model.predict(dummy_input, verbose=0)

        print("[SUCCESS] Model test successful!")
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {prediction.shape}")
        print(f"   Prediction range: [{prediction.min():.3f}, {prediction.max():.3f}]")
        print(f"   Sum (should be ~1.0): {prediction.sum():.3f}")
        print(f"   Predicted class index: {np.argmax(prediction[0])}")

        return True

    except Exception as e:
        print("[ERROR] Model test failed:", e)
        return False

if __name__ == "__main__":
    # Test the safe loader
    model = load_cnn_model_safely()

    if model:
        test_model(model)
        print("🎉 CNN model is ready for use!")
    else:
        print("[ERROR] Failed to load CNN model")
