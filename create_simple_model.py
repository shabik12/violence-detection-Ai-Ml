import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ConvNeXtTiny
from tensorflow.keras.optimizers import Adam
import os

def create_simple_convnext_model(input_shape=(224, 224, 3), num_classes=6, freeze_backbone=True):
    """
    Create a simplified ConvNeXt-Tiny based model without LayerScale for compatibility.
    """
    # Load pre-trained ConvNeXt-Tiny model
    base_model = ConvNeXtTiny(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )

    # Freeze backbone if specified
    if freeze_backbone:
        for layer in base_model.layers:
            layer.trainable = False

    # Create model
    inputs = tf.keras.Input(shape=input_shape)

    # Preprocess input for ConvNeXt
    x = tf.keras.applications.convnext.preprocess_input(inputs)

    # Get features from backbone
    features = base_model(x, training=not freeze_backbone)

    # Add dropout for regularization
    x = layers.Dropout(0.5)(features)

    # Final classification layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    return model

def compile_model(model):
    """Compile the model with appropriate optimizer and loss."""
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Get number of classes from dataset
DATADIR = "Dataset"
CATEGORIES = sorted(os.listdir(DATADIR))
NUM_CLASSES = len(CATEGORIES)

print("Creating simplified ConvNeXt-Tiny model without LayerScale...")

# Create model
model = create_simple_convnext_model(
    input_shape=(224, 224, 3),
    num_classes=NUM_CLASSES,
    freeze_backbone=True
)

# Compile the model
model = compile_model(model)

# Save model
try:
    model.save('CNN_video.keras')
    model.save('CNN_video.h5')
    print("✅ Simplified model saved successfully!")
    print(f"   - Classes: {CATEGORIES}")
    print(f"   - Number of classes: {NUM_CLASSES}")
except Exception as e:
    print(f"❌ Error saving model: {e}")

print("\nModel is ready for loading with app.py")
