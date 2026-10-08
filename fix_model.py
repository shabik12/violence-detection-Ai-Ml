import tensorflow as tf
from model import create_convnext_model, compile_model
import os

# Get number of classes from dataset
DATADIR = "Dataset"
CATEGORIES = sorted(os.listdir(DATADIR))
NUM_CLASSES = len(CATEGORIES)

print("Creating new ConvNeXt-Tiny model with current TensorFlow version...")

# Create model with current TensorFlow/Keras version
model = create_convnext_model(
    input_shape=(224, 224, 3),
    num_classes=NUM_CLASSES,
    freeze_backbone=True
)

# Compile the model
model = compile_model(model)

# Save model in compatible format
try:
    model.save('CNN_video.keras', save_format='keras')
    model.save('CNN_video.h5', save_format='h5')
    print("✅ Model saved successfully in both formats!")
    print(f"   - CNN_video.keras")
    print(f"   - CNN_video.h5")
    print(f"   - Classes: {CATEGORIES}")
    print(f"   - Number of classes: {NUM_CLASSES}")
except Exception as e:
    print(f"❌ Error saving model: {e}")

print("\nModel is now ready for loading with app.py")
