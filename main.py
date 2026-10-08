import cv2
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
from model import create_convnext_model, compile_model, create_data_augmentation, preprocess_image

# ================= CONFIGURATION ===================
DATADIR = "Dataset"  # folder containing category folders with videos
IMG_SIZE = 224  # resize frames for ConvNeXt
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = len(os.listdir(DATADIR))

# ================= VIDEO TO FRAMES ===================
CATEGORIES = sorted(os.listdir(DATADIR))
data = []

def extract_frames_from_video(video_path, max_frames_per_video=30, target_size=(IMG_SIZE, IMG_SIZE)):
    """Extract frames from a video file with resizing and preprocessing."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while len(frames) < max_frames_per_video:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames to get a good distribution
        frame_count += 1
        if frame_count % 5 != 0:  # Process every 5th frame
            continue

        # Convert to RGB and resize
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, target_size)
        frames.append(frame)

    cap.release()
    return frames

def load_dataset():
    """Load and preprocess the video dataset."""
    X, y = [], []

    for class_idx, category in enumerate(CATEGORIES):
        class_dir = os.path.join(DATADIR, category)
        print(f"Processing class: {category}")

        for video_file in os.listdir(class_dir):
            video_path = os.path.join(class_dir, video_file)
            try:
                frames = extract_frames_from_video(video_path)
                X.extend(frames)
                y.extend([class_idx] * len(frames))
                print(f"  - Processed {video_file}: {len(frames)} frames")
            except Exception as e:
                print(f"Error processing {video_file}: {e}")

    return np.array(X), np.array(y)

print("Loading dataset...")
X, y = load_dataset()
print(f"Loaded {len(X)} frames with {len(np.unique(y))} classes")

# ================= PREPARE DATA ===================
# Split into train and test/validation sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert to TensorFlow datasets
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size=1024)
val_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# Apply preprocessing and batching
def prepare(ds, augment=False):
    # Apply data augmentation only on training set
    if augment:
        data_augmentation = create_data_augmentation()
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                   num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

# Prepare datasets
train_ds = prepare(train_ds, augment=True)
val_ds = prepare(val_ds)

# ================= MODEL CREATION ===================
model = create_convnext_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=len(CATEGORIES))
model = compile_model(model)

# Callbacks
callbacks = [
    ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy'),
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.2, patience=3)
]

# ================= TRAINING ===================
print("Starting training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ================= SAVE MODEL ===================
model.save('CNN_video.keras')
print("Model saved as CNN_video.keras")

# ================= PLOTTING ===================
if 'history' in locals():
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    if 'accuracy' in history.history:
        plt.plot(history.history['accuracy'], label='train acc')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='val acc')
    plt.legend()
    plt.title('Model Accuracy')

    # Plot loss
    plt.subplot(1, 2, 2)
    if 'loss' in history.history:
        plt.plot(history.history['loss'], label='train loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='val loss')
    plt.legend()
    plt.title('Model Loss')

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# ================= EVALUATION ===================
try:
    # Evaluate on validation set
    if val_ds is not None:
        print("\nEvaluating on validation set...")
        val_loss, val_accuracy = model.evaluate(val_ds)
        print(f"Validation accuracy: {val_accuracy:.4f}")

        # Get predictions for confusion matrix
        y_true = np.concatenate([y for x, y in val_ds], axis=0)
        y_pred = model.predict(val_ds)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred_classes, target_names=CATEGORIES))

        # Plot confusion matrix
        cm = confusion_matrix(y_true, y_pred_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=CATEGORIES,
                   yticklabels=CATEGORIES)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()

except Exception as e:
    print(f"Error during evaluation: {e}")

print("\nTraining and evaluation completed!")

from sklearn import metrics

accuracy = metrics.accuracy_score(y_test, y_pred_classes)
print(f"Test Accuracy: {accuracy*100:.2f}%")

cm = metrics.confusion_matrix(y_test, y_pred_classes)
print("Confusion Matrix:")
print(cm)

# Sensitivity and specificity per class
for i, category in enumerate(CATEGORIES):
    TP = cm[i,i]
    FN = sum(cm[i,:]) - TP
    FP = sum(cm[:,i]) - TP
    TN = cm.sum() - (TP + FP + FN)

    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0

    print(f"\nCategory: {category}")
    print(f"  Sensitivity (Recall): {sensitivity:.2f}")
    print(f"  Specificity: {specificity:.2f}")

# Classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_classes, target_names=CATEGORIES))

# Confusion matrix plot
plt.figure(figsize=(8,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
