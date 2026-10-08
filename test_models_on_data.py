import os
import cv2
import numpy as np
import tensorflow as tf
from load_cnn_safe import load_cnn_model_safely

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

CATEGORIES = ["Abuse", "Arrest", "Assault", "Burglary", "Fighting", "Normal"]

def preprocess_frame(frame, target_size=(224, 224)):
    # Standard preprocessing for ConvNeXt
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame, target_size)
    # The models in load_cnn_safe.py perform their own normalization
    # but let's check if we should do it here too.
    # Actually, load_cnn_safe.py's compatible model DOES (x/127.5)-1.
    # If the standard model is loaded, it might expect 0-255 or -1 to 1.
    return np.expand_dims(frame_resized, axis=0)

def test_on_real_data(model_path):
    print(f"\n{'='*20} Testing {model_path} {'='*20}")
    model = load_cnn_model_safely(model_path)
    if model is None:
        print(f"Failed to load {model_path}")
        return

    dataset_dir = "Dataset"
    if not os.path.exists(dataset_dir):
        print("Dataset directory not found")
        return

    # Categories in the dataset
    dataset_cats = ["Abuse", "Arrest", "Assault", "Burglary", "Fighting", "normal"]

    for cat in dataset_cats:
        cat_path = os.path.join(dataset_dir, cat)
        if not os.path.isdir(cat_path):
            continue

        # Get first video or image
        files = [f for f in os.listdir(cat_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.mp4', '.avi'))]
        if not files:
            print(f"No files in {cat}")
            continue

        file_path = os.path.join(cat_path, files[0])

        if file_path.lower().endswith(('.mp4', '.avi')):
            cap = cv2.VideoCapture(file_path)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                print(f"Could not read video {file_path}")
                continue
        else:
            frame = cv2.imread(file_path)

        if frame is None:
            continue

        processed = preprocess_frame(frame)
        preds = model.predict(processed, verbose=0)[0]
        idx = np.argmax(preds)
        label = CATEGORIES[idx] if idx < len(CATEGORIES) else "Unknown"
        conf = preds[idx]

        print(f"Actual: {cat:10} | Predicted: {label:10} ({conf:.2f}) | Prob[Assault]: {preds[2]:.4f}")

if __name__ == "__main__":
    test_on_real_data("CNN_video.h5")
    test_on_real_data("best_model.h5")
    # test_on_real_data("CNN_video.keras") # Probably fails
