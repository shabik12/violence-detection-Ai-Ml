import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import cv2
import os
from glob import glob

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess a single image for model prediction."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not load image {image_path}")
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = tf.keras.applications.convnext.preprocess_input(img)
    return img

def load_dataset_and_labels(dataset_path):
    """Load images and labels from the dataset directory."""
    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    classes.sort()

    images = []
    labels = []

    print("Loading dataset...")
    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name)
        image_files = glob(os.path.join(class_path, "*.mp4"))

        # For video files, we'll extract the first frame
        for video_file in image_files:
            try:
                cap = cv2.VideoCapture(video_file)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (224, 224))
                    images.append(frame)
                    labels.append(class_idx)
                cap.release()
            except Exception as e:
                print(f"Error processing {video_file}: {e}")
                continue

    # Also check for image files
        image_files = glob(os.path.join(class_path, "*.jpg")) + glob(os.path.join(class_path, "*.png"))
        for img_file in image_files:
            img = load_and_preprocess_image(img_file)
            if img is not None:
                images.append(img)
                labels.append(class_idx)

    return np.array(images), np.array(labels), classes

def evaluate_model(model_path, dataset_path):
    """Evaluate the model and generate confusion matrix and classification report."""

    # Load the trained model
    print(f"Loading model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load dataset
    images, labels, class_names = load_dataset_and_labels(dataset_path)

    if len(images) == 0:
        print("No images found in dataset!")
        return

    print(f"Loaded {len(images)} images from {len(class_names)} classes")
    print(f"Classes: {class_names}")

    # Preprocess images for model
    X_test = np.array([tf.keras.applications.convnext.preprocess_input(img) for img in images])
    y_test = labels

    # Make predictions
    print("Making predictions...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    report_text = classification_report(y_test, y_pred, target_names=class_names)

    # Create visualizations
    plt.figure(figsize=(15, 6))

    # Confusion Matrix Heatmap
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Classification Report Heatmap
    plt.subplot(1, 2, 2)
    report_df = pd.DataFrame(report).iloc[:-1, :].T  # Exclude 'accuracy' row
    sns.heatmap(report_df[['precision', 'recall', 'f1-score']].astype(float),
                annot=True, cmap='YlOrRd', fmt='.2f')
    plt.title('Classification Report Metrics')
    plt.tight_layout()

    # Save the plot
    plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print detailed results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nTotal samples: {len(y_test)}")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Macro avg F1-score: {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted avg F1-score: {report['weighted avg']['f1-score']:.4f}")

    print("\n" + "-"*40)
    print("CLASSIFICATION REPORT")
    print("-"*40)
    print(report_text)

    print("\n" + "-"*40)
    print("CONFUSION MATRIX")
    print("-"*40)
    print(cm)

    # Per-class metrics
    print("\n" + "-"*40)
    print("PER-CLASS METRICS")
    print("-"*40)
    for i, class_name in enumerate(class_names):
        precision = report[class_name]['precision']
        recall = report[class_name]['recall']
        f1 = report[class_name]['f1-score']
        support = report[class_name]['support']
        print(f"{class_name:12} | P: {precision:.3f} | R: {recall:.3f} | F1: {f1:.3f} | Support: {support}")

def main():
    """Main function to run the evaluation."""
    # Configuration
    MODEL_PATH = "CNN_video.keras"  # Change this to your model path
    DATASET_PATH = "Dataset"        # Path to your dataset directory

    # Alternative model paths to try if the main one fails
    alternative_models = ["best_model.h5", "CNN_video.h5"]

    print("Violence Detection Model Evaluation")
    print("="*50)

    # Try to load the main model first
    if os.path.exists(MODEL_PATH):
        evaluate_model(MODEL_PATH, DATASET_PATH)
    else:
        # Try alternative models
        model_loaded = False
        for alt_model in alternative_models:
            if os.path.exists(alt_model):
                print(f"Main model not found. Using {alt_model} instead.")
                evaluate_model(alt_model, DATASET_PATH)
                model_loaded = True
                break

        if not model_loaded:
            print("No trained model found! Please ensure one of the following exists:")
            print(f"- {MODEL_PATH}")
            for alt_model in alternative_models:
                print(f"- {alt_model}")

if __name__ == "__main__":
    # Import pandas for the classification report visualization
    try:
        import pandas as pd
    except ImportError:
        print("Installing pandas for better visualization...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
        import pandas as pd

    main()
