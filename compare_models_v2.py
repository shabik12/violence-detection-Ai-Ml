import numpy as np
import tensorflow as tf
from load_cnn_safe import load_cnn_model_safely

def test_model_file(model_path):
    print(f"\n--- Testing {model_path} ---")
    model = load_cnn_model_safely(model_path)
    if model:
        # Test with random input (fixed seed)
        np.random.seed(42)
        x_rand = np.random.random((1, 224, 224, 3)) * 255.0
        p_rand = model.predict(x_rand, verbose=0)[0]

        predicted_idx = np.argmax(p_rand)
        confidence = p_rand[predicted_idx]

        print(f"Rand input: MaxIdx={predicted_idx}, Conf={confidence:.4f}")
        # Print full probabilities with indices
        for i, prob in enumerate(p_rand):
            print(f"  Class {i}: {prob:.4f}")

    else:
        print("Failed to load model.")

test_model_file("CNN_video.h5")
test_model_file("best_model.h5")
test_model_file("CNN_video.keras")
print("\n--- Listing CATEGORIES ---")
import os
print(sorted(os.listdir("Dataset")))
