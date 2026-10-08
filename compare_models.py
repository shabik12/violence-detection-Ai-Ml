import numpy as np
import tensorflow as tf
from load_cnn_safe import load_cnn_model_safely

def test_model_file(model_path):
    print(f"\n--- Testing {model_path} ---")
    model = load_cnn_model_safely(model_path)
    if model:
        # Test with zero input
        x_zero = np.zeros((1, 224, 224, 3))
        p_zero = model.predict(x_zero, verbose=0)[0]

        # Test with random input
        np.random.seed(42)
        x_rand = np.random.random((1, 224, 224, 3)) * 255.0
        p_rand = model.predict(x_rand, verbose=0)[0]

        print(f"Zero input: MaxIdx={np.argmax(p_zero)}, Conf={np.max(p_zero):.4f}")
        print(f"Rand input: MaxIdx={np.argmax(p_rand)}, Conf={np.max(p_rand):.4f}")
        print(f"Probs: {p_zero}")
        return p_zero
    return None

test_model_file("CNN_video.h5")
test_model_file("best_model.h5")
