"""Dataset loader and image tensor preprocessor."""

from typing import Tuple
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def load_normalized_digits(test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads 8x8 handwritten digits dataset normalized as (N, 1, 8, 8) image tensors."""
    digits = load_digits()
    x = digits.images  # (1797, 8, 8)
    y = digits.target  # (1797,)

    # Normalize pixel intensity from 0-16 to 0.0 - 1.0
    x_norm = x.astype(np.float32) / 16.0

    # Expand channel dimension: (N, 1, 8, 8)
    x_tensors = np.expand_dims(x_norm, axis=1)

    x_train, x_test, y_train, y_test = train_test_split(
        x_tensors, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return x_train, x_test, y_train, y_test
