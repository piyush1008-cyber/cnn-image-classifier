"""Convolutional Neural Network sequential architecture container."""

from __future__ import annotations

from typing import List, Tuple
import numpy as np
from src.layers import Conv2D, Dense, Flatten, Layer, MaxPool2D, ReLU, Softmax


class SequentialCNN:
    """Sequential Convolutional Neural Network container."""

    def __init__(self, layers: List[Layer]):
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Executes forward propagation through all stacked layers."""
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Computes categorical probability distribution."""
        return self.forward(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Returns predicted class index with highest confidence."""
        probs = self.predict_proba(x)
        return np.argmax(probs, axis=1)

    @classmethod
    def create_digit_classifier(cls, in_shape: Tuple[int, int, int] = (1, 8, 8), num_classes: int = 10) -> SequentialCNN:
        """Builds standard CNN architecture for 8x8 handwritten digits."""
        c, h, w = in_shape
        # Conv layer: 4 filters, 3x3 kernel, padding=1 -> output shape: (batch, 4, 8, 8)
        # MaxPool layer: 2x2, stride=2 -> output shape: (batch, 4, 4, 4)
        # Flatten -> 4 * 4 * 4 = 64 features
        # Dense -> 64 to 32 -> ReLU -> Dense 32 to 10 -> Softmax
        layers: List[Layer] = [
            Conv2D(num_filters=4, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2D(pool_size=2, stride=2),
            Flatten(),
            Dense(in_features=4 * 4 * 4, out_features=32),
            ReLU(),
            Dense(in_features=32, out_features=num_classes),
            Softmax(),
        ]
        return cls(layers)
