"""Neural network layers implementing forward propagation and transformations."""

from __future__ import annotations

import numpy as np


class Layer:
    """Base abstract neural network layer."""
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Conv2D(Layer):
    """2D Spatial Convolutional Layer with learnable filters."""

    def __init__(self, num_filters: int, kernel_size: int = 3, stride: int = 1, padding: int = 0):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = None
        self.bias = None

    def initialize(self, in_channels: int) -> None:
        scale = np.sqrt(2.0 / (in_channels * self.kernel_size * self.kernel_size))
        self.weights = np.random.randn(self.num_filters, in_channels, self.kernel_size, self.kernel_size) * scale
        self.bias = np.zeros(self.num_filters)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x shape: (batch_size, channels, H, W)
        batch_size, channels, h, w = x.shape
        if self.weights is None:
            self.initialize(channels)

        if self.padding > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode="constant")
        else:
            x_padded = x

        out_h = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (w + 2 * self.padding - self.kernel_size) // self.stride + 1
        output = np.zeros((batch_size, self.num_filters, out_h, out_w))

        for f in range(self.num_filters):
            kernel = self.weights[f]
            for i in range(out_h):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                for j in range(out_w):
                    w_start = j * self.stride
                    w_end = w_start + self.kernel_size

                    patch = x_padded[:, :, h_start:h_end, w_start:w_end]
                    output[:, f, i, j] = np.sum(patch * kernel, axis=(1, 2, 3)) + self.bias[f]

        return output


class MaxPool2D(Layer):
    """2D Spatial Max-Pooling Layer."""

    def __init__(self, pool_size: int = 2, stride: int = 2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x: np.ndarray) -> np.ndarray:
        batch_size, channels, h, w = x.shape
        out_h = (h - self.pool_size) // self.stride + 1
        out_w = (w - self.pool_size) // self.stride + 1
        output = np.zeros((batch_size, channels, out_h, out_w))

        for i in range(out_h):
            h_start = i * self.stride
            h_end = h_start + self.pool_size
            for j in range(out_w):
                w_start = j * self.stride
                w_end = w_start + self.pool_size

                patch = x[:, :, h_start:h_end, w_start:w_end]
                output[:, :, i, j] = np.max(patch, axis=(2, 3))

        return output


class Flatten(Layer):
    """Flattens spatial multi-channel tensor into a 2D matrix (batch_size, features)."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x.reshape(x.shape[0], -1)


class Dense(Layer):
    """Fully-connected Linear Dense Layer."""

    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        scale = np.sqrt(2.0 / in_features)
        self.weights = np.random.randn(in_features, out_features) * scale
        self.bias = np.zeros(out_features)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.dot(x, self.weights) + self.bias


class ReLU(Layer):
    """Rectified Linear Unit activation function."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)


class Softmax(Layer):
    """Numerically stable Softmax probability distribution."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Subtract max for numerical stability against overflow
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
