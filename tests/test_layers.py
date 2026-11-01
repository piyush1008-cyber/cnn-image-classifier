"""Unit tests for Conv2D, MaxPool2D, Flatten, Dense, and activation layers."""

import numpy as np
from src.layers import Conv2D, Dense, Flatten, MaxPool2D, ReLU, Softmax


def test_conv2d_forward_shape():
    # Input: batch=2, channels=1, H=8, W=8
    x = np.random.randn(2, 1, 8, 8)
    conv = Conv2D(num_filters=4, kernel_size=3, stride=1, padding=1)
    out = conv.forward(x)

    # Output should preserve H=8, W=8 with padding=1 and stride=1, with 4 channels
    assert out.shape == (2, 4, 8, 8)


def test_maxpool2d_forward_shape():
    # Input: batch=2, channels=4, H=8, W=8
    x = np.random.randn(2, 4, 8, 8)
    pool = MaxPool2D(pool_size=2, stride=2)
    out = pool.forward(x)

    # Output should downsample to 4x4
    assert out.shape == (2, 4, 4, 4)


def test_flatten_and_dense():
    x = np.random.randn(3, 4, 4, 4)
    flat = Flatten()
    flat_out = flat.forward(x)
    assert flat_out.shape == (3, 64)

    dense = Dense(in_features=64, out_features=10)
    dense_out = dense.forward(flat_out)
    assert dense_out.shape == (3, 10)


def test_softmax_probabilities():
    x = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.2]])
    softmax = Softmax()
    probs = softmax.forward(x)

    # Each row sum must equal 1.0
    row_sums = np.sum(probs, axis=1)
    np.testing.assert_allclose(row_sums, [1.0, 1.0], atol=1e-5)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
