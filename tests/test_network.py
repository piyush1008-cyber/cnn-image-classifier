"""Integration tests for SequentialCNN pipeline and dataset loading."""

import numpy as np
from src.dataset import load_normalized_digits
from src.metrics import compute_metrics
from src.network import SequentialCNN


def test_dataset_loader():
    x_train, x_test, y_train, y_test = load_normalized_digits(test_size=0.2)
    assert x_train.shape[1:] == (1, 8, 8)
    assert x_test.shape[1:] == (1, 8, 8)
    assert np.max(x_train) <= 1.0
    assert np.min(x_train) >= 0.0


def test_sequential_cnn_inference():
    model = SequentialCNN.create_digit_classifier()

    # Pass 5 sample images
    x = np.random.uniform(0.0, 1.0, size=(5, 1, 8, 8))
    probs = model.predict_proba(x)
    preds = model.predict(x)

    assert probs.shape == (5, 10)
    assert preds.shape == (5,)
    assert all(0 <= p <= 9 for p in preds)

    # Metrics computation
    dummy_true = np.array([0, 1, 2, 3, 4])
    metrics = compute_metrics(dummy_true, preds)
    assert "accuracy_pct" in metrics
    assert "confusion_matrix" in metrics
