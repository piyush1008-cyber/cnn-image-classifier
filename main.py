"""Command-line interface and CNN pipeline demonstration."""

import argparse
import numpy as np
from src.dataset import load_normalized_digits
from src.metrics import compute_metrics, print_confusion_matrix
from src.network import SequentialCNN


def print_banner():
    banner = """
===================================================================
   CNN IMAGE CLASSIFIER - Computer Vision Convolution Pipeline     
===================================================================
    """
    print(banner)


def cmd_demo(args):
    print("[*] Loading handwritten digits dataset (8x8 grayscale images)...")
    x_train, x_test, y_train, y_test = load_normalized_digits(test_size=0.25)
    print(f"  - Training samples: {x_train.shape[0]} (Tensor: {x_train.shape})")
    print(f"  - Test samples:     {x_test.shape[0]} (Tensor: {x_test.shape})")

    print("\n[*] Initializing Sequential CNN Architecture:")
    print("  1. Conv2D      : 4 filters (3x3), stride=1, padding=1 -> Output: (batch, 4, 8, 8)")
    print("  2. ReLU        : Activation")
    print("  3. MaxPool2D   : 2x2 spatial downsampling -> Output: (batch, 4, 4, 4)")
    print("  4. Flatten     : 64 features")
    print("  5. Dense       : 64 -> 32")
    print("  6. ReLU        : Activation")
    print("  7. Dense       : 32 -> 10 classes")
    print("  8. Softmax     : Probability Distribution")

    model = SequentialCNN.create_digit_classifier()

    # Inference forward pass on test sample batch
    test_batch = x_test[:50]
    true_labels = y_test[:50]

    print(f"\n[*] Executing CNN forward pass on batch of {test_batch.shape[0]} images...")
    probs = model.predict_proba(test_batch)
    preds = model.predict(test_batch)

    print("\n--- Sample Predictions ---")
    print(f"{'INDEX':<6} | {'TRUE LABEL':<12} | {'PREDICTED':<12} | {'CONFIDENCE'}")
    print("-" * 50)
    for i in range(10):
        conf = probs[i, preds[i]] * 100
        print(f"{i:<6} | {true_labels[i]:<12} | {preds[i]:<12} | {conf:.2f}%")

    metrics = compute_metrics(true_labels, preds)
    print("\n" + "=" * 60)
    print("               CNN CLASSIFICATION DIAGNOSTICS             ")
    print("=" * 60)
    print(f"Evaluation Batch Size: {len(true_labels)} samples")
    print(f"Top-1 Accuracy:        {metrics['accuracy_pct']}%")

    print("\n--- Confusion Matrix (10 Classes) ---")
    print_confusion_matrix(metrics["confusion_matrix"])
    print("=" * 60)
    print("[PASS] CNN Image Classification demonstration completed successfully!\n")


def main():
    print_banner()
    parser = argparse.ArgumentParser(description="CNN Image Classifier - Computer Vision Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_demo = subparsers.add_parser("demo", help="Run automated CNN inference demonstration")
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
