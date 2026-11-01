# CNN Image Classifier

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: Pytest](https://img.shields.io/badge/tests-pytest-brightgreen.svg)](https://docs.pytest.org/)
[![Deep Learning: CNN Architecture](https://img.shields.io/badge/deep%20learning-CNN%20Architecture-purple.svg)](https://github.com/piyush1008-cyber/cnn-image-classifier)
[![Computer Vision: Conv2D & Pooling](https://img.shields.io/badge/vision-Conv2D%20%26%20MaxPool2D-blue.svg)](https://github.com/piyush1008-cyber/cnn-image-classifier)

> **Modular Convolutional Neural Network (CNN) computer vision classification pipeline written in Python.** Implements 2D spatial convolution (`Conv2D`), max-pooling downsampling (`MaxPool2D`), dense linear layers (`Dense`), ReLU activation, numerically stable Softmax probability output, and multi-class confusion matrix diagnostics.

---

## Network Architecture Overview

```mermaid
flowchart LR
    A[Input Image\n1 x 8 x 8 Tensor] --> B[Conv2D Layer\n4 Filters 3x3, Padding=1]
    B --> C[ReLU Activation]
    C --> D[MaxPool2D\n2x2 Downsampling, Stride=2]
    D --> E[Flatten Layer\n64 Features]
    E --> F[Dense Layer\n64 to 32 Units]
    F --> G[ReLU Activation]
    G --> H[Dense Output Layer\n32 to 10 Classes]
    H --> I[Softmax Function\nProbability Distribution]
    I --> J[Predicted Class Label\nArgmax 0 - 9]
```

---

## Mathematical Formulation

### 1. 2D Spatial Convolution
$$(\mathbf{X} * \mathbf{K})[i, j] = \sum_{m} \sum_{n} \mathbf{X}[i + m, j + n] \cdot \mathbf{K}[m, n] + b$$

### 2. Max Pooling
$$P[i, j] = \max_{0 \le m, n < S} \mathbf{X}[i \cdot S + m, j \cdot S + n]$$

### 3. Numerically Stable Softmax
$$\sigma(\mathbf{z})_i = \frac{e^{z_i - \max(\mathbf{z})}}{\sum_{j=1}^{C} e^{z_j - \max(\mathbf{z})}}$$

---

## Quickstart & CLI Commands

### Run Automated Demonstration
```bash
python main.py demo
```

Output:
```
===================================================================
   CNN IMAGE CLASSIFIER - Computer Vision Convolution Pipeline     
===================================================================
[*] Loading handwritten digits dataset (8x8 grayscale images)...
  - Training samples: 1347 (Tensor: (1347, 1, 8, 8))
  - Test samples:     450 (Tensor: (450, 1, 8, 8))

[*] Initializing Sequential CNN Architecture:
  1. Conv2D      : 4 filters (3x3), stride=1, padding=1 -> Output: (batch, 4, 8, 8)
  2. ReLU        : Activation
  3. MaxPool2D   : 2x2 spatial downsampling -> Output: (batch, 4, 4, 4)
  4. Flatten     : 64 features
  5. Dense       : 64 -> 32
  6. ReLU        : Activation
  7. Dense       : 32 -> 10 classes
  8. Softmax     : Probability Distribution

[*] Executing CNN forward pass on batch of 50 images...
```

---

## Running the Test Suite

```bash
pytest -v
```

Unit and integration tests verify:
- Conv2D spatial dimensions with padding and stride
- MaxPool2D downsampling ratios
- Flatten and Dense tensor dimension matrix multiplication
- Softmax probability normalization ($\sum p_i = 1.0$)
- End-to-end dataset loading and forward inference.

---

## Author & License

- **Author:** Piyush Kangle ([@piyush1008-cyber](https://github.com/piyush1008-cyber))
- **Email:** piyushkangle@gmail.com
- **License:** [MIT License](LICENSE)
