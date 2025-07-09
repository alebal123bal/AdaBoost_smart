# 🚀 Optimized AdaBoost Implementation

**High-performance AdaBoost classifier with Numba acceleration for large-scale machine learning tasks.**

[[Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[[NumPy](https://img.shields.io/badge/numpy-required-orange.svg)](https://numpy.org/)
[[Numba](https://img.shields.io/badge/numba-accelerated-green.svg)](https://numba.pydata.org/)

## 📋 Overview

This project implements a highly optimized AdaBoost (Adaptive Boosting) classifier using NumPy and Numba JIT compilation. Designed for handling large feature matrices efficiently, it's particularly well-suited for computer vision tasks like face detection using Haar features.

The implementation uses weak learners (decision stumps) to create a strong classifier through iterative boosting, with advanced optimizations like the "integral trick" for O(n) threshold finding and cascade architecture for progressive sample filtering.

### ✨ Key Features

- **🏎️ Numba JIT Acceleration**: Up to 50x speedup over pure Python implementations
- **⚡ Integral Trick Optimization**: O(n) threshold finding instead of O(n²)
- **🔄 Debug Mode**: Switch between optimized and debug modes via environment variables
- **📊 Staged Training**: Progressive training with early stopping capabilities
- **🎯 Cascade Architecture**: Implements negative sample cropping for cascade classifiers
- **💾 Persistent Storage**: Automatic model saving/loading with pickle
- **📈 Real-time Statistics**: Comprehensive performance metrics during training
- **🧠 Memory Efficient**: Pre-computed sorted indices and vectorized operations

## 🚀 Quick Start

### Installation

```bash
pip install numpy numba
```

### Basic Usage

```python
import numpy as np
from adaboost import AdaBoost, generate_random_data_numba

# Generate test data
feature_matrix, weights, labels = generate_random_data_numba(
    size_x=5000,    # Number of features
    size_y=10000,   # Number of samples
    bias_strenght=30
)

# Train classifier
classifier = AdaBoost(
    feature_eval_matrix=feature_matrix,
    sample_weights=weights,
    sample_labels=labels,
    n_stages=6,
    aggressivness=1.0
)

classifier.train()
```

### Model Evaluation

```python
from adaboost import ClassifierScoreCheck

# Load and evaluate trained model
evaluator = ClassifierScoreCheck(
    feature_eval_matrix=feature_matrix,
    sample_labels=labels
)

evaluator.analyze()
```

### Simple Example

```python
# Quick example with small dataset
feature_matrix = np.array([
    [5, 10, 2, -1, 3],
    [-3, -6, 3, -2, 6],
    [10, 9, 4, 0, 9],
    [-7, 5, -2, 10, 6]
])

sample_weights = np.array([0.20, 0.15, 0.15, 0.3, 0.2])
sample_labels = np.array([1, -1, -1, 1, 1])

classifier = AdaBoost(
    feature_eval_matrix=feature_matrix,
    sample_weights=sample_weights,
    sample_labels=sample_labels,
    n_stages=3
)

classifier.train()
```

## 🔬 Core Algorithm: The Integral Trick for Optimal Thresholds

The most computationally intensive part of AdaBoost is finding optimal thresholds for each feature. Our implementation uses an efficient "integral trick" to reduce complexity from O(n²) to O(n).

### The Problem

For each feature, we need to find the threshold that minimizes weighted classification error. The naive approach would:
1. Try every unique feature value as a threshold
2. For each threshold, compute predictions for all samples  
3. Calculate weighted error for each threshold
4. Select the minimum error threshold

### The Integral Trick Solution

Our optimized approach uses cumulative operations:

1. **Sort samples** by feature values
2. **Compute signed weights**: `signed_weights = sample_weights * sample_labels`
3. **Calculate cumulative scores**: `cumulative_scores = cumsum(signed_weights)`
4. **Apply integral formula**: `weighted_scores = cumulative_scores * 2 - total_score`
5. **Calculate errors**: `weighted_errors = (1 - weighted_scores) / 2`

### Mathematical Foundation

**Weight Update Formula:**
```
w_i^(t+1) = w_i^(t) * exp(α_t * I(h_t(x_i) ≠ y_i))
```

**Alpha Calculation:**
```
α_t = 0.5 * log((1 - ε_t) / ε_t)
```

**Final Prediction:**
```
H(x) = sign(Σ α_t * h_t(x))
```

### Example: Finding Optimal Threshold

Let's walk through finding the optimal threshold for a feature using the integral method.

**Given Data:**
- Feature values: `[5, 10, 2, -1, 3]`
- Sample weights: `[0.20, 0.15, 0.15, 0.3, 0.2]`
- Sample labels: `[1, -1, -1, 1, 1]`

**Step 1: Sort samples by feature values**

| Feature Values | Sample Weights | Sample Labels |
|---------------|----------------|---------------|
| -1            | 0.3           | 1             |
| 2             | 0.15          | -1            |
| 3             | 0.2           | 1             |
| 5             | 0.2           | 1             |
| 10            | 0.15          | -1            |

**Step 2: Compute signed weights**
```python
signed_weights = [0.3×1, 0.15×(-1), 0.2×1, 0.2×1, 0.15×(-1)]
               = [0.3, -0.15, 0.2, 0.2, -0.15]
```

**Step 3: Calculate cumulative scores**
```python
cumulative_scores = [0.3, 0.15, 0.35, 0.55, 0.40]
```

**Step 4: Apply integral formula**
```python
final_sum = 0.40
weighted_scores = [0.3×2-0.40, 0.15×2-0.40, 0.35×2-0.40, 0.55×2-0.40, 0.40×2-0.40]
                = [0.2, -0.10, 0.30, 0.70, 0.40]
```

**Step 5: Convert to weighted errors**
```python
errors_leq = [(1-0.2)/2, (1-(-0.10))/2, (1-0.30)/2, (1-0.70)/2, (1-0.40)/2]
           = [0.40, 0.55, 0.35, 0.15, 0.30]
```

**Step 6: Find optimal threshold**
The minimum error is **0.15** at index 3, corresponding to threshold **≤ 5**.

This single-pass approach provides massive speedup for large datasets.

## 🛠️ Configuration

### Debug Mode

Enable debugging to disable Numba compilation:

```bash
export ADABOOST_DEBUG=true
python -m adaboost_test
```

VS Code launch configurations:
- **"Debug AdaBoost"**: Numba disabled for debugging
- **"Run AdaBoost"**: Full optimization enabled

### Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `n_stages` | Number of training stages | 6 | 1-20 |
| `aggressivness` | Weight update aggressiveness | 1.0 | 0.1-2.0 |
| `bias_strenght` | Data generation bias | 20 | 1-50 |

## 📊 Performance Benchmarks

### Training Times

Tested on Intel i7-8700K with 32GB RAM:

| Dataset Size | Features | Samples | Time (Debug) | Time (Optimized) |
|--------------|----------|---------|--------------|------------------|
| Small | 1K | 5K | ~5 minutes | ~30 seconds |
| Medium | 50K | 15K | ~2 hours | ~8 minutes |
| Large | 190K | 15K | ~8 hours | ~31 minutes |

### Complexity Analysis

- **Naive threshold finding**: O(n² × f) where n=samples, f=features
- **Optimized with integral trick**: O(n log n × f)
- **Memory usage**: ~2-4x feature matrix size
- **Numba speedup**: 10-50x over pure Python

## 🏗️ Architecture

### Cascade Structure

Each stage progressively filters samples:
- **Stage 1**: 2 weak learners
- **Stage 2**: 4 weak learners
- **Stage N**: 2×(N+1) weak learners

Samples must pass ALL stages to be classified as positive.

### Weak Learners

Each decision stump contains:
- **Feature index**: Which feature to evaluate
- **Threshold**: Decision boundary value  
- **Direction**: "≤" or ">" comparison
- **Alpha**: Voting weight based on accuracy

### Numba Optimizations

- **Parallel Processing**: Multi-threaded feature evaluation with `prange`
- **Memory Layout**: Contiguous arrays for cache efficiency
- **Type Specialization**: Compile-time type optimization
- **Loop Fusion**: Reduced memory allocations

## 📁 Project Structure

```
adaboost/
├── adaboost.py          # Main implementation
├── adaboost_test.py     # Test script and examples
├── .vscode/
│   └── launch.json      # Debug configurations
├── _pickle_folder/      # Saved models (auto-created)
└── README.md           # This file
```

## 📚 API Reference

### AdaBoost Class

#### Constructor
```python
AdaBoost(feature_eval_matrix, sample_weights, sample_labels, n_stages, **kwargs)
```

**Parameters:**
- `feature_eval_matrix` (np.ndarray): Feature evaluations [n_features × n_samples]
- `sample_weights` (np.ndarray): Initial sample weights
- `sample_labels` (np.ndarray): Binary labels (+1 or -1)
- `n_stages` (int): Number of boosting stages
- `aggressivness` (float): Weight update aggressiveness

#### Key Methods

**Training:**
```python
classifier.train()  # Train complete classifier
```

**Inference:**
```python
evaluator = ClassifierScoreCheck(feature_matrix, labels)
evaluator.analyze()  # Get performance statistics
predictions = evaluator.overall_predict()  # Get predictions
```

### Utility Functions

**Data Generation:**
```python
matrix, weights, labels = generate_random_data_numba(
    size_x=5000, size_y=10000, bias_strenght=30
)
```

**Model Persistence:**
```python
save_pickle_obj(classifier.trained_classifier, "model.pkl")
model = load_pickle_obj("model.pkl")
```

## 📈 Example Output

```
🚀 PRODUCTION MODE - Numba enabled

Feature Evaluation Matrix:
[[ 5 10  2 -1  3]
 [-3 -6  3 -2  6]
 [10  9  4  0  9]
 [-7  5 -2 10  6]]

🔄 Training AdaBoost Classifier...

✅ Done allocating memory for the AdaBoost classifier.
✅ Done precomputing sorted indices for feature evaluations.

🔄 Training stage 1 of 3...

🔍 Finding best feature for stage 1, iteration 1...
Stage 1, iteration 1 completed.
📏 Feature index: 0, Threshold: 5, Direction: <=, Alpha: 0.8673, Error: 0.15

📊 Statistics for stage 0:
⚖️ Percentage of correct predictions at stage 0: 80.0 %
📈 True positive percentage at stage 0: 66.67 %
📈 True negative percentage at stage 0: 100.0 %

🎉 Perfect stage 2. Stopping here.

💾 Classifier saved to _pickle_folder/trained_classifier.pkl

⏱️ Total training time: 0.022 seconds
```

## 🐛 Debugging & Troubleshooting

### Common Issues

1. **Float16 Error**: Ensure arrays use `float32` or `float64`
2. **Memory Issues**: Monitor RAM with large datasets  
3. **Numba Compilation**: First run includes compilation overhead
4. **Overfitting**: Use early stopping or reduce stages

### Debug Mode Features

```python
# Enable debugging
import os
os.environ['ADABOOST_DEBUG'] = 'true'

# Disable Numba for step-through debugging
# All @njit decorators become no-ops
# prange becomes regular range
```

## 🔧 Extending the Code

### Custom Feature Integration

```python
# Integrate custom features
def evaluate_custom_features(images, features):
    # Your feature evaluation logic
    return feature_matrix

# Use with existing pipeline
classifier = AdaBoost(
    feature_eval_matrix=custom_matrix,
    sample_weights=weights,
    sample_labels=labels,
    n_stages=6
)
```

### Performance Tuning

```python
# Adjust parallel threads
import numba
numba.set_num_threads(8)

# Memory management
import gc
gc.collect()  # Between stages for large datasets

# Optimize data types
feature_matrix = feature_matrix.astype(np.int16)  # Reduce memory
weights = weights.astype(np.float32)  # Numba compatibility
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Guidelines

- Maintain Numba compatibility
- Add debug mode support for new functions
- Include performance benchmarks
- Update documentation and examples

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Alessandro Balzan**
- Email: balzanalessandro2001@gmail.com
- Version: 2.0.0
- Date: 2025-07-04

---

*Built with ❤️ for high-performance machine learning and computer vision applications*