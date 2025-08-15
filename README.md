# 🚀 Optimized AdaBoost Implementation

**High-performance AdaBoost classifier with Numba acceleration for large-scale machine learning tasks.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/numpy-required-orange.svg)](https://numpy.org/)
[![Numba](https://img.shields.io/badge/numba-accelerated-green.svg)](https://numba.pydata.org/)

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
from data.synthetic_data import SyntheticData
from classifiers.adaboost_trainer import AdaBoostTrainer

# Generate test data
feature_matrix, weights, labels = SyntheticData.generate_random_data_numba(
    size_x=5000,    # Number of samples
    size_y=10000,   # Number of features
    bias_strenght=30
)

# Train classifier
classifier = AdaBoostTrainer(
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
from classifiers.classifier_score_check import ClassifierScoreCheck

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

classifier = AdaBoostTrainer(
    feature_eval_matrix=feature_matrix,
    sample_weights=sample_weights,
    sample_labels=sample_labels,
    n_stages=3
)

classifier.train()
```

## 🧩 Package Usage & Imports

- For **standalone use**, run scripts from the project root using:
  ```sh
  python -m test.adaboost_test
  ```
- For **submodule use** (imported in another project), ensure the parent project adds `AdaBoost_smart` to `PYTHONPATH` or installs it as a package.

**Note:** Use absolute imports (as shown above) for best compatibility.

## 📁 Project Structure

```
AdaBoost_smart/
├── classifiers/
│   ├── adaboost_trainer.py
│   ├── classifier_score_check.py
│   └── __init__.py
├── data/
│   ├── synthetic_data.py
│   └── __init__.py
├── test/
│   ├── adaboost_test.py
│   └── __init__.py
├── utils/
│   ├── io_operations.py
│   ├── numba_setup.py
│   ├── statistics.py
│   └── __init__.py
├── _pickle_folder/
├── .vscode/
│   └── launch.json
├── README.md
└── __init__.py
```

## 🛠️ Configuration

### Debug Mode

Enable debugging to disable Numba compilation:

```bash
export ADABOOST_DEBUG=true
python -m test.adaboost_test
```

VS Code launch configurations:
- **"Debug AdaBoost Test"**: Numba disabled for debugging
- **"Run AdaBoost Test"**: Full optimization enabled

### Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `n_stages` | Number of training stages | 6 | 1-20 |
| `aggressivness` | Weight update aggressiveness | 1.0 | 0.1-2.0 |
| `bias_strenght` | Data generation bias | 20 | 1-50 |

## 📚 API Reference

See the code for detailed docstrings and usage examples.

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
Email: balzanalessandro2001@gmail.com  
Version: 3.0.0  
Date: 2025-08-15

---

*Built with ❤️ for high-performance machine learning and computer vision applications*