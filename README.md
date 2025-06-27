# AdaBoost_smart

A highly optimized NumPy implementation of the AdaBoost (Adaptive Boosting) algorithm for binary classification tasks.

**Author:** Alessandro Balzan  
**mail:** balzanalessandro2001@gmail.com

## Overview

This implementation provides an efficient AdaBoost classifier that uses weak learners (decision stumps) to create a strong classifier through iterative boosting. The algorithm is optimized for performance using vectorized NumPy operations and includes features like cascade classification and intelligent sample cropping.

## Features

- **Optimized Performance**: Vectorized operations using NumPy for fast, in-place computation
- **Memory Efficient**: Pre-computed sorted indices for feature evaluations
- **Cascade Classification**: Multi-stage classifier with progressive sample filtering
- **Flexible Weak Learners**: Decision stumps with automatic threshold selection
- **Persistence**: Save and load trained models using pickle
- **Comprehensive Statistics**: Detailed performance metrics for each training stage

## Installation

No special installation required beyond standard Python scientific libraries:

```python
import numpy as np
import pickle
import time
```

## Quick Start

```python
# Generate sample data
feature_matrix, weights, labels = generate_random_data(
    size_x=1000, size_y=5000, bias_strenght=30
)

# Initialize and train the classifier
classifier = AdaBoost(
    feature_eval_matrix=feature_matrix,
    sample_weights=weights,
    sample_labels=labels,
    n_stages=3
)

classifier.train()

# Make predictions
predictions = classifier.get_predictions(stage_idx=0)

# Save the trained model
save_pickle_obj(classifier.trained_classifier, "my_model.pkl")
```

## Core Algorithm: Finding Optimal Thresholds with the Integral Trick

The most computationally intensive part of AdaBoost is finding the optimal threshold for each feature. This implementation uses an efficient "integral trick" to compute weighted errors for all possible thresholds in a single pass.

### The Problem

For each feature, we need to find the threshold that minimizes the weighted classification error. Traditionally, this would require:
1. Trying every unique feature value as a potential threshold
2. For each threshold, computing predictions for all samples
3. Calculating the weighted error for each threshold
4. Selecting the threshold with minimum error

This naive approach has O(n²) complexity for n samples.

### The Integral Trick Solution

Our optimized approach reduces this to O(n log n) complexity using cumulative operations:

1. **Sort samples** by feature values: `O(n log n)`
2. **Compute signed weights**: `signed_weights = sample_weights * sample_labels`
3. **Handle duplicate values** by merging weights for identical feature values
4. **Compute cumulative scores**: `cumulative_scores = cumsum(merged_weights)`
5. **Apply the integral formula**: `weighted_scores = cumulative_scores * 2 - cumulative_scores[-1]`
6. **Calculate errors efficiently**: `weighted_errors = (1 - weighted_scores) / 2`

### Mathematical Foundation

The key insight is that the weighted error for a threshold can be expressed as:

```
error = (total_positive_weight - 2 * cumulative_positive_weight_below_threshold) / 2
```

By pre-computing cumulative sums, we can evaluate all thresholds simultaneously rather than iteratively.

### Implementation Details

```python
def find_best_feature(self):
    for i, feature_eval in enumerate(self.feature_eval_matrix):
        # Sort samples by feature values
        ordered_idx = self.sorted_indices[i]
        
        # Compute signed weights (positive for positive samples, negative for negative)
        signed_weights = self.sample_weights * self.sample_labels
        ordered_signed_weights = signed_weights[ordered_idx]
        
        # Merge weights for duplicate feature values
        unique_values, inverse_indices = np.unique(feature_eval[ordered_idx], return_inverse=True)
        merged_weights = np.bincount(inverse_indices, weights=ordered_signed_weights)
        
        # Apply the integral trick
        cumulative_scores = np.cumsum(merged_weights)
        weighted_scores = cumulative_scores * 2 - cumulative_scores[-1]
        
        # Compute errors for both threshold directions
        weighted_errors_leq = (1 - weighted_scores) / 2      # threshold ≤
        weighted_errors_gt = 1 - weighted_errors_leq         # threshold >
        
        # Select best threshold and direction
        # ... (threshold selection logic)
```

This approach provides significant speedup, especially for datasets with many samples, making the algorithm practical for large-scale applications.

## API Reference

### AdaBoost Class

#### Constructor
```python
AdaBoost(feature_eval_matrix, sample_weights, sample_labels, n_stages)
```

**Parameters:**
- `feature_eval_matrix` (numpy.ndarray): Matrix where each row represents a feature and each column represents a sample
- `sample_weights` (numpy.ndarray): Initial weights for each sample
- `sample_labels` (numpy.ndarray): Binary labels (+1 or -1) for each sample
- `n_stages` (int): Number of boosting stages to train

#### Key Methods

**Training:**
- `train()`: Train the complete AdaBoost classifier
- `find_best_feature()`: Find optimal feature and threshold for current weights
- `weight_update()`: Update sample weights based on classification errors

**Inference:**
- `get_predictions(stage_idx)`: Get predictions for a specific stage
- `majority_vote(sample_idx, stage_idx)`: Predict single sample using weighted voting
- `cascade_predictions(matrix, weights, labels)`: Run cascade classification

**Utilities:**
- `crop_negatives(predictions)`: Remove negative samples for next stage
- `get_statistics(predictions)`: Calculate performance metrics

### Utility Functions

**Data Generation:**
```python
generate_random_data(size_x=5000, size_y=20000, bias_strenght=20)
```
Generates synthetic data for testing with controllable class separation.

**Model Persistence:**
```python
save_pickle_obj(obj, filename="trained_classifier.pkl")
load_pickle_obj(filename="trained_classifier.pkl")
```

## Algorithm Details

### Cascade Architecture

The classifier uses a cascade structure where each stage progressively filters samples:
- **Stage 1**: Uses 2 weak learners
- **Stage 2**: Uses 4 weak learners  
- **Stage N**: Uses 2*(N+1) weak learners

Samples must pass ALL stages to be classified as positive, creating a high-precision classifier.

### Weak Learners

Each weak learner is a decision stump with:
- **Feature index**: Which feature to evaluate
- **Threshold**: Decision boundary value
- **Direction**: Either "≤" or ">" comparison
- **Alpha**: Voting weight based on classification accuracy

### Weight Update

Sample weights are updated using the standard AdaBoost formula:
```
new_weight = old_weight * exp(alpha * error_indicator)
```
Where `error_indicator` is +1 for misclassified samples and -1 for correct ones.

## Performance Optimizations

1. **Pre-computed Sorting**: Feature values are sorted once at initialization
2. **Vectorized Operations**: All computations use NumPy array operations
3. **Memory Reuse**: In-place weight updates and matrix operations
4. **Early Stopping**: Training stops when perfect accuracy is achieved
5. **Sample Cropping**: Negative samples are removed between stages

## Example Output

```
Training stage 1 of 3...
Finding best feature for stage 1, iteration 1...
Stage 1, iteration 1 completed.
Feature index: 2, Threshold: 4, Direction: >, Alpha: 0.4055, Error: 0.4

Statistics for stage 1:
Percentage of correct predictions at stage 0: 80.0 %
True positive percentage at stage 0: 100.0 %
True negative percentage at stage 0: 66.67 %

Cropped negatives from the feature evaluation matrix.
```

## License

This implementation is provided as-is for educational and research purposes.
