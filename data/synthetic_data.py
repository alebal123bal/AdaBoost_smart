"""
Synthetic data generation for testing the AdaBoost classifier.
"""

import numpy as np
from utils.numba_setup import njit


class SyntheticData:
    """
    Synthetic data generation for testing the AdaBoost classifier.
    """

    @staticmethod
    @njit
    def generate_random_data_numba(size_x=5000, size_y=20000, bias_strenght=20):
        """
        Generate random data for testing the AdaBoost classifier (Numba-optimized).

        Args:
            size_x (int, optional): Columns. Defaults to 5000.
            size_y (int, optional): Rows. Defaults to 20000.
            bias_strenght (int, optional): Bias strength to differentiate positive and negative samples.
                Defaults to 20. Not more than 50.

        Returns:
            tuple: A tuple containing:
                - feature_eval_matrix (numpy.ndarray): Randomly generated feature evaluation matrix.
                - sample_weights (numpy.ndarray): Randomly generated sample weights.
                - sample_labels (numpy.ndarray): Randomly generated sample labels.
        """

        # Initialize the random feature evaluation matrix
        _feature_eval_matrix = np.random.randint(-100, 100, (size_x, size_y)).astype(
            np.int16
        )

        # Define split: 1/3 positive samples, 2/3 negative samples
        positive_count = size_y // 3
        negative_count = size_y - positive_count  # pylint: disable=unused-variable

        # Apply bias: Boost positive samples and reduce negative samples
        for i in range(size_x):
            for j in range(positive_count):
                _feature_eval_matrix[i, j] += bias_strenght  # Boost positive samples
            for j in range(positive_count, size_y):
                _feature_eval_matrix[i, j] -= bias_strenght  # Reduce negative samples

        # Initialize sample weights (start uniform)
        _sample_weights = np.ones(size_y) / size_y

        # Adjust weights: Give positives higher weight and negatives lower weight
        for i in range(positive_count):
            _sample_weights[i] *= 3  # Positive samples get 3x weight
        for i in range(positive_count, size_y):
            _sample_weights[i] *= 0.5  # Negative samples get 0.5x weight

        # Normalize weights
        total_weight = np.sum(_sample_weights)
        for i in range(size_y):
            _sample_weights[i] /= total_weight

        # Generate sample labels: 1 for positives, -1 for negatives
        _sample_labels = np.ones(size_y, dtype=np.int8)
        for i in range(positive_count, size_y):
            _sample_labels[i] = -1  # Negative samples

        return _feature_eval_matrix, _sample_weights, _sample_labels
