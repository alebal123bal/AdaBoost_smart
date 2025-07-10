"""
Test script for the AdaBoost implementation.
"""

import time
import numpy as np

from adaboost import AdaBoost, ClassifierScoreCheck, generate_random_data_numba

if __name__ == "__main__":
    # Set the seed for reproducibility
    np.random.seed(42)

    # Each row is a specific feature containing the feature evaluations on each image.
    FEATURE_EVAL_MATRIX = np.array(
        [
            [5, 10, 2, -1, 3],
            [-3, -6, 3, -2, 6],
            [10, 9, 4, 0, 9],
            [-7, 5, -2, 10, 6],
            # [9, 20, 20, 9, 9],
        ]
    )

    SAMPLE_WEIGHTS = np.array([0.20, 0.15, 0.15, 0.3, 0.2])

    SAMPLE_LABELS = np.array([1, -1, -1, 1, 1])

    # Try a big dataset
    FEATURE_EVAL_MATRIX, SAMPLE_WEIGHTS, SAMPLE_LABELS = generate_random_data_numba(
        size_x=1000, size_y=5000, bias_strenght=40
    )

    print("Feature Evaluation Matrix:")
    print(FEATURE_EVAL_MATRIX)

    print("Sample Weights:")
    print(SAMPLE_WEIGHTS)

    print("Sample Labels:")
    print(SAMPLE_LABELS)

    print("\n🔄 Training AdaBoost Classifier...\n")

    start_time = time.time()

    my_trainer = AdaBoost(
        feature_eval_matrix=FEATURE_EVAL_MATRIX,
        sample_weights=SAMPLE_WEIGHTS,
        sample_labels=SAMPLE_LABELS,
        n_stages=6,
        aggressivness=0.5,
        feature_per_stage=2,  # Number of features to select per stage
    )

    my_trainer.train()

    print("\n🎯 Training completed.\n")

    print(f"\n⏱️ Total training time: {(time.time() - start_time)} seconds")

    my_classifier = ClassifierScoreCheck(
        feature_eval_matrix=FEATURE_EVAL_MATRIX,
        sample_labels=SAMPLE_LABELS,
    )

    my_classifier.analyze()
