"""
Optimized AdaBoost numpy implementation
Author: Alessandro Balzan
Date: 2025-06-06
Version: 1.0.0
"""

import os
import time
import pickle
import numpy as np
from numba import njit, prange


## Pickle utils methods
def save_pickle_obj(obj, filename="trained_classifier.pkl"):
    """
    Save the classifier using pickle
    """
    # Get cwd
    cwd = os.getcwd()
    # Create folder if it does not exist
    folder = "_pickle_folder"
    if not os.path.exists(os.path.join(cwd, folder)):
        os.makedirs(os.path.join(cwd, folder))
    # Save the object to a file
    with open(os.path.join(cwd, folder, filename), "wb") as f:
        pickle.dump(obj, f)
    print(f"Classifier saved to {os.path.join(cwd, folder, filename)}")


def load_pickle_obj(filename="trained_classifier.pkl"):
    """
    Load the classifier from a file using pickle.
    If the file does not exist, it raises an error.
    Returns:
        classifier (object): The loaded classifier object.
    """

    try:
        with open(filename, "rb") as f:
            obj = pickle.load(f)

        if obj is None:
            raise ValueError(f"File '{filename}' is empty or corrupted.")

        return obj

    except FileNotFoundError as exc:
        raise FileNotFoundError(f"File '{filename}' not found.") from exc


## Random generation methods
@njit
def random_int_matrix(low, high, shape):  # Numba-compatible randint alternative
    return np.random.randint(low, high, size=shape)


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
    # Validation checks (Numba doesn't raise Python exceptions, so we use assert)
    assert size_x > 0 and size_y > 0, "size_x and size_y must be positive integers."
    assert 0 < bias_strenght <= 50, "bias_strenght must be between 1 and 50."

    # Initialize the random feature evaluation matrix
    _feature_eval_matrix = random_int_matrix(-50, 50, (size_x, size_y)).astype(np.int8)

    # Define split: 1/3 positive samples, 2/3 negative samples
    positive_count = size_y // 3
    negative_count = size_y - positive_count

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


@njit(parallel=True)
def find_best_feature_numba(
    feature_eval_matrix, sample_weights, sample_labels, sorted_indices
):
    """
    Find the best feature for the AdaBoost classifier using Numba.
    This function computes the best threshold and direction for each feature
    based on the weighted errors, and returns the feature index, threshold,
    direction, error, and alpha value.

    Args:
        feature_eval_matrix (numpy.ndarray): Matrix of feature evaluations.
        sample_weights (numpy.ndarray): Weights for each sample.
        sample_labels (numpy.ndarray): Labels for each sample.
        sorted_indices (numpy.ndarray): Precomputed sorted indices for each feature evaluation.

    Returns:
        tuple: A tuple containing:
            - best_idx (int): Index of the best feature.
            - best_threshold (float): Best threshold for the feature.
            - best_direction (int): Direction of the threshold (0 for <=, 1 for >).
            - best_error (float): Best error for the feature.
            - alpha (float): Alpha value for the feature.
    """

    n_features = feature_eval_matrix.shape[0]
    n_samples = feature_eval_matrix.shape[1]

    best_thresholds = np.zeros(n_features, dtype=feature_eval_matrix.dtype)
    best_errors = np.full(n_features, np.inf, dtype=np.float32)
    best_directions = np.zeros(n_features, dtype=np.int8)

    for i in prange(n_features):  # Parallelize across features
        feature_eval = feature_eval_matrix[i]
        ordered_idx = sorted_indices[i]
        signed_weights = sample_weights * sample_labels
        ordered_signed_weights = signed_weights[ordered_idx]

        # Compute cumulative scores
        cumulative_scores = np.cumsum(ordered_signed_weights)
        total_score = cumulative_scores[-1]
        weighted_scores = cumulative_scores * 2 - total_score

        # Compute weighted errors
        weighted_errors_leq = (1 - weighted_scores) / 2
        weighted_errors_gt = 1 - weighted_errors_leq

        # Find minimum weighted error
        min_error_leq_idx = np.argmin(weighted_errors_leq)
        min_error_gt_idx = np.argmin(weighted_errors_gt)

        min_error_leq = weighted_errors_leq[min_error_leq_idx]
        min_error_gt = weighted_errors_gt[min_error_gt_idx]

        if min_error_leq < min_error_gt:
            best_thresholds[i] = feature_eval[ordered_idx[min_error_leq_idx]]
            best_errors[i] = min_error_leq
            best_directions[i] = 0
        else:
            best_thresholds[i] = feature_eval[ordered_idx[min_error_gt_idx]]
            best_errors[i] = min_error_gt
            best_directions[i] = 1

    # Find global best feature
    best_idx = np.argmin(best_errors)
    epsilon = max(1e-10, best_errors[best_idx])  # Avoid zero division
    alpha = 0.5 * np.log((1 - epsilon) / epsilon)

    return (
        best_idx,
        best_thresholds[best_idx],
        best_directions[best_idx],
        best_errors[best_idx],
        alpha,
    )


@njit
def find_weight_update_array_numba(feature_eval, sample_labels, threshold, direction):
    """
    Find the indexes of the samples that are classified incorrectly.
    "+1" for incorrectly classified samples; "-1" for correctly classified samples.

    Args:
        feature_eval (np.ndarray): Evaluations of the feature for all samples.
        sample_labels (np.ndarray): True labels for the samples.
        threshold (float): Threshold value for the feature evaluation.
        direction (int): 0 for "lower than or equal to", 1 for "greater than".

    Returns:
        np.ndarray: "+1" for incorrect samples; "-1" for correct ones.
    """
    n_samples = feature_eval.shape[0]
    weight_arr = np.ones(n_samples, dtype=np.int8)

    # Iterate through each sample to compute predictions and weights
    for i in range(n_samples):
        # Perform threshold comparison based on direction
        if direction == 0:  # "lower than or equal to"
            prediction = 1 if feature_eval[i] <= threshold else -1
        else:  # "greater than"
            prediction = 1 if feature_eval[i] > threshold else -1

        # Mark as correct (-1) or incorrect (+1)
        if prediction == sample_labels[i]:
            weight_arr[i] = -1  # Correctly classified
        else:
            weight_arr[i] = 1  # Incorrectly classified

    return weight_arr


@njit
def weight_update_numba(sample_weights, weight_update_array, alpha):
    """
    Update and normalize sample weights according to the AdaBoost algorithm (Numba-optimized).

    Args:
        sample_weights (numpy.ndarray): Array of sample weights to be updated (in-place).
        weight_update_array (numpy.ndarray): +1 for incorrect samples; -1 for correct ones.
        alpha (float): Amount of say.

    Returns:
        None: The sample_weights array is updated in-place.
    """
    n_samples = sample_weights.shape[0]

    # Update sample weights
    for i in range(n_samples):
        sample_weights[i] *= np.exp(alpha * weight_update_array[i])

    # Normalize sample weights
    total_weight = np.sum(sample_weights)
    for i in range(n_samples):
        sample_weights[i] /= total_weight


class AdaBoost:
    """
    Optimized AdaBoost classifier using numpy.
    """

    def __init__(
        self,
        feature_eval_matrix: np.array,
        sample_weights: np.array,
        sample_labels: np.array,
        n_stages: int,
    ):
        """
        Initialize the AdaBoost classifier.

        Parameters:
            feature_eval_matrix (numpy.ndarray): Matrix of feature evaluations.
            sample_weights (numpy.ndarray): Weights for each sample.
            sample_labels (numpy.ndarray): Labels for each sample.
            n_stages (int): Number of stages for the AdaBoost algorithm.
        """

        self.n_stages = n_stages

        # Allocate memory for the feature evaluation matrix, sample weights and labels
        print("Allocating memory for the AdaBoost classifier...")
        self.feature_eval_matrix = feature_eval_matrix
        self.sample_weights = np.array(sample_weights)
        self.sample_labels = np.array(sample_labels)
        print("Done allocating memory for the AdaBoost classifier.\n")

        # Precomputed sorted indices for each feature evaluation
        print("Precomputing sorted indices for feature evaluations...")
        self.sorted_indices = np.argsort(self.feature_eval_matrix, axis=1)
        print("Done precomputing sorted indices for feature evaluations.\n")

        # Placeholder for the trained classifier
        self.trained_classifier = []

    ## Inference methods
    def majority_vote(self, sample_idx, stage_idx=0):
        """
        Perform majority voting on the specified sample index for the given stage.

        Args:
            sample_idx (int): Index of the sample to evaluate.
            stage_idx (int, optional): The index of the stage to evaluate. Defaults to 0.

        Returns:
            int: Predicted label for the sample based on the majority vote.
        """

        # Get the current stage of the trained classifier
        current_stage = self.trained_classifier[stage_idx]

        # Get the indexes of features involved in this stage
        stage_feature_idxs = [x["feature_idx"] for x in current_stage]

        # Get the evaluations at these feature indexes, for this sample
        sample_evaluations = self.feature_eval_matrix[stage_feature_idxs, sample_idx]

        # Do majority voting based on the evaluations and the thresholds
        predictions = np.array(
            [
                (
                    sample_evaluations[i] > x["threshold"]
                    if x["direction"] == 1
                    else sample_evaluations[i] <= x["threshold"]
                )
                for i, x in enumerate(current_stage)
            ]
        )

        # Convert boolean predictions to -1 or 1
        predictions = predictions.astype(int) * 2 - 1

        # Compute the weighted votes
        weighted_votes = predictions * np.array([x["alpha"] for x in current_stage])

        # Sum the weighted votes
        total_vote = np.sum(weighted_votes)

        # Determine the predicted label based on the sign of the total vote
        predicted_label = 1 if total_vote > 0 else -1

        return predicted_label

    def get_predictions(self, stage_idx: int):
        """
        Test the classifier on all samples in the feature evaluation matrix.
        This method applies the majority voting for each sample and returns the predictions.

        Args:
            stage_idx (int): The index of the stage to test on

        Returns:
            numpy.ndarray: An array of predicted labels for all samples
        """

        # Get predictions
        predictions = np.array(
            [
                self.majority_vote(
                    sample_idx=i,
                    stage_idx=stage_idx,
                )
                for i in range(self.feature_eval_matrix.shape[1])
            ]
        )

        return predictions

    ## Training methods

    def crop_negatives(self, predictions):
        """
        Updates the feature evaluation matrix, sample weights and labels
        to only include the samples classified as positive.
        Sort the feature evaluation matrix again after cropping.

        Args:
            predictions (numpy.ndarray): Array of predictions for the current stage.
        """

        # Find the samples that are classified as positive (1) by the majority vote
        positive_samples = predictions == 1

        # Remove the samples, weights and labels classified as negative by the majority vote
        self.feature_eval_matrix = self.feature_eval_matrix[:, positive_samples]
        self.sample_weights = self.sample_weights[positive_samples]
        self.sample_labels = self.sample_labels[positive_samples]

        # Sort again
        self.sorted_indices = np.argsort(self.feature_eval_matrix, axis=1)

    def get_statistics(self, predictions):
        """
        Return statistics from the given predictions.

        Args:
            predictions (numpy.ndarray): Array of predictions for the current stage.

        Returns:
            correct_predictions (float): Percentage of correct predictions.
            true_positives (float): Percentage of true positives.
            true_negatives (float): Percentage of true negatives.
        """

        correct_predictions = np.mean(predictions == self.sample_labels) * 100
        true_positives = np.mean(predictions[self.sample_labels == 1] == 1) * 100
        true_negatives = (
            np.mean(predictions[self.sample_labels == -1] == -1) * 100
            if -1
            in self.sample_labels  # Check if not all negative samples were cropped
            else 100.0
        )

        return correct_predictions, true_positives, true_negatives

    def print_statistics(
        self, stage_idx: int, corr_pred: float, true_pos: float, true_neg: float
    ):
        """
        Print statistics for this stage

        Args:
            stage_idx (int): Stage index
            corr_pred (float): Correct predictions
            true_pos (float): True positives
            true_neg (float): True negatives
        """

        print(f"\nStatistics for stage {stage_idx + 1}:\n")

        print(
            f"Percentage of correct predictions at stage {stage_idx}:",
            corr_pred,
            "%",
        )
        print(
            f"True positive percentage at stage {stage_idx}:",
            true_pos,
            "%",
        )

        print(
            f"True negative percentage at stage {stage_idx}:",
            true_neg,
            "%\n",
        )

    def train(self):
        """
        Train the AdaBoost classifier.
        """

        for stage_i in range(self.n_stages):
            print(f"Training stage {stage_i + 1} of {self.n_stages}...")

            stage_classifier = []  # Reset this stage's classifier

            for x in range(2 + 2 * stage_i):
                print(
                    f"Finding best feature for stage {stage_i + 1}, iteration {x + 1}..."
                )

                # Find the best feature
                best_feature_idx, best_threshold, best_direction, best_error, alpha = (
                    find_best_feature_numba(
                        feature_eval_matrix=self.feature_eval_matrix,
                        sample_weights=self.sample_weights,
                        sample_labels=self.sample_labels,
                        sorted_indices=self.sorted_indices,
                    )
                )

                # Get the weight-update array based on the best feature
                weight_update_array = find_weight_update_array_numba(
                    feature_eval=self.feature_eval_matrix[best_feature_idx],
                    sample_labels=self.sample_labels,
                    threshold=best_threshold,
                    direction=best_direction,
                )

                # Update weights
                weight_update_numba(
                    sample_weights=self.sample_weights,
                    weight_update_array=weight_update_array,
                    alpha=alpha,
                )

                # Append the feature to the stage's list of features
                stage_classifier.append(
                    {
                        "feature_idx": best_feature_idx,
                        "threshold": best_threshold,
                        "direction": best_direction,
                        "error": best_error,
                        "alpha": alpha,
                    }
                )

                print(
                    f"Stage {stage_i + 1}, iteration {x + 1} completed.\n"
                    f"Feature index: {best_feature_idx}, Threshold: {best_threshold}, "
                    f"Direction: {'>' if best_direction == 1 else '<='}, Alpha: {alpha:.4f}, "
                    f"Error: {best_error}"
                )

            # Append this stage to the full classifier's list of stages
            self.trained_classifier.append(stage_classifier)

            # Get this stage predictions
            predictions = self.get_predictions(stage_idx=stage_i)

            # Get statistics for this stage
            corr, tp, tn = self.get_statistics(predictions=predictions)

            # Print statistics
            self.print_statistics(
                stage_idx=stage_i, corr_pred=corr, true_pos=tp, true_neg=tn
            )

            # If the correct predictions are 100% for this stage, then stop early
            if corr > 99.9:
                print(f"Perfect stage {stage_i + 1}. Stopping here.\n")
                break

            # Remove the samples and weights that are classified as negative by the majority vote
            self.crop_negatives(predictions=predictions)
            print("Cropped negatives from the feature evaluation matrix.\n")

        # Save the trained classifier to a file
        save_pickle_obj(
            filename="trained_classifier.pkl",
            obj=self.trained_classifier,
        )


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
    # FEATURE_EVAL_MATRIX, SAMPLE_WEIGHTS, SAMPLE_LABELS = generate_random_data_numba(
    #     size_x=5000, size_y=10000, bias_strenght=40
    # )

    print("Feature Evaluation Matrix:")
    print(FEATURE_EVAL_MATRIX)

    print("Sample Weights:")
    print(SAMPLE_WEIGHTS)

    print("Sample Labels:")
    print(SAMPLE_LABELS)

    print("\nTraining AdaBoost Classifier...\n")

    start_time = time.time()

    my_trainer = AdaBoost(
        feature_eval_matrix=FEATURE_EVAL_MATRIX,
        sample_weights=SAMPLE_WEIGHTS,
        sample_labels=SAMPLE_LABELS,
        n_stages=6,
    )

    my_trainer.train()

    print("\nTraining completed.\n")

    print(f"\nTotal training time: {(time.time() - start_time)} seconds")
