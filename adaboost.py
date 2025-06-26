"""
Optimized AdaBoost nummpy implementation
Author: Alessandro Balzan
Date: 2025-06-06
Version: 1.0.0
"""

import time
import pickle
import numpy as np

# Debug flag
DEBUG = False


## Random generation methods
def generate_random_data(size_x=5000, size_y=20000, bias_strenght=20):
    """
    Generate random data for testing the AdaBoost classifier.
    This function generates a random feature evaluation matrix, sample weights,

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
    if size_x <= 0 or size_y <= 0:
        raise ValueError("size_x and size_y must be positive integers.")
    if bias_strenght <= 0 or bias_strenght > 50:
        raise ValueError("bias_strenght must be a positive integer between 1 and 50.")

    # Generate a random feature evaluation matrix
    # Keep the positive samples values higher than the negative ones
    # so that classification makes sense
    _feature_eval_matrix = np.random.randint(
        low=-100, high=100, size=(size_x, size_y)
    ).astype(int)

    # Define split: 1/3 positive, 2/3 negative (more negatives than positives)
    positive_count = size_y // 3
    _ = size_y - positive_count

    # Boost positive samples values (first 1/3)
    _feature_eval_matrix[
        :, :positive_count
    ] += bias_strenght  # Positive samples get higher values
    # Reduce negative samples values (remaining 2/3)
    _feature_eval_matrix[
        :, positive_count:
    ] -= bias_strenght  # Negative samples get lower values

    # Generate sample weights - give positives higher initial weight
    _sample_weights = np.ones(size_y) / size_y  # Start uniform
    # Give more weight to positive samples (first 1/3)
    _sample_weights[:positive_count] *= 3  # Positive samples get 3x weight
    # Give less weight to negative samples (remaining 2/3)
    _sample_weights[positive_count:] *= 0.5  # Negative samples get 0.5x weight
    _sample_weights /= np.sum(_sample_weights)  # Normalize weights

    # Generate sample labels: 1/3 positive, 2/3 negative
    _sample_labels = np.ones(size_y, dtype=int)  # Start all positive
    _sample_labels[positive_count:] = -1  # Make last 2/3 negative

    return _feature_eval_matrix, _sample_weights, _sample_labels


## Pickle utils methods
def save_pickle_obj(obj, filename="trained_classifier.pkl"):
    """
    Save the classifier using pickle
    """
    with open(filename, "wb") as f:
        pickle.dump(obj, f)
    print(f"Object saved to {filename}")


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

        # Get the features involved in this stage
        stage_feature_idxs = [x["feature_idx"] for x in current_stage]

        # Get the evaluations for the sample
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

    def cascade_predictions(self, matrix, weights, labels):
        """
        Perform cascade predictions on all samples in the feature evaluation matrix.
        This method applies the majority voting for each sample across all stages
        and returns the predictions.

        Returns:
            numpy.ndarray: An array of predicted labels for all samples.
        """
        # Reload the matrix (instead of locally copying it)
        self.feature_eval_matrix = matrix
        self.sample_weights = weights
        self.sample_labels = labels

        # Get predictions for all stages
        predictions = np.array(
            [
                self.get_predictions(stage_idx=i)
                for i in range(len(self.trained_classifier))
            ]
        )

        # Binary decision: only the samples that passed all stages are considered positive
        all_ones_mask = np.sum(predictions == 1, axis=0)
        # Put it to 1 where it equals to the number of stages
        all_ones_mask = all_ones_mask == len(self.trained_classifier)

        # Final predictions: 1 for positive samples, -1 for negative samples
        final_predictions = np.where(all_ones_mask, 1, -1)

        # Compare with original sample labels
        print(
            "Final percentage of correct predictions in cascade:",
            np.mean(final_predictions == self.sample_labels) * 100,
            "%",
        )

        return final_predictions

    ## Training methods
    def find_best_feature(self):
        """
        Find the best feature given the feature evaluation matrix, sample weights and labels.

        Returns:
            tuple: A tuple containing:
                - best_idx (int): Index of the best feature.
                - best_threshold (float): Best threshold for the feature.
                - best_direction (int): Direction of the threshold (0 for "leq", 1 for "gt").
                - best_error (float): Weighted error of the best feature.
                - alpha (float): Amount of say for the best feature.
        """

        # Best thresholds vector, one for each feature
        best_thresholds = np.array([0] * self.feature_eval_matrix.shape[0], dtype=int)
        # Weighted error vector for using the threshold (both directions)
        best_errors = np.array([0] * self.feature_eval_matrix.shape[0], dtype=float)

        # Best directions vector, 0 for "lower than or equal to", 1 for "greater than"
        best_directions = np.array([0] * self.feature_eval_matrix.shape[0], dtype=int)

        for i, feature_eval in enumerate(self.feature_eval_matrix):
            # Order the feature evaluations
            ordered_idx = self.sorted_indices[i]

            # Compute signed weights
            signed_weights = self.sample_weights * self.sample_labels

            # Order the signed weights
            ordered_signed_weights = signed_weights[ordered_idx]

            # Optional: merge non unique feature evaluations
            unique_feature_eval, inverse_indices = np.unique(
                feature_eval[ordered_idx], return_inverse=True
            )
            # Merge the weights
            merged_weights = np.bincount(
                inverse_indices, weights=ordered_signed_weights
            )
            # Merge the labels
            merged_labels = np.bincount(
                inverse_indices, weights=self.sample_labels[ordered_idx]
            )
            # Where the label is now 0, set it to 1
            merged_labels[merged_labels == 0] = 1

            # Compute cumulative scores
            cumulative_scores = np.cumsum(merged_weights)

            # Use this formula for the weighted scores
            weighted_scores = cumulative_scores * 2 - cumulative_scores[-1]

            # Compute the weighted error (lower equal to)
            weighted_errors_leq = (1 - weighted_scores) / 2

            # Compute the weighted error (greater than)
            weighted_errors_gt = 1 - weighted_errors_leq

            # Find the index of the minimum weighted error (lower equal to)
            min_error_leq_idx = np.argmin(weighted_errors_leq)

            # Find the index of the minimum weighted error (greater than)
            min_error_gt_idx = np.argmin(weighted_errors_gt)

            # Compute feature value corresponding to the minimum weighted error (lower equal to)
            threshold_leq = unique_feature_eval[min_error_leq_idx]

            # Compute feature value corresponding to the minimum weighted error (greater than)
            threshold_gt = unique_feature_eval[min_error_gt_idx]

            # Select the minimum error between leq or gt
            min_error_leq = weighted_errors_leq[min_error_leq_idx]
            min_error_gt = weighted_errors_gt[min_error_gt_idx]

            if min_error_leq < min_error_gt:
                # Minimum error is better for "lower equal to"
                best_thresholds[i] = threshold_leq
                best_errors[i] = min_error_leq
                best_directions[i] = 0
            else:
                # Minimum error is better for "greater than"
                best_thresholds[i] = threshold_gt
                best_errors[i] = min_error_gt
                best_directions[i] = 1

        # Outside the loop, find the best feature
        best_idx = np.argmin(best_errors)

        # Compute amount of say ("alpha")
        epsilon = np.max([1e-10, best_errors[best_idx]])
        alpha = 0.5 * np.log((1 - epsilon) / (epsilon))

        return (
            best_idx,
            best_thresholds[best_idx],
            best_directions[best_idx],
            best_errors[best_idx],
            alpha,
        )

    def find_weight_update_array(self, feature_idx, threshold, direction):
        """
        Find the indexes of the samples that are classified incorrectly.
        "+1" for incorrectly classified samples; "-1" for correctly classified samples.

        Args:
            feature_idx (int): Index of the feature to evaluate.
            threshold (float): Threshold value for the feature evaluation.
            direction (int): 0 for "lower than or equal to", 1 for "greater than".

        Returns:
            numpy.ndarray: "+1" for incorrect samples; "-1" for correct ones
        """

        # Get the feature evaluation for the given feature index
        feature_eval = self.feature_eval_matrix[feature_idx]

        # Initialize an array assuming all wrong (+1) predictions
        weight_arr = np.ones(feature_eval.shape[0], dtype=int)

        # Compact version
        prediction_indexes = (
            (feature_eval <= threshold)
            if direction == 0
            else (feature_eval > threshold)
        )

        # Convert boolean indexes to -1 or 1
        prediction_indexes = prediction_indexes.astype(int) * 2 - 1

        # Set the weight array to 1 for the samples that are classified incorrectly
        right_indexes = prediction_indexes == self.sample_labels
        weight_arr[right_indexes] = -1

        return weight_arr

    def weight_update(self, weight_update_array, alpha):
        """
        Update and normalize sample weights according to the AdaBoost algorithm.

        Args:
            weight_update_array (numpy.ndarray): +1 for incorrect samples; -1 for correct ones
            alpha (float): Amount of say
        """

        # Update the sample weights
        self.sample_weights *= np.exp(alpha * weight_update_array)
        # Normalize the sample weights
        self.sample_weights /= np.sum(self.sample_weights)

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

    def print_statistics(self, predictions, stage_idx):
        """
        Print statistics about the predictions at the given stage.

        Args:
            predictions (numpy.ndarray): Array of predictions for the current stage.
            stage_idx (int): Index of the stage to print statistics for.
        """

        correct_predictions = np.mean(predictions == self.sample_labels) * 100
        true_positives = np.mean(predictions[self.sample_labels == 1] == 1) * 100
        true_negatives = np.mean(predictions[self.sample_labels == -1] == -1) * 100

        print(f"\nStatistics for stage {stage_idx + 1}:\n")

        print(
            f"Percentage of correct predictions at stage {stage_idx}:",
            correct_predictions,
            "%",
        )
        print(
            f"True positive percentage at stage {stage_idx}:",
            true_positives,
            "%",
        )

        print(
            f"True negative percentage at stage {stage_idx}:",
            true_negatives,
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
                    self.find_best_feature()
                )

                # Get the weight-update array based on the best feature
                weight_update_array = self.find_weight_update_array(
                    feature_idx=best_feature_idx,
                    threshold=best_threshold,
                    direction=best_direction,
                )

                # Update weights
                self.weight_update(
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
                    f"Stage {stage_i + 1}, iteration {x + 1} completed. "
                    f"Feature index: {best_feature_idx}, Threshold: {best_threshold}, "
                    f"Direction: {'>' if best_direction == 1 else '<='}, Alpha: {alpha:.4f}"
                )

                if np.abs(best_error) < 1e-10:
                    print(
                        f"\nPerfect feature found at stage {stage_i + 1}, iteration {x + 1}. "
                    )
                    break  # Stop if a perfect feature is found

            # Append this stage to the full classifier's list of stages
            self.trained_classifier.append(stage_classifier)

            if np.abs(best_error) < 1e-10:
                print("Stopping training early due to prefect feature.\n")
                break  # Stop training if a perfect feature is found

            # Get this stage predictions
            predictions = self.get_predictions(stage_idx=stage_i)

            # Print statistics for this stage
            self.print_statistics(predictions=predictions, stage_idx=stage_i)

            # Remove the samples and weights that are classified as negative by the majority vote
            self.crop_negatives(predictions=predictions)
            print("Cropped negatives from the feature evaluation matrix.\n")

        # Save the trained classifier to a file
        save_pickle_obj(
            filename="_pickle_folder/trained_classifier.pkl",
            obj=self.trained_classifier,
        )


## Test

if True:  # Set to True to run the test
    # Set the seed for reproducibility
    np.random.seed(42)

    # Each row is a specific feature.
    # This row array contains the feature evaluation on each image.
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
    FEATURE_EVAL_MATRIX, SAMPLE_WEIGHTS, SAMPLE_LABELS = generate_random_data(
        size_x=5000, size_y=10000, bias_strenght=40
    )

    print("Feature Evaluation Matrix:")
    print(FEATURE_EVAL_MATRIX)

    print("Sample Weights:")
    print(SAMPLE_WEIGHTS)

    print("Sample Labels:")
    print(SAMPLE_LABELS)

    print("\nTraining AdaBoost Classifier...\n")

    start_time = time.time()

    DEBUG = True
    my_trainer = AdaBoost(
        feature_eval_matrix=FEATURE_EVAL_MATRIX,
        sample_weights=SAMPLE_WEIGHTS,
        sample_labels=SAMPLE_LABELS,
        n_stages=6,
    )

    my_trainer.train()

    print("\nTraining completed.\n")

    print(f"\nTotal training time: {(time.time() - start_time)} seconds")
