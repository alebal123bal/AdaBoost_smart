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


## Pickel utils methods
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
        feature_eval_matrix,
        sample_weights,
        sample_labels,
        n_stages=5,
    ):
        """
        Initialize the AdaBoost classifier.

        Parameters:
        feature_eval_matrix (numpy.ndarray): Matrix of feature evaluations.
        sample_weights (numpy.ndarray): Weights for each sample.
        sample_labels (numpy.ndarray): Labels for each sample.
        """
        self.n_stages = n_stages

        print("Allocating memory for the AdaBoost classifier...")
        self.feature_eval_matrix = feature_eval_matrix
        self.sample_weights = np.array(sample_weights)
        self.sample_labels = np.array(sample_labels)

        # Precomputed sorted indices for each feature evaluation
        self.sorted_indices = np.argsort(self.feature_eval_matrix, axis=1)

        print("Done allocating memory for the AdaBoost classifier.\n")

        self.trained_classifier = []  # Placeholder for the trained classifier

    ## Inference methods
    def majority_vote(self, sample_idx, stage_idx=0):
        """
        Perform majority voting on the specified sample (could be image etc...).
        Valid for a single stage only out of the many stages of the AdaBoost algorithm.

        Args:
            sample_idx (_type_): Index of the sample to perform majority voting on.
        Returns:
            int: The predicted label for the sample based on majority voting.
        """
        # Choose if using the non cropped data or the cropped one
        eval_matrix = self.feature_eval_matrix
        # Get the current stage of the trained classifier
        current_stage = self.trained_classifier[stage_idx]

        # Get the features involved in this stage
        stage_feature_idxs = [x["feature_idx"] for x in current_stage]

        # Get the evaluations for the sample
        sample_evaluations = eval_matrix[stage_feature_idxs, sample_idx]

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

    def get_predictions(self, stage_idx=0):
        """
        Test the classifier on all samples in the feature evaluation matrix.
        This method applies the majority voting for each sample and returns the predictions.

        Args:
            stage_idx (int, optional): The index of the stage to test on. Defaults to 0.

        Returns:
            numpy.ndarray: An array of predicted labels for all samples.
        """
        # Choose if using the non cropped data or the cropped one

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

        print(
            f"Percentage of correct predictions at stage {stage_idx}:",
            np.mean(predictions == self.sample_labels) * 100,
            "%",
        )
        print(
            f"True positive percentage at stage {stage_idx}:",
            np.mean(predictions[self.sample_labels == 1] == 1) * 100,
            "%",
        )

        print(
            f"True negative percentage at stage {stage_idx}:",
            np.mean(predictions[self.sample_labels == -1] == -1) * 100,
            "%\n",
        )

        if DEBUG:
            print(predictions)

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

        if DEBUG:
            print("Final cascade predictions:")
            print(final_predictions)
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
            # Merge the weights as well
            merged_weights = np.bincount(
                inverse_indices, weights=ordered_signed_weights
            )
            # Merge the labels as well
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

            # Find the index of the minimum weighted error
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
                # If the minimum error is better for "lower equal to"
                best_thresholds[i] = threshold_leq
                best_errors[i] = min_error_leq
                best_directions[i] = 0
            else:
                # If the minimum error is better for "greater than"
                best_thresholds[i] = threshold_gt
                best_errors[i] = min_error_gt
                best_directions[i] = 1

        # Outside the loop, find the best feature
        best_idx = np.argmin(best_errors)
        # Compute amount of say ("alpha")
        epsilon = np.max([1e-10, best_errors[best_idx]])
        alpha = 0.5 * np.log((1 - epsilon) / (epsilon))

        if DEBUG:
            # Print the threshold and the corresponding minimum weighted error
            print(f"Best feature: index {best_idx}")
            print(self.feature_eval_matrix[best_idx])
            print("With threshold:")
            print(best_thresholds[best_idx])
            print("And direction (0 for lower equal to, 1 for greater than):")
            print(best_directions[best_idx])
            print("And minimum weighted error:")
            print(best_errors[best_idx])
            print("And alpha:")
            print(alpha)

        return best_idx, best_thresholds[best_idx], best_directions[best_idx], alpha

    def find_weight_update_array(self, feature_idx, threshold, direction):
        """
        Find the indexes of the samples that are classified incorrectly
        based on the given feature index, threshold, and direction.
        This array has the same length as the number of samples in the dataset.
        It has a "1" for the samples that are classified incorrectly
        and a "-1" for the samples that are classified correctly.
        Useful for later steps in the AdaBoost algorithm.
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
        prediction_indexes = prediction_indexes.astype(int) * 2 - 1
        right_indexes = prediction_indexes == self.sample_labels
        weight_arr[right_indexes] = -1

        if DEBUG:
            print("Wrong indexes are those with +1 value:")
            print(weight_arr)
            print("Total mistakes:")
            print(np.sum(weight_arr == 1))

        return weight_arr

    def weight_update(self, weight_update_array, alpha):
        """
        Update the sample weights based on the weight update array and alpha.
        This method updates the sample weights according to the AdaBoost algorithm.
        """
        # Update the sample weights
        self.sample_weights *= np.exp(alpha * weight_update_array)
        # Normalize the sample weights
        self.sample_weights /= np.sum(self.sample_weights)
        if DEBUG:
            print("Updated sample weights:")
            print(self.sample_weights)
            print("\n")

    def crop_negatives(self, stage_idx):
        """
        Crop the negative samples based on this stage prediction.
        """

        # Get this stage predictions
        predictions = self.get_predictions(stage_idx=stage_idx)

        # Find the samples that are classified as positive (1) by the majority vote
        positive_samples = predictions == 1

        # Remove the samples, weights and labels
        # that are classified as negative by the majority vote
        self.feature_eval_matrix = self.feature_eval_matrix[:, positive_samples]
        self.sample_weights = self.sample_weights[positive_samples]
        self.sample_labels = self.sample_labels[positive_samples]
        self.sorted_indices = np.argsort(self.feature_eval_matrix, axis=1)

        if DEBUG:
            print(f"Cropping samples and weights at stage {stage_idx}...")
            print(f"Samples and weights cropped at stage {stage_idx}.")
            print("Remaining samples shape:", self.feature_eval_matrix.shape)
            print("Remaining sample weights:", self.sample_weights)

    def train(self):
        """
        Train the AdaBoost classifier.
        This method is currently a placeholder and does not implement the full training process.
        """

        for stage_i in range(self.n_stages):
            stage_classifier = []  # Reset this stage's classifier

            for x in range(2 + 2 * stage_i):
                print("Iteration:", x + 1)
                # Find the best feature based on the feature evaluation matrix,
                # sample weights, and labels
                best_idx, best_threshold, best_direction, alpha = (
                    self.find_best_feature()
                )

                # Get the weight update array based on the best feature found
                weight_update_array = self.find_weight_update_array(
                    feature_idx=best_idx,
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
                        "feature_idx": best_idx,
                        "threshold": best_threshold,
                        "direction": best_direction,
                        "alpha": alpha,
                    }
                )

            # Append this stage to the full classifier's list of stages
            self.trained_classifier.append(stage_classifier)

            # Remove the samples and weights that are classified as negative by the majority vote
            self.crop_negatives(stage_idx=stage_i)

        # Save the trained classifier to a file
        save_pickle_obj(
            filename="_pickle_folder/trained_classifier.pkl",
            obj=self.trained_classifier,
        )


## Test

if False:  # Set to True to run the test
    # Set the seed for reproducibility
    np.random.seed(42)

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
        size_x=5000, size_y=10000, bias_strenght=15
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
        n_stages=5,
    )

    my_trainer.train()

    print("\nTraining completed.\n")

    print(f"\nTotal training time: {(time.time() - start_time)} seconds")
