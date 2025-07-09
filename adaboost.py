"""
Optimized AdaBoost numpy implementation. Now with numba.
Author: Alessandro Balzan
Date: 2025-07-04
Version: 2.0.0
"""

import os
import pickle
import numpy as np

# Debug flag - setup from launch configuration or environment variable
DEBUG_MODE = os.environ.get("ADABOOST_DEBUG", "False").lower() in ("true", "1", "yes")

# Conditional imports and setup
if DEBUG_MODE:
    print("🐛 DEBUG MODE - Numba disabled")

    # Define dummy decorators that do nothing
    def njit(*args, **kwargs):  # pylint: disable=unused-argument
        """
        Dummy njit decorator for debugging purposes.
        """

        def decorator(func):
            """
            Dummy decorator function that returns the function as is.
            """
            return func

        if len(args) == 1 and callable(args[0]):
            # Called as @njit without parentheses
            return args[0]
        # Called as @njit() or @njit(parallel=True)
        return decorator

    def prange(n):
        """
        Dummy prange function for debugging purposes.
        """
        return range(n)

else:
    print("🚀 PRODUCTION MODE - Numba enabled")
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
    print(f"💾 Classifier saved to {os.path.join(cwd, folder, filename)}")


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

        print(f"📂 Classifier loaded from {filename}")
        return obj

    except FileNotFoundError as exc:
        raise FileNotFoundError(f"File '{filename}' not found.") from exc


## Random generation methods
@njit
def random_int_matrix(low, high, shape):
    """
    Generate a random integer matrix with specified shape and range.

    Args:
        low (int): Lower bound for the random integers.
        high (int): Upper bound for the random integers.
        shape (tuple): Shape of the matrix to be generated.
    """
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

    # Initialize the random feature evaluation matrix
    _feature_eval_matrix = random_int_matrix(-100, 100, (size_x, size_y)).astype(
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


## Init methods
@njit
def sort_feature_matrix_numba(feature_eval_matrix):
    """
    Perform row-wise argsort on the feature evaluation matrix using Numba.

    Args:
        feature_eval_matrix (np.ndarray): 2D array of feature evaluations (shape: [n_features, n_samples]).

    Returns:
        np.ndarray: 2D array of sorted indices for each row (shape: [n_features, n_samples]).
    """
    n_features, n_samples = feature_eval_matrix.shape
    sorted_indices = np.empty((n_features, n_samples), dtype=np.uint32)

    for i in range(n_features):
        # Perform argsort on each row and cast to uint32
        sorted_indices[i] = np.argsort(feature_eval_matrix[i]).astype(np.uint32)

    return sorted_indices


## Training methods
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
    n_samples = feature_eval_matrix.shape[1]  # pylint: disable=unused-variable

    best_thresholds = np.zeros(n_features, dtype=feature_eval_matrix.dtype)
    best_errors = np.full(n_features, np.inf, dtype=np.float32)
    best_directions = np.zeros(n_features, dtype=np.int8)

    for i in prange(n_features):  # pylint: disable=not-an-iterable
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
def weight_update_numba(sample_weights, weight_update_array, alpha, aggressivness=1.0):
    """
    Update and normalize sample weights according to the AdaBoost algorithm (Numba-optimized).

    Args:
        sample_weights (numpy.ndarray): Array of sample weights to be updated (in-place).
        weight_update_array (numpy.ndarray): +1 for incorrect samples; -1 for correct ones.
        alpha (float): Amount of say.
        aggressivness (float, optional): Aggressiveness of the update. Defaults to 1.0.

    Returns:
        None: The sample_weights array is updated in-place.
    """
    n_samples = sample_weights.shape[0]

    # Update sample weights
    for i in range(n_samples):
        sample_weights[i] *= np.exp(aggressivness * alpha * weight_update_array[i])

    # Normalize sample weights
    total_weight = np.sum(sample_weights)
    for i in range(n_samples):
        sample_weights[i] /= total_weight


def unpack_stage(feature_eval_matrix_dtype, stage):
    """
    Unpack the trained classifier stage into Numba-compatible arrays.

    Args:
        feature_eval_matrix_dtype: (dtype): dtype of Matrix of feature evaluations
        stage (list): A list of dictionaries representing the stage.

    Returns:
        tuple: Numba-compatible arrays for feature indices, thresholds, directions, and alphas.
    """
    feature_idxs = np.array([x["feature_idx"] for x in stage], dtype=np.int32)
    thresholds = np.array(
        [x["threshold"] for x in stage], dtype=feature_eval_matrix_dtype
    )
    directions = np.array([x["direction"] for x in stage], dtype=np.int8)
    alphas = np.array([x["alpha"] for x in stage], dtype=np.float32)

    return feature_idxs, thresholds, directions, alphas


@njit
def majority_vote_numba(
    feature_eval_matrix, feature_idxs, thresholds, directions, alphas, sample_idx
):
    """
    Perform majority voting for the specified sample index using Numba.

    Args:
        feature_eval_matrix (np.ndarray): Matrix of feature evaluations.
        feature_idxs (np.ndarray): Array of feature indices for the stage.
        thresholds (np.ndarray): Array of thresholds for each feature.
        directions (np.ndarray): Array of decision directions (1 for ">", 0 for "<=").
        alphas (np.ndarray): Array of alpha values (weights) for each feature.
        sample_idx (int): Index of the sample to evaluate.

    Returns:
        int: Predicted label for the sample based on the majority vote.
    """
    n_features = feature_idxs.shape[0]
    total_vote = 0.0

    for i in range(n_features):
        # Get the feature index and the evaluation for the sample
        feature_idx = feature_idxs[i]
        sample_eval = feature_eval_matrix[feature_idx, sample_idx]

        # Apply the threshold comparison based on direction
        if directions[i] == 1:  # Direction: ">"
            prediction = 1 if sample_eval > thresholds[i] else -1
        else:  # Direction: "<="
            prediction = 1 if sample_eval <= thresholds[i] else -1

        # Compute the weighted vote
        weighted_vote = prediction * alphas[i]
        total_vote += weighted_vote

    # Determine the predicted label based on the sign of the total vote
    predicted_label = 1 if total_vote > 0 else -1

    return predicted_label


@njit
def get_predictions_numba(
    feature_eval_matrix, feature_idxs, thresholds, directions, alphas
):
    """
    Compute predictions for all samples using majority voting.

    Args:
        feature_eval_matrix (np.ndarray): Matrix of feature evaluations.
        feature_idxs (np.ndarray): Array of feature indices for the stage.
        thresholds (np.ndarray): Array of thresholds for each feature.
        directions (np.ndarray): Array of decision directions (1 for ">", 0 for "<=").
        alphas (np.ndarray): Array of alpha values (weights) for each feature.

    Returns:
        np.ndarray: Array of predicted labels for all samples.
    """
    n_samples = feature_eval_matrix.shape[1]
    predictions = np.empty(n_samples, dtype=np.int8)

    for sample_idx in range(n_samples):
        predictions[sample_idx] = majority_vote_numba(
            feature_eval_matrix,
            feature_idxs,
            thresholds,
            directions,
            alphas,
            sample_idx,
        )

    return predictions


@njit
def crop_negatives_numba(
    feature_eval_matrix, sample_weights, sample_labels, predictions
):
    """
    Update the feature evaluation matrix, sample weights, and labels
    to only include samples classified as positive. Sort the feature evaluation
    matrix again after cropping.

    Args:
        feature_eval_matrix (np.ndarray): Matrix of feature evaluations (shape: [n_features, n_samples]).
        sample_weights (np.ndarray): Array of sample weights (shape: [n_samples]).
        sample_labels (np.ndarray): Array of sample labels (shape: [n_samples]).
        predictions (np.ndarray): Array of predictions for the current stage (shape: [n_samples]).

    Returns:
        tuple: Updated feature_eval_matrix, sample_weights, sample_labels, and sorted_indices.
    """
    n_features, n_samples = feature_eval_matrix.shape

    # Count the number of positive samples
    positive_count = 0
    for i in range(n_samples):
        if predictions[i] == 1:
            positive_count += 1

    # Allocate new arrays for positive samples
    new_feature_eval_matrix = np.empty(
        (n_features, positive_count), dtype=feature_eval_matrix.dtype
    )
    new_sample_weights = np.empty(positive_count, dtype=sample_weights.dtype)
    new_sample_labels = np.empty(positive_count, dtype=sample_labels.dtype)

    # Copy positive samples to the new arrays
    positive_index = 0
    for i in range(n_samples):
        if predictions[i] == 1:
            for j in range(n_features):
                new_feature_eval_matrix[j, positive_index] = feature_eval_matrix[j, i]
            new_sample_weights[positive_index] = sample_weights[i]
            new_sample_labels[positive_index] = sample_labels[i]
            positive_index += 1

    # Sort the updated feature evaluation matrix row-wise
    sorted_indices = sort_feature_matrix_numba(
        feature_eval_matrix=new_feature_eval_matrix
    )

    return (
        new_feature_eval_matrix,
        new_sample_weights,
        new_sample_labels,
        sorted_indices,
    )


@njit
def get_statistics_numba(predictions, sample_labels):
    """
    Return statistics from the given predictions (Numba-compatible).

    Args:
        predictions (numpy.ndarray): Array of predictions for the current stage.
        sample_labels (numpy.ndarray): Array of true labels for the samples.

    Returns:
        tuple: (correct_predictions, true_positives, true_negatives)
            - correct_predictions (float): Percentage of correct predictions.
            - true_positives (float): Percentage of true positives.
            - true_negatives (float): Percentage of true negatives.
    """
    # Correct predictions percentage
    correct_predictions = (
        np.sum(predictions == sample_labels) / sample_labels.shape[0]
    ) * 100.0

    # True positives percentage
    positive_mask = sample_labels == 1
    true_positives = (
        (np.sum(predictions[positive_mask] == 1) / np.sum(positive_mask)) * 100.0
        if np.sum(positive_mask) > 0
        else 0.0
    )

    # True negatives percentage
    negative_mask = sample_labels == -1
    true_negatives = (
        (np.sum(predictions[negative_mask] == -1) / np.sum(negative_mask)) * 100.0
        if np.sum(negative_mask) > 0
        else 0.0
    )

    return correct_predictions, true_positives, true_negatives


def print_statistics(stage, corr_pred: float, true_pos: float, true_neg: float):
    """
    Print statistics for this stage

    Args:
        stage: Stage index
        corr_pred (float): Correct predictions
        true_pos (float): True positives
        true_neg (float): True negatives
    """

    print(f"\n📊 Statistics for stage {stage}:\n")

    print(
        f"⚖️ Percentage of correct predictions at stage {stage}:",
        corr_pred,
        "%",
    )
    print(
        f"📈 True positive percentage at stage {stage}:",
        true_pos,
        "%",
    )

    print(
        f"📈 True negative percentage at stage {stage}:",
        true_neg,
        "%\n",
    )


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
        **kwargs,
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
        self.aggressivness = kwargs.get("aggressivness", 1.0)

        # Allocate memory for the feature evaluation matrix, sample weights and labels
        print("🔄 Allocating memory for the AdaBoost classifier...")
        self.feature_eval_matrix = feature_eval_matrix
        self.sample_weights = np.array(sample_weights)
        self.sample_labels = np.array(sample_labels)
        print("✅ Done allocating memory for the AdaBoost classifier.\n")

        # Precomputed sorted indices for each feature evaluation
        print("🔄 Precomputing sorted indices for feature evaluations...")
        self.sorted_indices = sort_feature_matrix_numba(
            feature_eval_matrix=self.feature_eval_matrix
        )
        print("✅ Done precomputing sorted indices for feature evaluations.\n")

        # Placeholder for the trained classifier
        self.trained_classifier = []

    def train(self):
        """
        Train the AdaBoost classifier.
        """

        for stage_i in range(self.n_stages):
            print(f"🔄 Training stage {stage_i + 1} of {self.n_stages}...\n")

            stage_classifier = []  # Reset this stage's classifier

            for x in range(2 + 2 * stage_i):
                print(
                    f"🔍 Finding best feature for stage {stage_i + 1}, iteration {x + 1}..."
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
                    aggressivness=self.aggressivness,
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
                    f"📏 Feature index: {best_feature_idx}, Threshold: {best_threshold}, "
                    f"Direction: {'>' if best_direction == 1 else '<='}, Alpha: {alpha:.4f}, "
                    f"Error: {best_error}\n"
                )

            # Append this stage to the full classifier's list of stages
            self.trained_classifier.append(stage_classifier)

            # Preprocess the stage classifier for Numba compatibility
            (
                feature_idxs,
                thresholds,
                directions,
                alphas,
            ) = unpack_stage(self.feature_eval_matrix.dtype, stage_classifier)

            # Get this stage predictions
            predictions = get_predictions_numba(
                feature_eval_matrix=self.feature_eval_matrix,
                feature_idxs=feature_idxs,
                thresholds=thresholds,
                directions=directions,
                alphas=alphas,
            )

            # Get statistics for this stage
            corr, tp, tn = get_statistics_numba(
                predictions=predictions,
                sample_labels=self.sample_labels,
            )

            # Print statistics
            print_statistics(stage=stage_i, corr_pred=corr, true_pos=tp, true_neg=tn)

            # If the correct predictions are 100% for this stage, then stop early
            if corr > 99.9:
                print(f"🎉 Perfect stage {stage_i + 1}. Stopping here.\n")
                break

            # Remove the samples and weights that are classified as negative by the majority vote
            print(
                "🔄 Cropping negatives from the feature evaluation matrix and sorting again..."
            )
            (
                self.feature_eval_matrix,
                self.sample_weights,
                self.sample_labels,
                self.sorted_indices,
            ) = crop_negatives_numba(
                feature_eval_matrix=self.feature_eval_matrix,
                sample_weights=self.sample_weights,
                sample_labels=self.sample_labels,
                predictions=predictions,
            )
            print("✅ Cropped negatives from the feature evaluation matrix.\n")

        # Save the trained classifier to a file
        save_pickle_obj(
            filename="trained_classifier.pkl",
            obj=self.trained_classifier,
        )


class ClassifierScoreCheck:
    """
    Classifier for checking the scores of the trained AdaBoost classifier.
    """

    def __init__(self, feature_eval_matrix: np.array, sample_labels=np.array):
        """
        Initialize the Classifier with the feature evaluation matrix.

        Parameters:
            feature_eval_matrix (numpy.ndarray): Matrix of feature evaluations.
            sample_labels (numpy.ndarray, optional): Labels for the samples.
        """
        self.feature_eval_matrix = feature_eval_matrix
        self.sample_labels = sample_labels
        self.trained_classifier = load_pickle_obj(
            filename="_pickle_folder/trained_classifier.pkl"
        )

    def overall_predict(self):
        """
        Make predictions on the feature evaluation matrix using the trained classifier.
        Once a feature is clasifed as negative, it is 'discarded', so it cannot become
        positive anymore.

        Returns:
            numpy.ndarray: Array of predicted labels for all samples.
        """
        # Iterate through each stage of the trained classifier
        predictions = np.ones(self.feature_eval_matrix.shape[1], dtype=np.int8)

        for stage in self.trained_classifier:
            # Unpack the stage
            (
                feature_idxs,
                thresholds,
                directions,
                alphas,
            ) = unpack_stage(self.feature_eval_matrix.dtype, stage)

            # Get predictions for this stage
            stage_predictions = get_predictions_numba(
                feature_eval_matrix=self.feature_eval_matrix,
                feature_idxs=feature_idxs,
                thresholds=thresholds,
                directions=directions,
                alphas=alphas,
            )

            # Update overall predictions
            negative_predictions = stage_predictions == -1
            predictions[negative_predictions] = -1

        return predictions

    def analyze(self):
        """
        Analyze the predictions made by the classifier.
        Prints statistics about the predictions.
        """
        if self.sample_labels is None:
            raise ValueError("Labels are not provided for analysis.")

        print("\n🔄 Analyzing predictions...")
        # Get overall predictions
        predictions = self.overall_predict()
        print("✅ Overall predictions computed.\n")

        # Get statistics
        correct_predictions, true_positives, true_negatives = get_statistics_numba(
            predictions=predictions, sample_labels=self.sample_labels
        )

        # Print statistics
        print_statistics(
            stage="TOTAL",
            corr_pred=correct_predictions,
            true_pos=true_positives,
            true_neg=true_negatives,
        )
