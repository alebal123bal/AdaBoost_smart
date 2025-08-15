"""
Classifier for checking the scores of the trained AdaBoost classifier.
"""

import numpy as np
from utils.io_operations import PickleUtils
from utils.statistics import Statistics
from adaboost import unpack_stage, get_predictions_numba


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
        self.trained_classifier = PickleUtils.load_pickle_obj(
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
        correct_predictions, true_positives, true_negatives = Statistics.get_statistics(
            predictions=predictions, sample_labels=self.sample_labels
        )

        # Print statistics
        Statistics.print_statistics(
            stage_index=777,
            corr_pred=correct_predictions,
            true_pos=true_positives,
            true_neg=true_negatives,
        )
