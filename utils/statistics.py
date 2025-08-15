"""
Statistics utility functions.
"""

import numpy as np


class Statistics:
    @staticmethod
    def get_statistics(predictions, sample_labels):
        """
        Return statistics from the given predictions.

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

    @staticmethod
    def print_statistics(
        stage_index: int, corr_pred: float, true_pos: float, true_neg: float
    ):
        """
        Print statistics for this stage

        Args:
            stage_index: Stage index
            corr_pred (float): Correct predictions
            true_pos (float): True positives
            true_neg (float): True negatives
        """

        print(f"\n📊 Statistics for stage {stage_index}:\n")

        print(
            f"⚖️ Percentage of correct predictions at stage {stage_index}:",
            corr_pred,
            "%",
        )
        print(
            f"📈 True positive percentage at stage {stage_index}:",
            true_pos,
            "%",
        )

        print(
            f"📈 True negative percentage at stage {stage_index}:",
            true_neg,
            "%\n",
        )
