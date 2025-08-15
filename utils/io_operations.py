"""
Utility functions for saving and loading objects using pickle.
"""

import os
import pickle


class PickleUtils:
    @staticmethod
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

    @staticmethod
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
