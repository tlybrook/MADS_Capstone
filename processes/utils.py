"""
Utility/Helper functions for model building.
"""
import os
import logging
import dill as pickle

logger = logging.getLogger(__name__)

def get_model_tracker(file: str = 'model_tracker.pickle', folder_path = None):
    """Get the model tracker dictionary from pickle file.
    Create blank dictionary if the file does not exist.

    Parameters
    -----------
    file: name of the model tracker file. If it does not already exist a new file will be created.
    folder_path: path to the folder where the pickle file should be saved.
    """
    if file not in os.listdir(folder_path):
        logger.warning(f"file '{file}' not found! Creating blank dictionary for model tracker")
        tracker = {}
    else:
        with open(file, 'rb') as handle:
            tracker = pickle.load(handle)
        logger.debug(f"Model tracker file '{file}' was found! Loading into current process.")

    return tracker

def get_key(model_tracker: dict):
    """Helper function to update the keys of the model tracker
    file each time a new model is run.

    Parameters
    ----------
    model_tracker: model tracking file as a python dictionary
    """
    if len(model_tracker.keys()) > 0:
        key = max(list(model_tracker.keys())) + 1
    else:
        key = 0
    return key

