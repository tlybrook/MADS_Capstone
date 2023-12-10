"""
Utility/Helper functions for model building.
"""

import os
import logging
import dill as pickle
from keras.models import Sequential

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


def get_model_summary(model: Sequential):
    """Gets the model summary as a string which will be properly formatted when printing.

    ** ONLY WORKS IN KERAS AND NOT IN PYTORCH!
    """
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    logger.info(f"Model Summary: \n{short_model_summary}")
    print(short_model_summary)

    return short_model_summary


def get_steps_per_epoch(df, batch_size):
    """Calculates the steps per epoch.

    ** ONLY WORKS IN KERAS AND NOT IN PYTORCH!
    """
    total_ct = 0 
    for image_batch, y_batch in df:
        total_ct += len(y_batch)
    
    return int(total_ct / batch_size)


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

