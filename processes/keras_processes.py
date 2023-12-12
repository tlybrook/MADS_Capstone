"""Moving all keras funcitonality here.
There are potential issues installing tensorflow across Mac/Windows.
"""

import tensorflow as tf
from keras.models import Sequential
import logging

logger = logging.getLogger(__name__)


def split_data(data_dir, batch_size = 32):
    """Reads data from directory and formats
    into tensorflow dataset.
    
    THIS IS USED FOR KERAS ONLY!
    """
    image_resize = (256, 256)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.3,
        labels="inferred",
        label_mode='categorical',
        color_mode="grayscale",
        subset="training",
        seed=42,
        batch_size=batch_size,
        image_size=image_resize)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.3,
        labels="inferred",
        label_mode='categorical',
        color_mode="grayscale",
        subset="validation",
        shuffle=True,
        seed=42,
        batch_size=batch_size,
        image_size=image_resize)

    subset_size = int(0.33 * len(val_ds))
    test_ds = val_ds.take(subset_size)
    val_ds = val_ds.skip(subset_size)
    
    return train_ds, val_ds, test_ds

def normalize_data(ds):
    """Normalizes Keras data.

    THIS IS USED FOR KERAS ONLY!
    """
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    normalized_ds = ds.map(lambda x, y: (normalization_layer(x), y))
    return normalized_ds

def data_augmentation(train_ds, rotation_val = 0.5, flip_orientation = None):
    """Perform random flips and rotations on tensorflow dataset
    
    THIS IS USED FOR KERAS ONLY!
    """
    flip_layer = tf.keras.layers.RandomFlip(flip_orientation, input_shape=(256, 256, 1), seed=42)
    rotation_layer = tf.keras.layers.RandomRotation(rotation_val, seed=42)
    aug_train_ds = train_ds.map(lambda x, y: (flip_layer(x), y))
    rotated_train_ds = aug_train_ds.map(lambda i, k: (rotation_layer(i), k))
    final_train_ds = rotated_train_ds.concatenate(rotated_train_ds)
    return final_train_ds

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

