"""
Functions for preprocessing the data.
"""

import tensorflow as tf

def split_data(data_dir, batch_size = 32):
    """Reads data from directory and formats
    into tensorflow dataset.
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
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    normalized_ds = ds.map(lambda x, y: (normalization_layer(x), y))
    return normalized_ds

def data_augmentation(train_ds, rotation_val = 0.5, flip_orientation = None):
    flip_layer = tf.keras.layers.RandomFlip(flip_orientation, input_shape=(256, 256, 1), seed=42)
    rotation_layer = tf.keras.layers.RandomRotation(rotation_val, seed=42)
    aug_train_ds = train_ds.map(lambda x, y: (flip_layer(x), y))
    rotated_train_ds = aug_train_ds.map(lambda i, k: (rotation_layer(i), k))
    final_train_ds = rotated_train_ds.concatenate(rotated_train_ds)
    return final_train_ds
