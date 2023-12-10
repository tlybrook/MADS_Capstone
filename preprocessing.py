"""
Functions for preprocessing the data.
"""

# Package imports
import torch
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from torchvision import datasets, transforms
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data import DataLoader, random_split, TensorDataset


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



def pytorch_split(data_dir, train_ratio, val_ratio):
    """Split a pytorch dataset from data dir.
    test_ratio is not given as an argument since it well be inferred from 
    the train_ratio and val_ratio.

    i.e. if train_ratio = 0.7 and val_ratio = 0.2 then test_ratio will be 0.1

    (1 - (0.7 + 0.3)) = 0.1

    Parameters
    -----------
    data_dir: location of the directory containing all the images
    train_ratio: [0 to 1] ratio of images to use in training set
    val_ratio: [0 to 1] ratio of images to use in validation set. 
    """
    full_dataset = datasets.ImageFolder(root=data_dir, transform=None)

    # Split the dataset into train, test, validation
    num_data = len(full_dataset)
    num_train = int(train_ratio * num_data)
    num_val = int(val_ratio * num_data)
    num_test = num_data - num_train - num_val

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [num_train, num_val, num_test])

    return train_dataset, val_dataset, test_dataset


def get_data_loaders(train_dataset, val_dataset, test_dataset, aug_transform, general_transform, batch_size):
    """Gets the batched data loaders from the pytorhc datasets.
    """
    augmented_images = []
    augmented_labels = []

    # Perform augmentation on the training dataset
    for img, label in train_dataset:
        augmented_img = aug_transform(img)

        original_img_tensor = transforms.ToTensor()(img)
        # Compare original and augmented images
        if not torch.equal(original_img_tensor, augmented_img):
                augmented_images.append(augmented_img)
                augmented_labels.append(label)

    # Create TensorDataset from augmented data
    final_augmented_dataset = TensorDataset(torch.stack(augmented_images), torch.tensor(augmented_labels))

    #apply the transformations to each split
    train_dataset.dataset.transform = general_transform
    val_dataset.dataset.transform = general_transform
    test_dataset.dataset.transform = general_transform

    # Assuming train_dataset contains your original train dataset
    imgs = []
    labels = []
    for data in train_dataset:
        imgs.append(data[0])
        labels.append(data[1])

    # Concatenate datasets
    final_train_dataset = ConcatDataset([
        final_augmented_dataset, 
        TensorDataset(torch.stack(imgs), torch.tensor(labels))
    ])

    # Create DataLoaders for train, validation, test
    train_loader = DataLoader(final_train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def get_class_weights(data_loader):
    """Gets the class weights on training data.
    """
    y = []
    for image_batch, label_batch in data_loader:
        for i in label_batch:
            y.append(int(i))

    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)

    class_weight = torch.tensor(class_weights, dtype=torch.float32)
    
    return class_weight

