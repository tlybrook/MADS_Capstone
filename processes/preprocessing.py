"""
Functions for preprocessing the data.
"""

# Package imports
import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from torchvision import datasets, transforms
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data import DataLoader, random_split, TensorDataset



def pytorch_split(data_dir, train_ratio, val_ratio):
    """Split a pytorch dataset from data dir.
    test_ratio is not given as an argument since it well be inferred from 
    the train_ratio and val_ratio.

    i.e. if train_ratio = 0.7 and val_ratio = 0.2 then test_ratio will be 0.1

    (1 - (0.7 + 0.3)) = 0.1

    ** ONLY FOR PYTORCH, WONT WORK IN KERAS!

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
    """Gets the batched data loaders from the PyTorch datasets.

    ** ONLY FOR PYTORCH, WONT WORK IN KERAS!

    Parameters
    -----------
    train_dataset: the train PyTorch dataset
    val_dataset: the validation PyTorch dataset
    test_dataset: the test PyTorch dataset
    aug_transform: (PyTorch torchvision transformation) a transformation containing the general transformations
                    needed as well as any data augmentation. This is used for the training dataset.
    general_transform: (PyTorch torchvision transformation) a general transformation (image resize, grayscale, normalization)
                       of the image this is used for the test and validation datasets.
    batch_size: (int) the size of each batch
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

    ** ONLY FOR PYTORCH, WONT WORK IN KERAS!

    Parameters
    -----------
    data_loader: The PyTorch dataloader for the training dataset.
    """
    y = []
    for image_batch, label_batch in data_loader:
        for i in label_batch:
            y.append(int(i))

    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)

    class_weight = torch.tensor(class_weights, dtype=torch.float32)
    
    return class_weight

