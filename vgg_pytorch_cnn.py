"""
This file runs a VGG16 model from scratch using custom pytorch design.
"""

# Package imports
import logging
import torch
import matplotlib.pyplot as plt
import torchsummary
from sklearn.metrics import recall_score, confusion_matrix
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import ConcatDataset
from processes.utils import (
    get_model_tracker,
    get_key
)
from processes.visualization import (
    convolution_heatmap,
    confusion_matrix_viz,
    eval_curve
)
from processes.model_designs_pytorch import (
     CNN,
     model_loop,
     VGG16CNN,
     pytorch_cnn_predict
)

# Get global logger
from logs.logger_settings import *
logger = logging.getLogger(__name__)

# Other macro level variables
SEED = 113
data_dir = './final_dataset'  
image_size = (400, 300)
torch.manual_seed(SEED)
train_ratio = 0.7  # 80% training data
val_ratio = 0.2    # 10% validation data
test_ratio = 0.1   # 10% test data
batch_size = 32
num_epochs = 100
learning_rate = 0.0001
patience = 5  # Number of epochs to wait for improvement

# Define transformations - general is for valid and test and aug is for training set
general_transform = transforms.Compose([
    transforms.Resize(image_size),  # Resize the images to a consistent size
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=(0.5), std=(0.5))  # Normalize the images to have a mean and std between 0, 1
])

# create the dataset and complete preprocessing steps
full_dataset = datasets.ImageFolder(root=data_dir, transform=None)

# Split the dataset into train, test, validation
num_data = len(full_dataset)
num_train = int(train_ratio * num_data)
num_val = int(val_ratio * num_data)
num_test = num_data - num_train - num_val

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [num_train, num_val, num_test])

#augment training 
aug_transform = transforms.Compose([
    transforms.Resize(image_size),  # Resize the images to a consistent size
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(30, 270)),
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=(0.5), std=(0.5))  # Normalize the images to have a mean and std between 0, 1
])

# Lists to store augmented data
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

#getting the class weights after data augmentation
y = []
for image_batch, label_batch in train_loader:
    for i in label_batch:
        y.append(int(i))

class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)

class_weight = torch.tensor(class_weights, dtype=torch.float32)

# Get model Tracker
model_tracker = get_model_tracker(file='pytorch_model_tracker.pickle', folder_path=None)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VGG16CNN().to(device)
criterion = nn.CrossEntropyLoss(weight=class_weight)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Fit the model
model, results = model_loop(
    model=model, 
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    optimizer=optimizer,
    criterion=criterion,
    num_epochs=num_epochs,
    patience=patience
)

# Get model summary
torchsummary.summary(model, input_size=(1, 400, 300), batch_size=batch_size)

# Get predictions on test set
test_predictions, test_y_true, test_acc = pytorch_cnn_predict(
     model=model,
     data_loader=test_loader,
     device=device,
     criterion=criterion
)

cm = confusion_matrix(test_y_true, test_predictions)
test_recall = recall_score(test_y_true, test_predictions, average='micro')
print(test_recall)

