"""
This script contains the code to build our final CNN model.
Final model is ~1GB if you chose to run this file and save it.
You can comment the command to save the model as well.
"""

# Package imports
import logging
import dill as pickle
import torch
import torchsummary
from sklearn.metrics import recall_score, confusion_matrix
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import ConcatDataset
from processes.utils import (
    get_model_tracker,
    get_key
)
from processes.preprocessing import (
    pytorch_split,
    get_data_loaders,
    get_class_weights
)
from processes.visualization import (
    convolution_heatmap,
    eval_curve
)
from processes.model_designs_pytorch import (
    CNN,
    model_loop,
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
train_dataset, val_dataset, test_dataset = pytorch_split(data_dir=data_dir, train_ratio=train_ratio, val_ratio=val_ratio)

#augment training 
aug_transform = transforms.Compose([
    transforms.Resize(image_size),  # Resize the images to a consistent size
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(30, 270)),
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=(0.5), std=(0.5))  # Normalize the images to have a mean and std between 0, 1
])


train_loader, val_loader, test_loader = get_data_loaders(
     train_dataset=train_dataset, 
     val_dataset=val_dataset,
     test_dataset=test_dataset,
     aug_transform=aug_transform,
     general_transform=general_transform,
     batch_size=batch_size
)

# Get class weights to deal with imbalanced classes
class_weight = get_class_weights(data_loader=train_loader)

# Get model Tracker
model_tracker = get_model_tracker(file='pytorch_model_tracker.pickle', folder_path=None)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
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

cm_test = confusion_matrix(test_y_true, test_predictions)
test_recall = recall_score(test_y_true, test_predictions, average='weighted')
print(test_recall)

# Get predictions on val set
val_predictions, val_y_true, val_acc = pytorch_cnn_predict(
     model=model,
     data_loader=val_loader,
     device=device,
     criterion=criterion
)

cm_val = confusion_matrix(val_y_true, val_predictions)
val_recall = recall_score(val_y_true, val_predictions, average='weighted')
print(val_recall)

# Get predictions on train set
train_predictions, train_y_true, train_acc = pytorch_cnn_predict(
     model=model,
     data_loader=train_loader,
     device=device,
     criterion=criterion
)

cm_train = confusion_matrix(train_y_true, train_predictions)
train_recall = recall_score(train_y_true, train_predictions, average='weighted')
print(train_recall)

# add the results to the model_tracker
key = get_key(model_tracker=model_tracker)
model_tracker[key] = {}
model_tracker[key]['epoch_acc_table'] = results
model_tracker[key]['SEED'] = SEED
model_tracker[key]['train_cm'] = cm_train
model_tracker[key]['val_cm'] = cm_val
model_tracker[key]['test_cm'] = cm_test

# Save the .pickle model tracker file
with open('pytorch_model_tracker.pickle', 'wb') as handle:
    pickle.dump(model_tracker, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Get conv heatmap
convolution_heatmap(
    model, 
    general_transform, 
    device, 
    image_path='./final_dataset/adenocarcinoma/file13.jpg', 
    save_name='feature_maps_final_model'
)

# Reminder - The weighted Recall is the same as the accuracy for multiclass problems so 
# we use the acc values but label it is as Recall.
eval_curve('Recall', results['train_acc'], results['val_acc'], 'final_weighted_recall')

