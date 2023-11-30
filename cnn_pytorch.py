#%%
import logging
import torch
from PIL import Image
from sklearn.metrics import recall_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import ConcatDataset
from utils import (
    get_model_tracker,
    get_key
)

from logger_settings import *
logger = logging.getLogger(__name__)

#%%
data_dir = './final_dataset'  
image_size = (256, 256)

# Define transformations - general is for valid and test and train is for training set
general_transform = transforms.Compose([
    transforms.Resize(image_size),  # Resize the images to a consistent size
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=(0.5), std=(0.5))  # Normalize the images to have a mean and std between 0, 1
])

# create the dataset and complete preprocessing steps
full_dataset = datasets.ImageFolder(root=data_dir, transform=None)

#%%
# Split the dataset into train, test, validation
train_ratio = 0.7  # 80% training data
val_ratio = 0.2    # 10% validation data
test_ratio = 0.1   # 10% test data

num_data = len(full_dataset)
num_train = int(train_ratio * num_data)
num_val = int(val_ratio * num_data)
num_test = num_data - num_train - num_val

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [num_train, num_val, num_test])

#%%
#augment training 
aug_transform = transforms.Compose([
    transforms.Resize(image_size),  # Resize the images to a consistent size
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(30, 270)),
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=(0.5), std=(0.5))  # Normalize the images to have a mean and std between 0, 1
])


#%%
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

#%%
#apply the transformations to each split
train_dataset.dataset.transform = general_transform
val_dataset.dataset.transform = general_transform
test_dataset.dataset.transform = general_transform

#%%
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

#%%
# Create DataLoaders for train, validation, test
batch_size = 32

train_loader = DataLoader(final_train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

#%%
#getting the class weights after data augmentation
y = []
for image_batch, label_batch in train_loader:
    for i in label_batch:
        y.append(int(i))

class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)

class_weight = torch.tensor(class_weights, dtype=torch.float32)

#%% Get model Tracker
model_tracker = get_model_tracker(file='pytorch_model_tracker.pickle', folder_path=None)

# %%
num_epochs = 100
learning_rate = 0.0001

# CNN model definition
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 128 * 32, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(512, 4)  # Output layer for 4 classes

    def forward(self, x):
        x = self.pool1(nn.functional.relu(self.conv1(x)))
        # x = self.pool2(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 128 * 128 * 32)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Initialize model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
criterion = nn.CrossEntropyLoss(weight=class_weight)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Initialize variables for early stopping
best_val_loss = float('inf')
patience = 5  # Number of epochs to wait for improvement
counter = 0  # Counter to track epochs since the last improvement

#%%
train_acc = []
val_acc = []
train_loss = []
val_loss_list = []
train_recall = []
val_recall = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    epoch_y_preds = []
    epoch_y_true = []

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        epoch_y_preds.extend(predicted)
        epoch_y_true.extend(labels)

    # Calculate the recall
    epoch_train_recall = recall_score(epoch_y_true, epoch_y_preds, average="micro")
    train_recall.append(epoch_train_recall)

    # Calculate train accuracy
    train_accuracy = 100 * correct_train / total_train
    train_acc.append(train_accuracy)

    #Save the loss values
    train_loss.append(running_loss)

    # Validation accuracy and loss
    model.eval()
    correct_val = 0
    total_val = 0
    val_loss = 0.0
    epoch_y_preds = []
    epoch_y_true = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
            epoch_y_preds.extend(predicted)
            epoch_y_true.extend(labels)

    # Calculate the recall
    epoch_val_recall = recall_score(epoch_y_true, epoch_y_preds, average="micro")
    val_recall.append(epoch_val_recall)

    # Calculate validation accuracy
    val_accuracy = 100 * correct_val / total_val
    val_acc.append(val_accuracy)

    # Save the loss into a list
    val_loss_list.append(val_loss)

    # Calculate average validation loss
    avg_val_loss = val_loss / len(val_loader)

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0  # Reset counter
        # Save the model state using torch.save if desired
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break  # End training loop

    # Print metrics
    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Training Loss: {running_loss / len(train_loader):.4f}, '
          f'Train Accuracy: {train_accuracy:.2f}%, '
          f'Train Recall: {epoch_train_recall:.2f}%, '
          f'Validation Loss: {avg_val_loss:.4f}, '
          f'Validation Accuracy: {val_accuracy:.2f}%'
          f'Validation Recall: {epoch_val_recall:.2f}%, ')

print('Finished Training')

# %%
# Assuming new_data_loader contains your DataLoader instance for new data
model.eval()  # Set the model to evaluation mode
all_predictions = []
with torch.no_grad():
    for images, labels in test_loader:
        # Assuming 'images' is the input format expected by the model
        outputs = model(images)
        # Assuming it's a classification task, get the predicted class (index)
        _, predicted = torch.max(outputs, 1)
        # Collect predictions
        all_predictions.extend(predicted.cpu().numpy())
# 'all_predictions' contains the predicted class labels for the new data
print(all_predictions)


#%% Get model tracking info saved in the tracker
import pandas as pd
import dill as pickle 

key = get_key(model_output=model_tracker)
results = pd.DataFrame(data={'train_loss': train_loss, 'val_loss': val_loss, 'train_acc': train_acc, 'val_acc': val_acc, 'train_recall': train_recall, 'val_recall': val_recall})
model_tracker[key] = {}
model_tracker[key]['epoch_acc_table'] = results

# %%
correct_val = 0
total_val = 0
val_loss = 0.0
all_predictions = []
all_y_true = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_y_true.extend(labels.cpu().numpy())
        total_val += labels.size(0)
        correct_val += (predicted == labels).sum().item()

# Calculate validation accuracy
val_accuracy = 100 * correct_val / total_val

# %%
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(all_y_true, all_predictions)
print('Validation CM:')
print(cm)
model_tracker[key]['val_cm'] = cm

# %%
correct_val = 0
total_val = 0
val_loss = 0.0
all_predictions = []
all_y_true = []

with torch.no_grad():
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_y_true.extend(labels.cpu().numpy())
        total_val += labels.size(0)
        correct_val += (predicted == labels).sum().item()

# Calculate validation accuracy
val_accuracy = 100 * correct_val / total_val

cm = confusion_matrix(all_y_true, all_predictions)
print('Train CM:')
print(cm)
model_tracker[key]['train_cm'] = cm

# %%
correct_val = 0
total_val = 0
val_loss = 0.0
all_predictions = []
all_y_true = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_y_true.extend(labels.cpu().numpy())
        total_val += labels.size(0)
        correct_val += (predicted == labels).sum().item()

# Calculate validation accuracy
val_accuracy = 100 * correct_val / total_val

cm = confusion_matrix(all_y_true, all_predictions)
print('Test CM:')
print(cm)
model_tracker[key]['test_cm'] = cm

print(val_accuracy)

#%% Save the .pickle model tracker file
with open('pytorch_model_tracker.pickle', 'wb') as handle:
    pickle.dump(model_tracker, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%% Save the model
# torch.save(model, 'model_pytorch_13.h5')
# loaded_model = torch.load('model_pytorch_13.h5')

#%% We copied from this link, cite in paper
# https://ravivaishnav20.medium.com/visualizing-feature-maps-using-pytorch-12a48cd1e573
model_weights =[]
#we will save the 49 conv layers in this list
conv_layers = []
# get all the model children as list
model_children = list(model.children())
#counter to keep count of the conv layers
counter = 0
#append all the conv layers and their respective wights to the list
for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        counter+=1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            for child in model_children[i][j].children():
                if type(child) == nn.Conv2d:
                    counter+=1
                    model_weights.append(child.weight)
                    conv_layers.append(child)
print(f"Total convolution layers: {counter}")
print("conv_layers")

image = Image.open('./final_dataset/adenocarcinoma/file13.jpg')
image = general_transform(image)
image = image.to(device)

outputs = []
names = []
count = 0
for layer in conv_layers[0:]:
    image = layer(image)
    outputs.append(image)
    count += 1
    names.append(str(f"{layer} {count}"))
print(len(outputs))
#print feature_maps
for feature_map in outputs:
    print(feature_map.shape)

processed = []
for feature_map in outputs:
    feature_map = feature_map.squeeze(0)
    gray_scale = torch.sum(feature_map,0)
    gray_scale = gray_scale / feature_map.shape[0]
    processed.append(gray_scale.data.cpu().numpy())
for fm in processed:
    print(fm.shape)

fig = plt.figure(figsize=(30, 50))
for i in range(len(processed)):
    a = fig.add_subplot(5, 4, i+1)
    imgplot = plt.imshow(processed[i])
    a.axis("off")
    a.set_title(names[i].split('(')[0], fontsize=30)
# plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')


# %%
import matplotlib.pyplot as plt

# plt.plot(results.history['loss'], label='train loss')
# plt.plot(results.history['val_loss'], label='val loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.title('model Loss')
# plt.legend()
# plt.show()
# plt.savefig('Val_loss')

plt.plot(train_acc, label='train accuracy')
plt.plot(val_acc, label='val accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.title('model accuracy')
plt.legend()
plt.show()
plt.savefig('Val_acc')

# plt.plot(results.history['recall'], label='train recall')
# plt.plot(results.history['val_recall'], label='val recall')
# plt.ylabel('recall')
# plt.xlabel('epoch')
# plt.title('model recall')
# plt.legend()
# plt.show()
# plt.savefig('Val_recall')
