#%%
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.utils.class_weight import compute_class_weight
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim

#%%
data_dir = './final_dataset'  

image_size = (256, 256)
# Define transformations for image data
transform = transforms.Compose([
    transforms.Resize(image_size),  # Resize the images to a consistent size
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalize the images to have a mean and std between 0, 1
])

# create the dataset and complete preprocessing steps
full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

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

# Create DataLoaders for train, validation, test
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

#%%
y = []
count = 0
for i in os.listdir(data_dir):
    folder_len = len(os.listdir(f"{data_dir}/{i}"))
    folder_vals = [count] * folder_len
    count += 1
    y.extend(folder_vals)

class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)

class_weight = torch.tensor(class_weights, dtype=torch.float32)

# %%
num_epochs = 100
learning_rate = 0.001

# CNN model definition
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 64 * 64, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 4)  # Output layer for 4 classes


        #self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        #self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        #self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool1(nn.functional.relu(self.conv1(x)))
        x = self.pool2(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 64 * 64)
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
patience = 10  # Number of epochs to wait for improvement
counter = 0  # Counter to track epochs since the last improvement

#%%
train_acc = []
val_acc = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

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

    # Calculate train accuracy
    train_accuracy = 100 * correct_train / total_train
    train_acc.append(train_accuracy)

    # Validation accuracy and loss
    model.eval()
    correct_val = 0
    total_val = 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    # Calculate validation accuracy
    val_accuracy = 100 * correct_val / total_val
    val_acc.append(val_accuracy)

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
          f'Validation Loss: {avg_val_loss:.4f}, '
          f'Validation Accuracy: {val_accuracy:.2f}%')

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

# %%
