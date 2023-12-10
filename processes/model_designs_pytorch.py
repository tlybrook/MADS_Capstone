"""
Functions for designing and fitting the model.
"""
import torch
import torch.nn as nn
from sklearn.metrics import recall_score

class BaseCNN(nn.Module):
    """
    Base CNN containing only a single convolution layer
    Then a single pooling layer
    Then two dense layers followed by dropout and a final softmax.

    LR can be adjusted outside of this class to help tune the model.
    """
    def __init__(self):
        super(BaseCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(200 * 150 * 32, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(512, 4)  # Output layer for 4 classes

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.pool1(x)

        x = x.view(-1, 200 * 150 * 32)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
class ExapndedCNN(nn.Module):
    """
    Expanded CNN containing two concolutional layers convolution layers
    Then a single pooling layer
    Then two dense layers followed by dropout and a final softmax.

    LR can be adjusted outside of this class to help tune the model.
    """
    def __init__(self):
        super(ExapndedCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(200 * 150 * 32, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(512, 4)  # Output layer for 4 classes

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = self.pool1(x)

        x = x.view(-1, 200 * 150 * 32)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
class ExapndedCNN2(nn.Module):
    """
    Expanded CNN containing a single convolution layer followed by a single pooling layer.
    Then it has another single convolutional layer followed by another max pooling layer.
    Then two dense layers followed by dropout and a final softmax.

    LR can be adjusted outside of this class to help tune the model.
    """
    def __init__(self):
        super(ExapndedCNN2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(200 * 150 * 32, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(512, 4)  # Output layer for 4 classes

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.pool1(x)

        x = nn.functional.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.view(-1, 200 * 150 * 32)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
class BatchNormCNN(nn.Module):
    """
    Expanded CNN with batch norm included on each conv layer
    Same model structure as ExapndedCNN2()

    LR can be adjusted outside of this class to help tune the model.
    """
    def __init__(self):
        super(BatchNormCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(32) 
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(32) 
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(200 * 150 * 32, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(512, 4)

    def forward(self, x):
        x = nn.functional.relu(self.batch_norm1(self.conv1(x)))
        x = nn.functional.relu(self.batch_norm2(self.conv2(x)))

        x = self.pool1(x)
        x = x.view(-1, 200 * 150 * 32)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
class CNN(nn.Module):
    """
    Final CNN with batch norm included on each conv layer
    Same model structure as ExapndedCNN2()

    LR can be adjusted outside of this class to help tune the model.
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(100 * 75 * 32, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(512, 4)  # Output layer for 4 classes

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.pool1(x)

        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))

        x = self.pool2(x)
        x = x.view(-1, 100 * 75 * 32)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class FlaskCNN(nn.Module):
    """
    Expanded CNN containing a single convolution layer followed by a single pooling layer.
    Then it has another single convolutional layer followed by another max pooling layer.
    Then two dense layers followed by dropout and a final softmax.

    LR can be adjusted outside of this class to help tune the model.
    """
    def __init__(self):
        super(FlaskCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(100 * 75 * 2, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, 4)  # Output layer for 4 classes

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = self.pool1(x)

        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))
        x = nn.functional.relu(self.conv5(x))
        x = nn.functional.relu(self.conv6(x))
        x = nn.functional.relu(self.conv7(x))
        x = self.pool2(x)

        x = x.view(-1, 100 * 75 * 2)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class VGG16CNN(nn.Module):
    def __init__(self):
        super(VGG16CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv8 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv11 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(12 * 9 * 128, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(512, 4)  # Output layer for 4 classes

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = self.pool1(x)

        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))
        x = self.pool2(x)

        x = nn.functional.relu(self.conv5(x))
        x = nn.functional.relu(self.conv6(x))
        x = nn.functional.relu(self.conv7(x))
        x = self.pool3(x)

        x = nn.functional.relu(self.conv8(x))
        x = nn.functional.relu(self.conv9(x))
        x = nn.functional.relu(self.conv10(x))
        x = self.pool4(x)

        x = nn.functional.relu(self.conv11(x))
        x = nn.functional.relu(self.conv12(x))
        x = nn.functional.relu(self.conv13(x))
        x = self.pool5(x)

        x = x.view(-1, 12 * 9 * 128)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def model_loop(model, train_loader, val_loader, device, optimizer, criterion, num_epochs, patience):
    """
    Funciton to run the model loop.

    Parameters
    -----------
    model: PyTorch cnn using model_processes.CNN()
    train_loader: train data (batched PyTorch dataset)
    val_loader: validation data (batched PyTorch dataset)
    device: PyTorch device
    optimizer: model optimizer (i.e. optim.Adam(model.parameters(), lr=learning_rate))
    criterion: model criteria (i.e. nn.CrossEntropyLoss(weight=class_weight))
    num_epochs: number of epochs to train for if early stopping doesn't trigger
    patience: number of epochs without an improvement to validation loss to trigger early stopping. Useful for preventing overfitting
    """
    best_val_loss = float('inf')
    counter = 0  # Counter to track epochs since the last improvement
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
        epoch_train_recall = recall_score(epoch_y_true, epoch_y_preds, average="macro")
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
        epoch_val_recall = recall_score(epoch_y_true, epoch_y_preds, average="macro")
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
    results = {
        'train_loss': train_loss,
        'val_loss': val_loss_list,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'train_recall': train_recall,
        'val_recall': val_recall
    }

    return model, results

def pytorch_cnn_predict(model, data_loader, device, criterion):
    """Gets predictions from PyTorch model and accuracy score.
    Returns tuples (predictions, accuracy).

    Parameters
    ----------
    model: PyTorch model that has already been trained
    data_loader: PyTorch batched dataset to predict on
    device: PyTorch device 
    criterion: model criteria (i.e. nn.CrossEntropyLoss(weight=class_weight))
    """
    correct_val = 0
    total_val = 0
    val_loss_num = 0.0
    all_predictions = []
    all_y_true = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss_num += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_y_true.extend(labels.cpu().numpy())
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    accuracy = 100 * correct_val / total_val

    return all_predictions, all_y_true, accuracy

