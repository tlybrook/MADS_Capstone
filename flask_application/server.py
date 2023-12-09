from flask import Flask, render_template, request
import os
import numpy as np
from torchvision import transforms
from PIL import Image
import dill as pickle
import torch
import torch.nn as nn
import uuid

app = Flask(__name__)
upload_folder = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = upload_folder
model_path = './static/model_pytorch_cnn_12_6.pickle'

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=8, out_channels=2, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(100 * 75 * 2, 256)
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
        x = self.dropout(x)
        x = self.fc3(x)
        return x

image_size = (300, 400)
general_transform = transforms.Compose([
    transforms.Resize(image_size),  # Resize the images to a consistent size
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=(0.5), std=(0.5))  # Normalize the images to have a mean and std between 0, 1
])

@app.route('/')
def load_page():
    return render_template('loading.html')

@app.route('/app', methods=['GET', 'POST'])
def upload_file():
    filename = None
    prediction = None
    prob_string = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            isExist = os.path.exists(upload_folder)
            if not isExist:
                os.makedirs(upload_folder)
            unique_filename = str(uuid.uuid4())

            filename = f"{app.config['UPLOAD_FOLDER']}/{unique_filename}".replace('\\', '/')
            file.save(filename)
            im = Image.open(filename)
            processed_image = general_transform(im)

            model_instance = CNN()
            with open(model_path, 'wb') as f:
                pickle.dump(model_instance, f)

            with torch.no_grad():
                outputs = model_instance(processed_image)
                _, predicted = torch.max(outputs.data, 1)

                class_dict = {0: 'adenocarcinoma', 1: 'large cell carcinoma', 2: 'normal', 3: 'squamous cell carcinoma'}
                prediction = class_dict[(predicted.cpu().numpy()[0])]
                prob = nn.functional.softmax(outputs, dim=1)
                prob_array = np.asarray(prob[0])
                prob_string = f"The probabilities for the classes are adenocarcinoma: {str(np.round(prob_array[0]*100, 1))}%, \
                                large cell carcinoma: {str(np.round(prob_array[1]*100, 1))}%, \
                                normal: {str(np.round(prob_array[2]*100, 1))}%, \
                                squamous cell carcinoma: {str(np.round(prob_array[3]*100, 1))}%"

    return render_template('index.html', filename=filename, prediction=prediction, probabilities=prob_string)

if __name__ == '__main__':
    app.run(debug=True, port=4000)