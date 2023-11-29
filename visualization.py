'''
This file contains all visualization functions for our project.
'''
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch.nn as nn
from PIL import Image
import torch
import os

# Folder where all visualizations will be stored.
visualization_folder = './visualization_outputs'

# This function takes a metric like loss, accuracy, recall and creates a curve over each CNN 
# epoch for training and validation. The plot is then saved to the visualization folder.
# The save name is the name of the file. 
def eval_curve(metric, train_vals, valid_vals, save_name):
    plt.plot(train_vals, label=f"Train {metric}")
    plt.plot(valid_vals, label=f"Valid {metric}")
    plt.ylabel(f"{metric}")
    plt.xlabel(f"Epoch")
    plt.title(f"Model {metric}")
    plt.legend()

    isExist = os.path.exists(visualization_folder)
    if not isExist:
        os.makedirs(visualization_folder)

    plt.savfig(f"{visualization_folder}/{save_name}.png")
    return 

#Still need to add visualization for confusion matrix
#cm = confusion_matrix(all_y_true, all_predictions)

# This function creates a heatmap for each convolution layer of the CNN to visualize the features.
# The following code comes from https://ravivaishnav20.medium.com/visualizing-feature-maps-using-pytorch-12a48cd1e573
def convolution_heatmap(model, transform, device, image_path, save_name):
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

    image = Image.open(image_path)
    image = transform(image)
    image = image.to(device)

    outputs = []
    names = []
    for layer in conv_layers[0:]:
        image = layer(image)
        outputs.append(image)
        names.append(str(layer))

    processed = []
    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map,0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())

    fig = plt.figure(figsize=(30, 50))
    for i in range(len(processed)):
        a = fig.add_subplot(5, 4, i+1)
        imgplot = plt.imshow(processed[i])
        a.axis("off")
        a.set_title(names[i].split('(')[0], fontsize=30)

    isExist = os.path.exists(visualization_folder)
    if not isExist:
        os.makedirs(visualization_folder)

    plt.savefig(str(f"{visualization_folder}/{save_name}.jpg"), bbox_inches='tight')
    return

