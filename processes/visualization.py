'''
This file contains all visualization functions for our project.
'''
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import torch.nn as nn
from PIL import Image
import torch
import os

# Folder where all visualizations will be stored.
visualization_folder = './visualization_outputs'

def eval_curve(metric, train_vals, valid_vals, save_name):
    """Create a plot comparing training metrics vs recall metrics for each epoch of model training.

    Parameters
    ----------
    metric: (str) name of the metric. Will be used for labelling the chart
    train_vals: (list or np.array) values correspinding to the training metric
    valid_vals: (list or np.array) values correpsoding to the validation metrics.
    save_name: (str) name of the file where the chart will be saved. Do not include a suffix such as .png or .jpg. 
    """
    plt.plot(train_vals, label=f"Train {metric}")
    plt.plot(valid_vals, label=f"Valid {metric}")
    plt.ylabel(f"{metric}")
    plt.xlabel(f"Epoch")
    plt.title(f"Model {metric}")
    plt.legend()

    isExist = os.path.exists(visualization_folder)
    if not isExist:
        os.makedirs(visualization_folder)

    plt.savefig(f"{visualization_folder}/{save_name}.png")
    return 

def confusion_matrix_viz(y_true, y_pred, save_name, viz_title):
    """Generate a confusion matrix visualization for all 4 classes and save it to the output folder.

    Parameters
    ----------
    y_true: (list or np.array) list of the true labels.
    y_pred: (list or np.array) list of the predicted labels.
    save_name: (str) name of the file where the chart will be saved. Do not include a suffix such as .png or .jpg. 
    viz_title: (str) name of the visualiztion title.
    """
    cm = confusion_matrix(y_true, y_pred)
    label_key = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']
    df_cm = pd.DataFrame(cm, index = [i for i in label_key],
                         columns = [i for i in label_key])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.title(viz_title, fontsize=16)
    plt.xlabel('Predicted labels', fontsize=12)
    plt.ylabel('True labels', fontsize=12)

    isExist = os.path.exists(visualization_folder)
    if not isExist:
        os.makedirs(visualization_folder)

    plt.savefig(f"{visualization_folder}/{save_name}.png")
    return

def convolution_heatmap(model, transform, device, image_path, save_name):
    """Creates a feature map for each convolution layer of the CNN to visualize the features it is learning. 
    The following code is copied from the following source:
        Vaishnav, Ravi. “Visualizing Feature Maps Using PyTorch.” Medium, 28 June 2021, 
            ravivaishnav20.medium.com/visualizing-feature-maps-using-pytorch-12a48cd1e573. Accessed 29 Nov. 2023.  
    
    Parameters 
    ----------
    model: The trained PyTorch model.
    transform: (PyTorch torchvision transformation) a general transformation of the image that matches the CNN test/validation
                transformations.
    device: Specifies the hardware (CPU or GPU) where computations will be executed within the function. (Is created
            when creating the CNN in PyTorch)
    image_path: (str) the path to the image you want to generate the heatmap for.
    save_name: (str) name of the file where the chart will be saved. Do not include a suffix such as .png or .jpg. 
    """
    model_weights =[]
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
    count = 0
    for layer in conv_layers[0:]:
        image = layer(image)
        outputs.append(image)
        count += 1
        names.append(str(f"{layer} {count}"))

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
        a.set_title(names[i].split('(')[0], fontsize=14)

    isExist = os.path.exists(visualization_folder)
    if not isExist:
        os.makedirs(visualization_folder)

    plt.savefig(str(f"{visualization_folder}/{save_name}.png"), bbox_inches='tight')
    return

def image_dims_viz(image_dims, save_name):
    """Generates a scatterplot displaying all the image dimensions in the dataset.

    Parameters
    ----------
    image_dims: (pandas dataframe) a dataframe with the following columns: 'FileName', 'Width', 'Height'.
    save_name: (str) name of the file where the chart will be saved. Do not include a suffix such as .png or .jpg. 
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(image_dims.Width, image_dims.Height, color='blue', alpha=0.5, picker=True)
    plt.title("Image Dimensions", fontsize=18)
    plt.xlabel("Width", fontsize=12)
    plt.ylabel("Height", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    isExist = os.path.exists(visualization_folder)
    if not isExist:
        os.makedirs(visualization_folder)

    plt.savefig(str(f"{visualization_folder}/{save_name}.png"))
    return