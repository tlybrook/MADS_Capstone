#%%
from PIL import Image
#import cv2
import os
import imagesize
import pandas as pd
import matplotlib.pyplot  as plt
import numpy as np
import tensorflow as tf

root_folder = "./chest_scan_data/"
split_folders = ["train", "test", "valid"]
outcome_folders = ["/large.cell.carcinoma", "/normal", ]
#"/squamous.cell.carcinoma" "/adenocarcinoma",

#%%
def img_preprocess(root_folder=root_folder, split_folders=split_folders, outcome_folders=outcome_folders):
    data_list = []
    for i in split_folders:
        imgs = []
        values = []
        for j in outcome_folders:
            folder_concat = root_folder + i + j
            files = os.listdir(folder_concat)
            for file in files:
                im = Image.open(f"{folder_concat}/{file}")
                image_array  = tf.convert_to_tensor(im, dtype=tf.float32)
                imgs.append(image_array)

                if j != "/normal":
                    values.append(0)
                else:
                    values.append(1)
        data_list.append(pd.DataFrame({"images": imgs, "tumor": values}))

    return data_list

#%%
def determine_dim_resize(root_folder=root_folder, split_folders=split_folders, outcome_folders=outcome_folders):
    image_dict = {}
    for i in split_folders:
        for j in outcome_folders:
            folder_concat = root_folder + i + j
            files = os.listdir(folder_concat)
            for file in files:
                im = Image.open(f"{folder_concat}/{file}")
                width, height = im.size
                image_dict[str(file)] = (width, height)

    widths = [v[0] for v in image_dict.values()]
    heights = [v[1] for v in image_dict.values()]
    filenames = image_dict.keys()

    image_dims = pd.DataFrame(data={'FileName': filenames, 'Width': widths, 'Height': heights})

    #graph for determining image dimensions to resize
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    points = ax.scatter(image_dims.Width, image_dims.Height, color='blue', alpha=0.5, picker=True)
    ax.set_title("Image Dimensions")
    ax.set_xlabel("Width", size=14)
    ax.set_ylabel("Height", size=14)
    return



# %%
