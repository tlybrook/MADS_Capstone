'''
This file cleans up the downloaded dataset from Kaggle and removes all duplicates.
'''
from PIL import Image, ImageOps
import os
import shutil
import pandas as pd
import matplotlib.pyplot  as plt
import numpy as np
from processes.visualization import image_dims_viz

def data_clean(data_path, folder_labels):
    """Remove duplications from the downloaded Kaggle dataset and save the non-duplicate images to a new
    folder to be ready for data splitting and preprocessing. Nothing is returned from this function, but the 
    new directory should be created and accessible in your working directory. 
    
    Parameters
    -----------
    data_path: The path to the downloaded Kaggle dataset.
    folder_labels: The labels of the folders for the new directory for non-duplicate images.
    """

    doop1_path = f"{data_path}/train/large.cell.carcinoma_left.hilum_T2_N2_M0_llla/14.png"
    doop2_path = f"{data_path}/train/squamous.cell.carcinoma_left.hilum_T1_N2_M0_llla/sq2.png"

    if os.path.exists(doop1_path):
        os.remove(doop1_path)
    
    if os.path.exists(doop2_path):
        os.remove(doop2_path)

    try:
        for i in folder_labels:
            os.makedirs(f"./final_dataset/{i}")
            print("Folder %s created!" % i)
    except FileExistsError:
        print("Folder %s already exists" % i)
        print("Oops! Please make sure to delete folders and try running again!")
        return
    
    imgs_dict = {}
    counter = 0
    imgs = []
    img_w_doop = []
    top_level_folders = sorted(os.listdir(data_path))
    for i in top_level_folders:
        folders = sorted(os.listdir(f"{data_path}/{i}"))
        for k in folders:
            count = 0
            files = sorted(os.listdir(f"{data_path}/{i}/{k}"))
            for file in files:
                img_w_doop.append(file)
                im = Image.open(f"{data_path}/{i}/{k}/{file}")
                new_image = im.resize((512, 512))
                image_array  = np.asarray(ImageOps.grayscale(new_image)).astype('float32')
                index = next(i for i, e in enumerate(folder_labels) if k[:6] in e)

                imgs_dict[counter] = (image_array, folder_labels[index], k, i, file)
                counter += 1

    doops = {}
    imgs = []
    for key, info in imgs_dict.items():
        if len(imgs) == 0:
            imgs.append(info[0])
        else:
            check_frame = pd.DataFrame(np.all(info[0] == imgs, axis=1))
            matches = check_frame.sum(axis=1)
            matches2 = matches.loc[lambda x: x > 250]
            if matches2.empty:
                imgs.append(info[0])
                continue
            else:
                doops[key] = matches2
    
    final_non_doops = {}
    dup_keys = []
    for key, info in imgs_dict.items():
        if key not in doops:
            final_non_doops[key] = info
        else:
            dup_keys.append(key)

    count = 0
    for key, info in final_non_doops.items():
        shutil.copyfile(f"{data_path}/{info[3]}/{info[2]}/{info[4]}", f"./final_dataset/{info[1]}/file{str(count)}.jpg")
        count += 1

    print(f"{len(final_non_doops)} files added")
    return

def determine_dim_resize(data_path='./final_dataset'):
    """Save the dimensions of each image from the non-duplicated dataset and generate a scatterplot
    visualization to determine optimal dimensions to resize images to.

    Parameters
    -----------
    data_path: The path to the non-duplicated dataset (by default this is ./final_dataset)
    """
    image_dict = {}
    top_level_folders = sorted(os.listdir(data_path))
    for i in top_level_folders:
        files = sorted(os.listdir(f"{data_path}/{i}/"))
        for file in files:
            im = Image.open(f"{data_path}/{i}/{file}")
            width, height = im.size
            image_dict[str(file)] = (width, height)

    widths = [v[0] for v in image_dict.values()]
    heights = [v[1] for v in image_dict.values()]
    filenames = image_dict.keys()

    image_dims = pd.DataFrame(data={'FileName': filenames, 'Width': widths, 'Height': heights})

    image_dims_viz(image_dims, "Image Size Plot")

    return

if __name__ == '__main__':
    data_path = "./Data"
    folder_labels = ["adenocarcinoma", 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']
    data_clean(data_path, folder_labels)
    determine_dim_resize()


