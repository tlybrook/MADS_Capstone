#%%
from PIL import Image
#import cv2
import os
import imagesize
import pandas as pd
import matplotlib.pyplot  as plt

#image = cv2.imread(f"{image_path}/{files[0]}")
#cv2.imshow('color image', image)

#cv2.waitKey(0) 
  
# closing all open windows 
#cv2.destroyAllWindows()

# print(im.format, im.size, im.mode)
# im.show()

#%%
image_paths = ["./chest_scan_data/train/adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib",
               "./chest_scan_data/train/large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa",
               "./chest_scan_data/train/normal",
               "./chest_scan_data/train/squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa"]

image_dict = {}
for i in image_paths:
    files = os.listdir(i)
    for file in files:
        im = Image.open(f"{i}/{file}")
        width, height = im.size
        image_dict[str(file)] = (width, height)

widths = [v[0] for v in image_dict.values()]
heights = [v[1] for v in image_dict.values()]
filenames = image_dict.keys()

image_dims = pd.DataFrame(data={'FileName': filenames, 'Width': widths, 'Height': heights})

print(image_dims)

#%%
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
points = ax.scatter(image_dims.Width, image_dims.Height, color='blue', alpha=0.5, picker=True)
ax.set_title("Image Dimensions")
ax.set_xlabel("Width", size=14)
ax.set_ylabel("Height", size=14)


# %%
