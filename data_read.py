from PIL import Image
import cv2
import os

image_path = "./chest_scan_data/train/adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib"
files = os.listdir(image_path)

image = cv2.imread(f"{image_path}/{files[0]}")
cv2.imshow('color image', image)

cv2.waitKey(0) 
  
# closing all open windows 
cv2.destroyAllWindows()

# for file in files:
#     im = Image.open(f"{image_path}/{file}")
#     print(im.size)

# print(im.format, im.size, im.mode)
# im.show()
