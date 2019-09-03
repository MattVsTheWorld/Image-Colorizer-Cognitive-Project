import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
import os
from tqdm import tqdm
from skimage import color

matplotlib.interactive(True)

# Number of discrete regions of LAB colorspace

color_map = np.zeros((256, 256))
mydir = os.pardir + '/test_images/bird'

for name in tqdm(os.listdir(mydir)):
    # print(name)
    img = cv2.cvtColor(cv2.imread(mydir + "/" + name), cv2.COLOR_BGR2RGB)
    img = color.rgb2lab(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            color_map[int(img[i, j, 1]) + 127, int(img[i, j, 2]) + 127] += 1


np.save("color_distribution", color_map)

print("saved")
