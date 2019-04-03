import glob,cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os.path as path


# read all images
imgs = []
imgids = []
for img in glob.glob("./dataset/*.png"):
    imgids.append(int(path.splitext(path.basename(img))[0]))
    n = cv2.imread(img,0)
    imgs.append(n)

# np array of images
# print shape of image array
imgids,imgs = zip(*sorted(zip(imgids,imgs)))
imgs = np.array(imgs)
imgids = np.array(imgids)
print(imgids.shape)
print(imgs.shape)

# read all subjective scores
subjective_scores = np.array(pd.read_excel("./dataset/DMOS_DIBR.xlsx", index_col=None, header=None))[:,0]
print(subjective_scores.shape)

# numpy array to store all objective scores
objective_scores = np.zeros(subjective_scores.shape)
print(objective_scores.shape)

kernel = np.ones((5,5),np.uint8)
for i in range(imgs.shape[0]):
    erosion = cv2.erode(imgs[i],kernel,iterations = 1)
    # filtered = cv2.bilateralFilter(imgs[i],3,7,7)
    temp = imgs[i]-erosion
    # plt.imshow(temp,"gray")
    # plt.show()
    # break
    ret,threshImg = cv2.threshold(temp,65,255,cv2.THRESH_BINARY_INV)
    objective_scores[i] = np.mean(threshImg)
    # print(objective_scores[i])

result = np.corrcoef(subjective_scores, objective_scores)[0, 1]
print(result)
