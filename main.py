import glob, cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os.path as path
from scipy.optimize import curve_fit
from pandas.plotting import table

# read all images
imgs = []
rgbimgs = []
imgids = []
for img in glob.glob("./dataset/*.png"):
    imgids.append(int(path.splitext(path.basename(img))[0]))
    n = cv2.imread(img)
    rgbimgs.append(n)
    n2 = cv2.cvtColor(n, cv2.COLOR_BGR2GRAY)
    imgs.append(n2)

# np array of images
# print shape of image array
imgids,imgs = zip(*sorted(zip(imgids,imgs)))
# imgids, rgbimgs = zip(*sorted(zip(imgids,rgbimgs)))
imgs = np.array(imgs)
imgids = np.array(imgids)
# rgbimgs = np.array(rgbimgs)
print(imgids.shape)
print(imgs.shape)
# print(rgbimgs.shape)

# read all subjective scores
subjective_scores = np.array(pd.read_excel("./dataset/DMOS_DIBR.xlsx", index_col=None, header=None),dtype=np.float64)[:,0]
print(subjective_scores.shape)

# numpy array to store all objective scores
objective_scores = np.zeros(subjective_scores.shape, dtype=np.float64)
print(objective_scores.shape)

kernel = np.ones((3, 3),np.float32)
lowpass = np.ones((3, 3),np.float32)/9

def operation(name, img):
    operationDict = {
        "median" : cv2.medianBlur(img, 3),
        "erosion" : cv2.erode(img,kernel, iterations = 1),
        "filtered" : cv2.bilateralFilter(img,3,7,7),      
        "opening" : cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel),
        "closing" : cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel),
        "gradient" : cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel),
        "tophat" : cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel),
        "blackhat" : cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel),
        "laplacian" : cv2.Laplacian(img,cv2.CV_64F),
        "sobelx" : cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3),
        "sobely" : cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3),
        "sobelxy" : cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3),
    }
    return operationDict[name]
        
# opertationsNameList = ["median", "erosion", "opening", "closing", "gradient", "tophat", "blackhat","laplacian", "sobelx", "sobely"]
opertationsNameList = ["blackhat"]
operationBestResult = [0]*len(opertationsNameList)

for j in range(len(opertationsNameList)):
    best = 0
    wait = 0
    print("=========================", opertationsNameList[j], "=============================")
    for k in range(0,255):
        if wait > 15:
            break

        for i in range(imgs.shape[0]):
            temp = np.array(imgs[i], dtype=np.float32) - np.array(operation(opertationsNameList[j], imgs[i]), dtype=np.float32)
            temp = np.abs(temp)
            # temp = cv2.filter2D(temp,-1,lowpass)

            ret,threshImg = cv2.threshold(temp, k, 255, cv2.THRESH_BINARY)
            objective_scores[i] =  np.std(threshImg)
        
        result = np.corrcoef(subjective_scores, objective_scores)[0, 1]
        
        if(best < np.abs(result)):
            wait = 0
            best = np.abs(result)
            print("Current Best: ", k, best)
        else:
            wait += 1
    operationBestResult[j] = best


outputTable = pd.DataFrame(data={'operation':opertationsNameList, "Best PLCC": operationBestResult})

ax = plt.subplot(111, frame_on=False) # no visible frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis
table(ax, outputTable)  # where df is your data frame
# plt.show()