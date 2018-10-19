# -*- coding: utf-8 -*-
"""
To run the code , you must download INRIAPeron data set and put this .py file to that
directory. Create a another file to store the trimmed train data. The trim data function is 
provided below. Change all the directory to your own settings!!!
"""

import time
#from PIL import Image
#from pathlib import Path
import glob
import imageio
import numpy as np
#from matplotlib import pyplot as plt
from skimage.feature import hog
#import math
import cv2
#from sklearn.decomposition import PCA might be used later
#from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.externals import joblib
#from numba import vectorize,cuda might be used later to accelerate calculation
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
#from sklearn.multiclass import OneVsRestClassifier 

#get all the window pictures of 70x134 size of a picture(still has bug...)
def windowpic(inputpicmat):
    retmat=np.zeros((inputpicmat.shape[0],inputpicmat.shape[1],3),dtype=np.uint8)
    for i in range(0,inputpicmat.shape[0]-134+1):
        for j in range(0,inputpicmat.shape[1]-70+1):
            window=np.zeros((134,70,3),dtype=np.uint8)
            window=inputpicmat[i:i+134,j:j+70,0:3]
            clftest=[]
            clftest.append(hog(window,orientations=8,pixels_per_cell=(16,16),cells_per_block=(1,1),transform_sqrt=True,multichannel=True,block_norm='L1'))
            if (clf.predict(clftest)[0]==1):
                retmat[i:i+134,j:j+70,0]=254
    return retmat


#Function for trimming a photo as a numpy matrix into a 70x134 submatrix locates at the center of that input matrix
def getim(inputMat):
    height_upper_bound=int((len(inputMat)-1)/2)-66
    height_lower_bound=height_upper_bound+134-1
    width_lower_bound=int((len(inputMat[0])-1)/2)-34
    width_upper_bound=width_lower_bound+69
    return inputMat[height_upper_bound:height_lower_bound+1,width_lower_bound:width_upper_bound+1,0:3]

'''

for picid,filename in enumerate(glob.glob('./Train/neg/*.png')):
    im=imageio.imread(filename)
    im=getim(im)
    imageio.imwrite('./resizeddata/neg_70x134_three2/'+str(picid)+'.png',im)
'''
#import RandomForestClassifier which takes much less time than linear svm, but produces nearly the same accuracy
clf=RandomForestClassifier(min_samples_leaf=30)

#read the train data hog_features is a list that restore all the hog feature vector
hog_features=[]
hog_labels=[]
start=time.time()
for filename in glob.glob('./test_64x128_H96/pos/*.png'):
    image_pos=imageio.imread(filename)
    image_pos=image_pos[:,:,0:3]
    hog_features.append(hog(image_pos,orientations=8,pixels_per_cell=(32,32),cells_per_block=(1,1),transform_sqrt=True,multichannel=True,block_norm='L1'))
    hog_labels.append([1])
for filename in glob.glob('./resizeddata/neg_70x134_three/*.png'):
    image_neg=imageio.imread(filename)
    image_neg=image_neg[:,:,0:3]
    hog_features.append(hog(image_neg,orientations=8,pixels_per_cell=(32,32),cells_per_block=(1,1),transform_sqrt=True,multichannel=True,block_norm='L1'))
    hog_labels.append([0])
for filename in glob.glob('./resizeddata/neg_70x134_three2/*.png'):
    image_neg2=imageio.imread(filename)
    image_neg2=image_neg2[:,:,0:3]
    hog_features.append(hog(image_neg2,orientations=8,pixels_per_cell=(32,32),cells_per_block=(1,1),transform_sqrt=True,multichannel=True,block_norm='L1'))
    hog_labels.append([0])
end=time.time()
print(end-start)
#combining the hog_features and hog_labels into dataframes in order to shuffle 
#it before training(very important because we are using RandomForestClassifier 
#which does not take all the train data)
df=np.hstack((hog_features,hog_labels))
np.random.shuffle(df)


#Training the data and prints out the training time
start=time.time()
clf.fit(df[:,0:64],df[:,64])
end=time.time()
print(end-start)


