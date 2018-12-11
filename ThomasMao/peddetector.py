# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 15:37:03 2018

@author: 13015
"""
from imutils.object_detection import non_max_suppression
import numpy as np
#import imutils
import cv2
from skimage.feature import hog
hog2=cv2.HOGDescriptor()
hog2.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
def detectped_BGRinput(originalpic):
    originalcopy=originalpic.copy()
    (rects,weights)=hog2.detectMultiScale(originalcopy,winStride=(8,8),padding=(16,16),scale=1.05)
    rects=np.array([[x,y,x+w,y+h] for (x,y,w,h) in rects])
    windows=non_max_suppression(rects,probs=None,overlapThresh=0.65)
    for (xa,ya,xb,yb) in windows:
        cv2.rectangle(originalcopy,(xa,ya),(xb,yb),(0,255,0),2)
    return originalcopy