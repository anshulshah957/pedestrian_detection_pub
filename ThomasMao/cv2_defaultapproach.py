# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 19:31:36 2018

@author: 13015
"""
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
hog=cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    orginalcopy=imutils.resize(frame,width=min(400,frame.shape[1]))
    (rects,weights)=hog.detectMultiScale(orginalcopy,winStride=(4,4),padding=(8,8),scale=1.05)
    rects=np.array([[x,y,x+w,y+h] for (x,y,w,h) in rects])
    windows=non_max_suppression(rects,probs=None,overlapThresh=0.65)
    for (xa,ya,xb,yb) in windows:
        cv2.rectangle(orginalcopy,(xa,ya),(xb,yb),(0,255,0),2)
    cv2.imshow("webcam",orginalcopy)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

