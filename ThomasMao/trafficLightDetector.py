# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 14:19:32 2018

@author: 13015
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

def detectcircle(inputMat):
    imasgray=cv2.cvtColor(inputMat, cv2.COLOR_BGR2GRAY)
    imasgray = cv2.medianBlur(imasgray,5)
    circles = cv2.HoughCircles(imasgray,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)
    return circles
def cvtColor(inputMat):
    out=cv2.cvtColor(inputMat, cv2.COLOR_BGR2GRAY)
    return out

im=cv2.imread('./download4.png')    

circles=detectcircle(im)
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(im,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(im,(i[0],i[1]),2,(0,0,255),3)

plt.imshow(im)
 
