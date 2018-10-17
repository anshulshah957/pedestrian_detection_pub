import cv2
import numpy as np

vid = cv2.VideoCapture('clip_highway_video.mp4')

while vid.isOpened():
    ret, frame = vid.read()
    cv2.imshow('frame',frame)
    cv2.waitKey(1)

vid.release()
cv2.destroyAllWindows()
