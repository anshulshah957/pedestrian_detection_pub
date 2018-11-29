import cv2
import numpy as np
import glob
import csv

#Some help from https://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/

def threshold(img):
	cv2.imshow('imafge',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	newImg = img[:,:,0]
	for i in range(0,img.shape[0]):
		for j in range(0,img.shape[1]):
			pixSum = int(img[i,j,0]) + int(img[i,j,1]) + int(img[i,j,2])
			blue = img[i,j,0]
			green = img[i,j,1]
			red = img[i,j,2]
			#threshold
			isRed = (red > 165 and green < 100 and blue < 100)
			isGreen = (green > 170 and blue > 70 and red < 100)
			isYellow = (red > 170 and green > 170 and blue < 110)
			if (isRed):
				newImg[i,j] = 240
			elif (isGreen):
				newImg[i,j] = 160
			elif (isYellow):
				newImg[i,j] = 80
			else:
				newImg[i,j] = 0
	cv2.imshow('image',newImg)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return newImg
def getCircle(img):
	circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 75, param2 = 10)
	output = img.copy()
	retList = []
	if circles is not None:
		circles = np.round(circles[0, :]).astype("int")
		for (x,y,r) in circles:
			print(x)
			print(y)
			print(img[y][x])
			print(r)
			retList.append(img[y][x])
			cv2.circle(output, (x, y), r, (0, 255, 0), 4)
			cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
	return retList
	cv2.imshow("output", np.hstack([img, output]))
	cv2.waitKey(0)
	cv2.destroyAllWindows()


img = cv2.imread('GTAYellowAndRed1.jpg',1)
print(getCircle(threshold(img)))



	