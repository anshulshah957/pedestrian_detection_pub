import cv2
import numpy as np
from statistics import mode

# Some help from https://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/

def threshold(img):
	# cv2.imshow('image',img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	redM1 = img[:,:,0] < 100
	redM1.astype(np.int)
	redM2 = img[:,:,1] < 100
	redM2.astype(np.int)
	redM3 = img[:,:,2] > 180
	redM3.astype(np.int)
	redArr = np.multiply(np.multiply(redM1,redM2),redM3) * 240
	greenM1 = img[:,:,0] > 70
	greenM1.astype(np.int)
	greenM2 = img[:,:,1] > 170
	greenM2.astype(np.int)
	greenM3 = img[:,:,2] < 100
	greenM3.astype(np.int)
	greenArr = np.multiply(np.multiply(greenM1,greenM2),greenM3) * 160
	yellowM1 = img[:,:,0] < 110
	yellowM1.astype(np.int)
	yellowM2 = img[:,:,1] > 170
	yellowM2.astype(np.int)
	yellowM3 = img[:,:,2] > 170
	yellowM3.astype(np.int)
	yellowArr = np.multiply(np.multiply(yellowM1,yellowM2),yellowM3) * 80
	retArr = np.add(np.add(redArr,greenArr),yellowArr)
	# cv2.imshow('l' , np.array(retArr, dtype = np.uint8))
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	return np.array(retArr, dtype = np.uint8)
def getCircle(img):
	img = threshold(img)
	circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 75, param2 = 3, maxRadius = 15)
	retList = []
	if circles is not None:
		circles = np.round(circles[0, :]).astype("int")
		for (x,y,r) in circles:
			# print(img[y,x])
			# print(r)
			# print("__________")
			if (not(img[y,x] == 0)):
				retList.append(img[y,x])
		try:
			return mode(retList)
		except:
			try:
				return retList[0]
			except:
				return 0
	return 0
	# cv2.imshow("output", np.hstack([img, output]))
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

#img = cv2.imread('GTARed1.jpg',1)
#print(getCircle(img))



	