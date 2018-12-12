from mss import mss
import numpy as np
from PIL import Image
import time
from vision.lane_detection.lane_detection import main as main_lane
from pynput.keyboard import Key, Controller
keyboard = Controller()
COEF = .001
if __name__ == "__main__":
	while True:
		monitor = mss().monitors[1]
		sct_img = mss().grab(monitor)
		#cv2.waitKey(10)
		img = Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')
		img = np.array(img)
		main(img)
def main(img):
	poly_left, poly_right, min_y, max_y = main_lane(img)
	
	width = img.shape[1]
	height = img.shape[0]
	checkY = (max_y + min_y) // 2
	leftX = poly_left(checkY)
	rightX = poly_right(checkY)
	#leftLowX = poly_left(min_y)
	#rightLowX = poly_right(min_y)
	centerX = width // 2
	leftDif = abs(centerX - leftX)
	rightDif = abs(centerX - rightX)
	
	if poly_left == None and poly_right == None:
		continue
	elif poly_left == None:
		magnitude = rightDif
		keyboard.press('d')
		time.sleep(COEF * magnitude)
		keyboard.release('d')
		continue
	elif poly_right == None:
		magnitude = leftDif
		keyboard.press('a')
		time.sleep(COEF * magnitude)
		keyboard.release('a')
		continue

	if (centerX > leftX and centerX < rightX):
		continue
	if (centerX < leftX):
		magnitude = leftDif
		keyboard.press('d')
		time.sleep(COEF * magnitude)
		keyboard.release('d')
		continue
	if (centerX > rightX):
		magnitude = rightDif
		keyboard.press('a')
		time.sleep(COEF * magnitude)
		keyboard.release('a')
		continue



