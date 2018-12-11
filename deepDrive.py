from mss import mss
import numpy as np
from PIL import Image
from vision.lane_detection.lane_detection import main as main_lane
from pynput.keyboard import Key, Controller
if __name__ == "__main__":
	while True:
		monitor = mss().monitors[1]
		sct_img = mss().grab(monitor)
		cv2.waitKey(10)
		img = Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')
		img = np.array(img)