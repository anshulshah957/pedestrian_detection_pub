from vision.lane_detection.lane_detection import main as main_lane
from vision.traffic_light.trafficLight import getCircle as traffic_data
from vision.lane_detection.lane_detection import intersect_lines
from pynput.keyboard import Key, Controller

from vision.object_detection.darknet.python import darknet as dn
import pdb

import cv2
import numpy as np
import time

moving = False
speedTime = 0

keyboard = Controller()
#TODO: Find good values for paramters
STOP_TIME_BUFFER = 0.1
SHORT_BRAKE_TIME = 0.050
SHORT_ACCEL_TIME = 0.050
SHORT_TURN_TIME = 0.050
SPEED_CAP = 2

def pedestrians_and_cars(frame):
    # change location of files if not working
    net = dn.load_net("vision/object_detection/darknet/cfg/yolov3.cfg".encode("utf-8"), "vision/object_detection/darknet/yolov3.weights".encode("utf-8"), 0)
    meta = dn.load_meta("vision/object_detection/darknet/cfg/coco.data".encode("utf-8"))

    r = dn.detect(net, meta, frame)
    return r

#TODO: Make sure poly_left and poly_right map y->x and not x->y because otherwise this is all wrong
#TODO: Integrate traffic-light detection
def main(frame):
	distancePast = None

	#Start loop here
	ped_and_car_info = pedestrians_and_cars(frame)
	traffic_lights = traffic_data(frame)
	poly_left, poly_right = main_lane(frame)
	#Can get this once and export it to outside loop
	frame_height = frame.size(0)
	#TODO: Find better ranges: top means EMERGENCY STOP, bottom should not be outside of lane accuracy range
	xLeftDown = poly_left(0.55 * frame_height)
	xRightDown = poly_right(0.55 * frame_height)
	xLeftUp = poly_left(0.9 * frame_height)
	xRightUp = poly_right(0.9 * frame_height)
	isTop = False
	isBottom = False
	moving = False
	distancePast = distance
	distanceList = []
	for box in ped_and_car_info:
		centerX = box[2][0] + box[2][2] // 2
		centerY = box[2][1] + box[2][3] // 2

		action_classes = ["person", "car", "motorbike", "bus"]
		for name in action_classes:
			if box[0] == name:
				continue

		if centerY in range(0.5 * frame_height,0.65 * frame_height):
			if centerX in range(xLeftDown,xRightDown):
				isBottom = True
				distanceList.append((frame_height - centerY))
		if centerY in range(0.65 * frame_height, frame_height):
			if centerX in range(xLeftUp,xRightUp):
				isTop = True
				distanceList.append((frame_height - centerY))
	try:
		distance = min(distanceList)
	except:
		distance = frame_height
	move(isTop, isBottom, moving, poly_left, poly_right, distance, distancePast)

def adjust_direction(poly_left,poly_right,frame_width):
	res = intersect_lines(poly_left, poly_right)
	x = res[0]
	middle_x = frame_width // 2

	if x == middle_x:
		pass
	elif x < middle_x:
		keyboard.press('a')
		time.sleep(SHORT_TURN_TIME)
		keyboard.release('a')
	else:
		keyboard.press('d')
		time.sleep(SHORT_TURN_TIME)
		keyboard.release('d')

def move(isTop, isBottom, moving, poly_left, poly_right, distance, distancePast):
	if (isTop):
		stop()
		moving = False
		return
	adjust_direction(poly_left, poly_right)
	if (distancePast > distance):
		keyboard.press(Key.space)
		time.sleep(SHORT_BRAKE_TIME)
		keyboard.release(Key.space)
		speedTime -= SHORT_BRAKE_TIME
	if (distance > distancePast):
		if (speedTime >= SPEED_CAP):
			return
		keyboard.press('w')
		time.sleep(SHORT_ACCEL_TIME)
		keyboard.release('w')
		speedTime += SHORT_ACCEL_TIME

def stop():
	keyboard.press(Key.space)
	time.sleep(speedTime + STOP_TIME_BUFFER)
	keyboard.release(Key.space)


if __name__ == "__main__":
    pass
    #main()

