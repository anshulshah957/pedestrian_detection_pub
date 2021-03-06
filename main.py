from vision.lane_detection.lane_detection import main as main_lane
from vision.traffic_light.trafficLight import getCircle as traffic_data
#from vision.lane_detection.lane_detection import intersect_lines
from pynput.keyboard import Key, Controller
from directkeys import PressKey,ReleaseKey, W, A, S, D
#from vision.object_detection.darknet.python import darknet as dn
from svm_pipeline import *
from yolo_pipeline import *
from lane import *
import keyboard
import pdb
import cv2
import numpy as np
import time
from PIL import Image
from mss import mss
from PIL import ImageGrab
import pyautogui
moving = False
distancePast = None
speedTime = 0
#TODO: Find good values for paramters
STOP_TIME_BUFFER = 0.1
SHORT_BRAKE_TIME = 0.050
SHORT_ACCEL_TIME = 0.050
SHORT_TURN_TIME = 0.050
SPEED_CAP = 2

def pipeline_yolo(img):

    img_undist, img_lane_augmented, lane_info = lane_process(img)
    output = vehicle_detection_yolo(img_undist, img_lane_augmented, lane_info)

    return output

def pipeline_svm(img):

    img_undist, img_lane_augmented, lane_info = lane_process(img)
    output = vehicle_detection_svm(img_undist, img_lane_augmented, lane_info)
    return output

'''
def pedestrians_and_cars(frame, net, meta):
	r = dn.detect(net, meta, frame)
	return r
'''
#TODO: Make sure poly_left and poly_right map y->x and not x->y because otherwise this is all wrong
#TODO: Integrate traffic-light detection
def main(frame, net, meta):
	#Start loop here
	ped_and_car_info = pedestrians_and_cars(frame, net, meta)
	print("detecting")
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
		
		var_continue = True
		action_classes = ["person", "car", "motorbike", "bus"]
		for name in action_classes:
			if box[0] == name:
				var_continue = False
				break
	
		if var_continue == True:
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
	move(isTop, isBottom, moving, poly_left, poly_right, distance, distancePast, frame_width)

def adjust_direction(poly_left, poly_right, frame_height, frame_width):
    keyboard = Controller()
    y_index = int(frame_height*(3/4))
    x_left = poly_left(y_index)
    x_right = poly_right(y_index)
    #
    x_rel = int((x_left + x_right)/2)
    x_rel_cen = int(frame_width/2) - x_rel
    print(x_rel_cen)
    if abs(x_rel_cen) > 400:
        if x_rel_cen < 0:
            x_rel_cen = -400
        else :
            x_rel_cen=400
    if x_rel_cen < 0:
        print('d is pressed')
        i = 0
        while i < (abs(x_rel_cen)/200):
            
            # print("here")
            keyboard.press('d')
            # print("here2")
            keyboard.release('d')
            # print("here3")
            i = i + 1
            time.sleep(.500)
        
    elif x_rel_cen > 0:
        print('a is pressed')
        i = 0
        while i < abs(x_rel_cen)/200:
            # print("here4")
            keyboard.press('a')
            # print("here5")
            keyboard.release('a')
            # print("here6")
            i = i + 1
            time.sleep(.500)
    # res = intersect_lines(poly_left, poly_right)
	#x = res[0]
	#middle_x = frame_width // 2
    '''
	if x == middle_x:
		pass
	elif x < middle_x:
		keyboard.press('a')
		time.sleep(SHORT_TURN_TIME)
		keyboard.release('a')aq
	else:
		keyboard.press('d')d 
		time.sleep(SHORT_TURN_TIME)
		keyboard.release('d')
    '''
def move(isTop, isBottom, moving, poly_left, poly_right, height, width):
	'''
    if (isTop):
		stop()
		moving = False
		return
    '''
	adjust_direction(poly_left, poly_right, height, width)
	'''
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
    '''

def stop():
	keyboard.press('s')
	time.sleep(speedTime + STOP_TIME_BUFFER)
	keyboard.release('s')
'''
def captureVideo(nameofthevideo):
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(nameofthevideo,fourcc, 20.0, (1920,1080))
    while True:
        img = ImageGrab.grab()
        img=np.array(img)
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(frame)
        cv2.imshow('Screen',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    out.release()
    cv2.destroyAllWindows()
'''


def main():
    keyboard=Controller()
    '''
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter("test1.mp4",fourcc, 20.0, (1920,1080))
    while True:
        monitor = mss().monitors[1]
        sct_img = mss().grab(monitor)
        cv2.waitKey(10)
        img = Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')
        img = np.array(img)
		#main(img, net, meta)
	
    	#cap = cv2.VideoCapture('pov.mp4')
	    #while(cap.isOpened()):
		#ret, frame = cap.read()
        
        poly_left, poly_right, lane_detection_image = main_lane(img)
        cv2.imshow('frame',lane_detection_image)
        out.write(lane_detection_image)
        move(False, False, False, poly_left, poly_right,img.shape[0], img.shape[1])
		# trafficlights = traffic_data(frame);
        # print(trafficlights)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    out.release()
    cv2.destroyAllWindows()
    '''
    
        
        
		

if __name__ == "__main__":
    
    cap = cv2.VideoCapture(0)
    while(True):
    # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720))
        frame = pipeline_yolo(frame)
    
        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
    '''
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter("deepdrivetry1.mp4",fourcc, 20.0, (1280,720))
    for i in range(4):
        print(i)
        time.sleep(1)
    keyboard = Controller()
    while True:
        frame = np.array(ImageGrab.grab(bbox=(0,40,1280,720)))
        frame, poly_left, poly_right = main_lane(frame)
        y_index = int(frame.shape[0]*(3/4))
        x_left = poly_left(y_index)
        x_right = poly_right(y_index)
        #
        x_rel = int((x_left + x_right)/2)
        x_rel_cen = int(frame.shape[1]/2) - x_rel
        print(x_rel_cen)
        if abs(x_rel_cen) > 400:
            if x_rel_cen < 0:
                x_rel_cen = -400
            else :
                x_rel_cen=400
        
        if x_rel_cen < -50:
            print('d is pressed')
            keyboard.press('a')
            time.sleep(0.0006*abs(x_rel_cen))
            keyboard.release('a')
            
            i = 0
            while i < (abs(x_rel_cen)/200):
                
                # print("here")
                keyboard.press('a')
                # print("here2")
                keyboard.release('a')
                # print("here3")
                i = i + 1
                time.sleep(.500)
                    
        if x_rel_cen >= -50:
            print('a is pressed')
            keyboard.press('d')
            time.sleep(0.0006*abs(x_rel_cen))
            keyboard.release('d')
            
            while i < abs(x_rel_cen)/200:
                # print("here4")
                keyboard.press('a')
                # print("here5")
                keyboard.release('a')
                # print("here6")
                i = i + 1
                time.sleep(.500)
            
        cv2.imshow('window', frame)
        out.write(frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    out.release()
    cv2.destroyAllWindows()
    '''
