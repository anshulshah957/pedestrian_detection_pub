#from .vision.lane_detection.lane_detection import main as main_lane
#from vision.traffic_light.trafficLight import getCircles as traffic_data

from vision.object_detection.darknet.python import darknet as dn
import pdb

moving = False
def lane_detect_data(frame):
	return main_lane(frame)
def traffic_light_data(frame):
	return traffic_data(frame)
def pedestrians_and_cars(frame):
    # change location of files if not working
    net = dn.load_net("vision/object_detection/darknet/cfg/yolov3.cfg".encode("utf-8"), "vision/object_detection/darknet/yolov3.weights".encode("utf-8"), 0)
    meta = dn.load_meta("vision/object_detection/darknet/cfg/coco.data".encode("utf-8"))

    #expects image as a file location!!!
    r = dn.detect(net, meta, frame)
    return r

# example
print(pedestrians_and_cars("vision/object_detection/darknet/data/dog.jpg".encode("utf-8")))

def main(frame):
	ped_and_car_info = pedestrians_and_cars(frame)
	traffic_lights = traffic_light_data(frame)
	frame, poly_left, poly_right = lane_detect_data(frame)
	frame_height = len(frame[0])
	func1 = poly_left
	func2 = poly_right
	xLeftDown = poly_left(0.5 * frame_height)
	xRightDown = poly_right(0.5 * frame_height)
	xLeftUp = poly_left(0.65 * frame_height)
	xRightUp = poly_right(0.65 * frame_height)
	isTop = False
	isBottom = False
	moving = False
	for box in ped_and_car_info:
		centerX = box[0][0] + box[2] // 2
		centerY = box[0][1] + box[1] // 2
		if centerY in range(0.45 * frame_height,0.55 * frame_height):
			if centerX in range(xLeftDown,xRightDown):
				isBottom = True
		if centerY in range(0.5 * frame_height, frame_height):
			if centerX in range(xLeftUp,xRightUp):
				isTop = True
	move(isTop, isBottom, moving)
def get_direction(poly_left,poly_right):
    pass
def move(isTop, isBottom, moving, poly_left, poly_right):
	if (isTop):
		Stop()
		moving = False
	elif(not moving):
		direction = get_direction(poly_left,poly_right)



if __name__ == "__main__":
    pass
    #main()

