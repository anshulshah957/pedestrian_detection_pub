import os
from vision.object_detection.darknet.python import darknet as dn
import pdb
import cv2

def pedestrians_and_cars(frame, net, meta):
	r = dn.detect(net, meta, frame)
	return r

if __name__ == "__main__":
	# specify absolute pathes if not working
	net = dn.load_net("vision/object_detection/darknet/cfg/yolov3.cfg".encode("utf-8"), "vision/object_detection/darknet/yolov3.weights".encode("utf-8"), 0)
	meta = dn.load_meta("vision/object_detection/darknet/cfg/coco.data".encode("utf-8"))

	# numpy array example on a single image
	image = cv2.imread("vision/object_detection/darknet/data/dog.jpg")
	
	result = pedestrians_and_cars(image, net, meta)

	f = open("output.txt", "w")
	f.write(repr(result))
