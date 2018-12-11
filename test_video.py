from vision.object_detection.darknet.python import darknet as dn
import pdb
import cv2
import numpy as np

def pedestrians_and_cars(frame, net, meta):
	r = dn.detect(net, meta, frame)
	return r

if __name__ == "__main__":
	net = dn.load_net("vision/object_detection/darknet/cfg/yolov3.cfg".encode("utf-8"), "vision/object_detection/darknet/yolov3.weights".encode("utf-8"), 0)
	meta = dn.load_meta("vision/object_detection/darknet/cfg/coco.data".encode("utf-8"))
	
	# prints bounding box data for each object in frame
	cap = cv2.VideoCapture('test.mp4')
	while(cap.isOpened()):
		ret, frame = cap.read()
		
		boxes = pedestrians_and_cars(frame, net, meta)
		print('\n')
		print('\n')
		for box in boxes:
			print(box)
		print('\n')
		print('\n')
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()

