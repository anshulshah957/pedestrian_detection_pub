# probably a bad idea. See detector.py in ./python
import sys, os
sys.path.append(os.path.join(os.getcwd(),'python/'))

import darknet as dn
import pdb

# change location of yolov3.weights
net = dn.load_net("cfg/yolov3.cfg", "/Users/timbaer/cs196/pedestrian_detection/darknet/yolov3.weights", 0)
meta = dn.load_meta("cfg/coco.data")

def get_data(image):
    r = dn.detect(net, meta, image)
    print r
