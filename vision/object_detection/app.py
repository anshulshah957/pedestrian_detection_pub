from flask import Flask, jsonify, request
import cv2
import numpy as np
import os
import subprocess

os.system("tar -zxvf darknet.tar.gz")
#os.system("sudo rm /var/lib/apt/lists/lock")
#os.system("sudo apt-get install build-essential")
cwd = os.getcwd()
try:
    os.chdir("./darknet")
    os.system("wget 'https://pjreddie.com/media/files/yolov3.weights'")
    os.system("make")
finally:
    os.chdir("..")
# process2 = subprocess.Popen(["wget", "https://pjreddie.com/media/files/yolov3.weights"], cwd = cwd + '/darknet')
# process2.wait()
# process = subprocess.Popen(["make"], cwd = cwd + '/darknet')
# process.wait()

from darknet.python import darknet as dn
import pdb
net = dn.load_net("darknet/cfg/yolov3.cfg".encode("utf-8"), "darknet/yolov3.weights".encode("utf-8"), 0)
meta = dn.load_meta("darknet/cfg/coco.data".encode("utf-8"))

image = cv2.imread("darknet/data/dog.jpg")
result = dn.detect(net, meta, image)
print(result)

app = Flask(__name__)
net = 0
meta = 0

def load_model():
    pass
    """Load and return the model"""
    # TODO: INSERT CODE
    # return model
# The request method is POST (this method enables your to send arbitrary data to the endpoint in the request body, including images, JSON, encoded-data, etc.)
@app.route('/', methods=["POST"])
def evaluate():
    r = request
    thisNet = net
    thisMeta = meta
    thisImage = image
    # print("0")
    # r = request
    # print(r.data)
    # print("1")
    # nparr = np.fromstring(r.data, np.uint8)
    # print("2")
    # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # print("3")
    toRet = dn.detect(thisNet, thisMeta, thisImage)
    # print("4")
    # print(toRet)
    # print("5")
    return "Hi tim"

if __name__ == "__main__":
    print("* Starting web server... please wait until server has fully started")
    app.run(host='0.0.0.0', threaded=True)