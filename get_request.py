import requests
import json
import cv2

url = 

im = cv2.imread("vision/object_detection/darknet/data/dog.jpg")
# encode image as jpeg
_,im_encoded = cv2.imencode('.jpg', im)
# send http request with image and recieve and recieve response
response = requests.post(url, data=im_encoded.tostring())
# decode response
print(response.tostring)

